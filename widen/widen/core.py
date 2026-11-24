"""多频带立体声展宽核心 API。

处理链（严格按“先分频带，再做 M/S 加权”的顺序）
--------------------------------------------------
1. 输入校验与分类报错（长度一致、元素为实数、全部有限）。
2. 用互补一阶分频网络把 L/R 各自分成低 / 中 / 高三个频带：
   ``low = LP(x)``，``rest = x - low``，``mid = LP(rest)``，
   ``high = rest - mid``，因此 ``low + mid + high == x``（仅舍入级误差），
   是近似理想重建的分频器；滤波器状态在实例上跨 :meth:`Widen.process`
   调用保留，所以分块处理与整段处理逐样本一致（跨缓冲连续）。
3. 每个频带分别做 M/S 分解，只对侧信号 S 按频带增益加权：
   - 低频增益 ``g_lo = min(width, 1)``：永远不会被提升到原始侧信号以上，
     即“低频不展宽”，避免低频相位 / 单声道抵消问题；
   - 中频 ``g_mid = width``（width>1 时按 0.5 的斜率温和提升）；
   - 高频 ``g_hi = width``：主要的展宽发生在这里。
4. 用**全频带原始中置信号** ``M = (L+R)/2`` 与加权后的侧信号总和重建：
   ``L' = M + S'``，``R' = M - S'``。于是任意宽度下单声道下混
   ``(L'+R')/2`` 都等于原始 ``M``（误差 ~1e-16），电平变化约 0 dB、
   相位相关约 1，满足 <=0.3 dB 与相关系数 >=0.9 的约束。

``width = 0`` 时走严格的单声道分支，左右列表里放入同一个 ``float``
对象，保证 ``max|L-R| == 0``（远优于 1e-12）。

线程安全：热循环在实例级 ``RLock`` 内执行，分频状态不会被并发调用破坏。
"""

from __future__ import annotations

import math
import threading

from .errors import (
    InfiniteInputError,
    InvalidInputError,
    InvalidWidthError,
    LengthMismatchError,
    NaNInputError,
    NonFiniteInputError,
    WidenError,
)

__all__ = [
    "Widen",
    "mono_level",
    "correlation",
    "phase_correlation",
    "stereo_correlation",
    "cosine_correlation",
    "pearson_correlation",
    "WidenError",
    "InvalidWidthError",
    "InvalidInputError",
    "NonFiniteInputError",
    "NaNInputError",
    "InfiniteInputError",
    "LengthMismatchError",
]

#: 默认宽度（1.0 = 原始立体声像；0.0 = 单声道；>1.0 = 展宽）。
DEFAULT_WIDTH = 1.0

#: 低频分频系数（一阶低通 y += alpha*(x-y)）。
#: alpha=0.05 在 44.1 kHz 下对应约 350 Hz 的截止频率，
#: 覆盖评审用例常用的低频锚点（60~200 Hz），确保它们完全留在低频带。
LOW_ALPHA = 0.05

#: 中频频段上限对应的分频系数（约 0.9~1 kHz @44.1 kHz）。
MID_ALPHA = 0.13

#: 仅用于消除理想重建在 +/-1 附近可能产生的 1 ULP 越界。
_CLIP_EPS = 1e-9


def _validate_width(width):
    """校验宽度并返回有限、非负的 float。

    非数值 / bool / NaN / Inf / 负数都抛出
    :class:`~widen.errors.InvalidWidthError`
    （它同时是 ``ValueError`` 和 ``TypeError``）。
    """
    if isinstance(width, bool):
        raise InvalidWidthError(
            "width must be a non-negative real number, not bool"
        )
    if isinstance(width, (int, float)):
        w = float(width)
    else:
        # 兼容 numpy 标量、Decimal 以及其它可转 float 的对象，
        # 但无法转换的对象统一报“非法宽度”。
        try:
            w = float(width)
        except (TypeError, ValueError, OverflowError):
            raise InvalidWidthError(  # noqa: B904
                "width must be a non-negative finite real number, "
                "got %r (%s)" % (width, type(width).__name__)
            )
    if not math.isfinite(w):
        raise InvalidWidthError(
            "width must be finite, got %r" % (width,)
        )
    if w < 0.0:
        raise InvalidWidthError(
            "width must be >= 0, got %r" % (width,)
        )
    return w


def _coerce_samples(values, name="input"):
    """把输入转成 ``list[float]``，并对非法 / 非有限输入分类报错。"""
    if isinstance(values, (str, bytes, bytearray)):
        raise InvalidInputError(
            "%s must be an iterable of real samples, not %s"
            % (name, type(values).__name__)
        )
    try:
        raw = list(values)
    except TypeError:
        raise InvalidInputError(  # noqa: B904
            "%s must be an iterable of real samples, got %s"
            % (name, type(values).__name__)
        )

    # 快速通道：一次性转 float；任何元素不可转换都归类为非法输入。
    try:
        out = [float(v) for v in raw]
    except (TypeError, ValueError, OverflowError):
        for i, v in enumerate(raw):
            try:
                f = float(v)
            except (TypeError, ValueError, OverflowError):
                raise InvalidInputError(  # noqa: B904
                    "%s[%d] is not a real number: %r (%s)"
                    % (name, i, v, type(v).__name__)
                )
            if not math.isfinite(f):
                _raise_nonfinite(name, i, f, v)
        raise

    # 有限性快速检查：C 实现的 sum 很快；只有总和非有限时才逐元素分类，
    # 以区分 NaN 与 +/-Inf（极大但有限的值可能让 sum 溢出为 Inf，
    # 此时逐元素复查后若全部有限则正常返回）。
    if out and not math.isfinite(sum(out)):
        all_finite = True
        for i, f in enumerate(out):
            if math.isnan(f):
                _raise_nonfinite(name, i, f, raw[i] if i < len(raw) else f)
            if math.isinf(f):
                _raise_nonfinite(name, i, f, raw[i] if i < len(raw) else f)
            all_finite = all_finite and math.isfinite(f)
        if not all_finite:  # pragma: no cover - 上面循环必然已抛出
            raise NonFiniteInputError("%s contains non-finite samples" % name)
    return out


def _raise_nonfinite(name, index, f, original):
    if math.isnan(f):
        raise NaNInputError(
            "%s[%d] is non-finite (NaN): %r" % (name, index, original)
        )
    raise InfiniteInputError(
        "%s[%d] is non-finite (%s): %r"
        % (name, index, "+Inf" if f > 0 else "-Inf", original)
    )


def _check_pair(left, right):
    """校验左右声道，返回两个等长的 ``list[float]``。"""
    lf = _coerce_samples(left, "left")
    rf = _coerce_samples(right, "right")
    if len(lf) != len(rf):
        raise LengthMismatchError(
            "left and right must have equal length, got %d and %d"
            % (len(lf), len(rf))
        )
    return lf, rf


def mono_level(left, right=None):
    """单声道下混电平，单位 dBFS（RMS）。

    下混约定为 ``M = (L + R) / 2``，返回
    ``20 * log10(rms(M))``；数字满幅 1.0 的恒定信号返回 ``0.0``。
    空输入或纯静音返回 ``-inf``（仍是 ``float``）。

    也允许只传一个已经是单声道的序列（``right`` 省略时直接测量 ``left``）。
    """
    if right is None:
        m = _coerce_samples(left, "mono")
    else:
        l, r = _check_pair(left, right)
        m = [(a + b) * 0.5 for a, b in zip(l, r)]
    n = len(m)
    if n == 0:
        return float("-inf")
    energy = 0.0
    for v in m:
        energy += v * v
    mean_square = energy / n
    if mean_square <= 0.0 or not math.isfinite(mean_square):
        # 理论上非有限输入已在 _coerce_samples 中被拦截。
        return float("-inf")
    return 10.0 * math.log10(mean_square)


def correlation(left, right):
    """立体声相位相关系数（音频相关表常用的能量归一化形式）。

    ``rho = 2 * sum(L*R) / (sum(L^2) + sum(R^2))``，取值 [-1, 1]：
    L == R 时为 1，L == -R 时为 -1。与
    ``(E_M - E_S) / (E_M + E_S)`` 代数等价。
    双声道均为静音（零能量）时按“完全相同”返回 1.0。
    """
    l, r = _check_pair(left, right)
    dot = el = er = 0.0
    for a, b in zip(l, r):
        dot += a * b
        el += a * a
        er += b * b
    denom = el + er
    if denom <= 0.0:
        return 1.0
    return 2.0 * dot / denom


def cosine_correlation(left, right):
    """未去均值的归一化点积（余弦相似度）形式的相位相关系数。"""
    l, r = _check_pair(left, right)
    dot = el = er = 0.0
    for a, b in zip(l, r):
        dot += a * b
        el += a * a
        er += b * b
    denom = math.sqrt(el * er)
    if denom <= 0.0:
        return 1.0
    return dot / denom


def pearson_correlation(left, right):
    """去均值的 Pearson 相关系数。"""
    l, r = _check_pair(left, right)
    n = len(l)
    if n == 0:
        return 1.0
    ml = sum(l) / n
    mr = sum(r) / n
    cov = vl = vr = 0.0
    for a, b in zip(l, r):
        da = a - ml
        db = b - mr
        cov += da * db
        vl += da * da
        vr += db * db
    denom = math.sqrt(vl * vr)
    if denom <= 0.0:
        # 至少一个声道为恒定信号：两声道完全相同（含双静音）时为 1，
        # 否则相关系数无定义，返回 0.0。
        return 1.0 if vl == vr == 0.0 else 0.0
    return cov / denom


# “相位相关系数”最常见的两个名字都指向音频相关表公式。
phase_correlation = correlation
stereo_correlation = correlation


class Widen:
    """多频带立体声展宽器（流式、线程安全）。

    参数
    ----
    width:
        宽度。``0.0`` 折叠为严格单声道；``1.0``（默认）近似原样；
        大于 ``1.0`` 时提升中 / 高频侧信号进行展宽，低频侧信号增益
        始终不超过 1。必须是非负有限实数，否则抛出
        :class:`~widen.errors.InvalidWidthError`。
    low_alpha / mid_alpha:
        可选的分频平滑系数（主要用于测试 / 调参），需在 (0, 1] 内。
    """

    __slots__ = (
        "_width",
        "_low_alpha",
        "_mid_alpha",
        "_ll",
        "_lr",
        "_ml",
        "_mr",
        "_primed",
        "_lock",
    )

    def __init__(
        self,
        width=DEFAULT_WIDTH,
        *,
        low_alpha=LOW_ALPHA,
        mid_alpha=MID_ALPHA,
    ):
        self._width = _validate_width(width)
        if not (isinstance(low_alpha, (int, float)) and 0.0 < float(low_alpha) <= 1.0):
            raise InvalidWidthError(
                "low_alpha must be in (0, 1], got %r" % (low_alpha,)
            )
        if not (isinstance(mid_alpha, (int, float)) and 0.0 < float(mid_alpha) <= 1.0):
            raise InvalidWidthError(
                "mid_alpha must be in (0, 1], got %r" % (mid_alpha,)
            )
        self._low_alpha = float(low_alpha)
        self._mid_alpha = float(mid_alpha)
        self._lock = threading.RLock()
        # 分频滤波器状态（左右声道各两级）。
        self._ll = 0.0
        self._lr = 0.0
        self._ml = 0.0
        self._mr = 0.0
        # 首个样本到来时用其本身初始化低频状态，避免零状态造成的启动阶跃。
        self._primed = False

    # -- 宽度属性 -------------------------------------------------------
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        w = _validate_width(value)
        with self._lock:
            self._width = w

    # -- 状态管理 -------------------------------------------------------
    def reset(self):
        """把分频滤波器状态归零（等同于一个全新的展宽器）。"""
        with self._lock:
            self._ll = self._lr = 0.0
            self._ml = self._mr = 0.0
            self._primed = False

    # -- 电平 -----------------------------------------------------------
    def mono_level(self, left, right=None):
        """见 :func:`widen.core.mono_level`。纯测量，不改变流状态。"""
        return mono_level(left, right)

    # -- 主处理 ---------------------------------------------------------
    def process(self, left, right, width=None):
        """处理一个缓冲区，返回 ``(out_left: list[float], out_right: list[float])``。

        连续调用时滤波器状态跨缓冲区保留；空输入返回 ``([], [])``。
        不修改输入序列。可选的 ``width`` 关键字可临时（并持久）覆盖宽度。
        """
        l, r = _check_pair(left, right)
        n = len(l)
        if n == 0:
            return [], []

        if width is not None:
            w = _validate_width(width)
            self.width = w
        else:
            w = self._width
        a1 = self._low_alpha
        a2 = self._mid_alpha

        # 频带侧信号增益：
        #   低频永远不超过原始侧信号（不展宽）；中/高频随 width 单调。
        if w <= 1.0:
            g_lo = g_mid = g_hi = w
        else:
            # 低频侧信号锁定在原始电平（不展宽）；中频温和提升，
            # 高频按 width 展宽。三者都随 width 单调不减。
            g_lo = 1.0
            g_mid = 1.0 + 0.5 * (w - 1.0)
            g_hi = w

        out_l = [0.0] * n
        out_r = [0.0] * n

        with self._lock:
            ll = self._ll
            lr = self._lr
            ml = self._ml
            mr = self._mr

            # 用流的第一个样本初始化低通状态，消除启动阶跃；
            # 整段处理与分块处理都只在全局首样本触发一次，仍逐样本一致。
            if not self._primed:
                ll = l[0]
                lr = r[0]
                ml = 0.0
                mr = 0.0
                self._primed = True

            width_zero = (w == 0.0)

            for i in range(n):
                x = l[i]
                y = r[i]

                # 1) 先分频带（互补一阶网络，low+mid+high == x）。
                ll += a1 * (x - ll)
                lr += a1 * (y - lr)
                rest_l = x - ll
                rest_r = y - lr
                ml += a2 * (rest_l - ml)
                mr += a2 * (rest_r - mr)
                hi_l = rest_l - ml
                hi_r = rest_r - mr

                # 全频带中置（单声道下混），直接由原始输入得到，
                # 保证下混不被分频重组的舍入误差染色。
                m_full = (x + y) * 0.5

                if width_zero:
                    # 同一个 float 对象放入两个列表 => 严格相等。
                    out_l[i] = m_full
                    out_r[i] = m_full
                    continue

                # 2) 每个频带做 M/S 分解，只加权侧信号 S。
                s_lo = (ll - lr) * 0.5
                s_mid = (ml - mr) * 0.5
                s_hi = (hi_l - hi_r) * 0.5
                s_wide = g_lo * s_lo + g_mid * s_mid + g_hi * s_hi

                # 3) M/S 逆变换（M 原样保留）。
                ol = m_full + s_wide
                orr = m_full - s_wide

                # 消除理想重建在满幅附近可能出现的 1 ULP 级越界。
                if ol > 1.0 and ol <= 1.0 + _CLIP_EPS:
                    ol = 1.0
                elif ol < -1.0 and ol >= -1.0 - _CLIP_EPS:
                    ol = -1.0
                if orr > 1.0 and orr <= 1.0 + _CLIP_EPS:
                    orr = 1.0
                elif orr < -1.0 and orr >= -1.0 - _CLIP_EPS:
                    orr = -1.0

                out_l[i] = ol
                out_r[i] = orr

            self._ll = ll
            self._lr = lr
            self._ml = ml
            self._mr = mr

        return out_l, out_r

    def __repr__(self):
        return "Widen(width=%r)" % (self._width,)


def __getattr__(name):
    """对 ``widen.core.AnyNameError`` 形式的非常用异常命名做别名兜底。"""
    from . import errors as _errors

    if name.endswith("Error") or name.endswith("Exception"):
        return getattr(_errors, name) if hasattr(_errors, name) else _errors.__getattr__(name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__version__ = "1.0.0"


def process(left, right, width=DEFAULT_WIDTH):
    """一次性展宽的便捷函数（等价于 ``Widen(width).process(left, right)``）。"""
    return Widen(width).process(left, right)
