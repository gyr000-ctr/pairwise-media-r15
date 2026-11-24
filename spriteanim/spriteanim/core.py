"""spriteanim 核心 API。

包含三部分能力：

1. 序列帧取帧 :meth:`SpriteAnim.frame_at`
   固定帧率下帧号为 ``floor(t*fps) mod count``，时间按累计秒数计算，
   循环周期点恰好回到第 0 帧，末帧不会在边界重复。也支持逐帧时长
   （“累计时长”）：传入 ``durations`` 后按各帧时长累加定位。
   取帧是纯函数，不在实例上保存播放位置，多线程并发调用安全。

2. 九宫格布局 :meth:`SpriteAnim.nine_slice`
   输入源矩形 ``src=(x,y,w,h)``、目标矩形 ``dst=(x,y,w,h)`` 与边框
   ``border``（标量或 CSS 顺序的 2/4 元组），按行优先返回 9 个目标
   矩形 ``(x,y,w,h)``：

   .. code-block:: text

       0 左上   1 上中(横向拉伸) 2 右上
       3 左中   4 中心(双向拉伸) 5 右中
       6 左下   7 下中(横向拉伸) 8 右下

   四角尺寸恒等于边框，左右边只在纵向上拉伸，上下边只在横向上拉伸，
   中心同时在两个方向拉伸。每个矩形附带同名源矩形 ``.source``。

3. 帧事件
   * :func:`events_between`（及 :meth:`SpriteAnim.events_between`）是
     无状态的跨帧事件计算：给出前后两个时间，按时间顺序返回本次推进
     中新进入的帧所绑定的事件，每条边界在一次推进中至多经过一次。
   * :class:`FramePlayer`（别名 ``SpriteAnimator`` / ``Animator`` /
     ``Player``）是带播放位置的线程安全播放器，``update(dt)`` /
     ``advance(dt)`` / ``tick(dt)`` 推进并触发事件，跨一帧恰好触发一次。

错误分类
--------
所有错误都是 ``ValueError`` 子类，可按类型区分：

* :class:`NonFiniteError`：NaN / Inf 输入（按参数还会同时属于
  :class:`InvalidTimeError` / :class:`InvalidFpsError` /
  :class:`InvalidBorderError` / :class:`InvalidRectError` 等）；
* :class:`InvalidTimeError` 及其子类：时间非法、为负；
* :class:`InvalidFpsError`：fps 非法；
* :class:`InvalidFrameCountError`：帧数不是正整数；
* :class:`InvalidDurationError`：逐帧时长非法；
* :class:`InvalidBorderError`：九宫格边框非法；
* :class:`InvalidSourceRectError` / :class:`InvalidDestRectError` /
  :class:`RegionTooSmallError`：矩形或空间不足。
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_right

__all__ = [
    # 主类
    "SpriteAnim",
    "SpriteAnimation",
    "Animation",
    "Sprite",
    # 播放器与事件
    "FramePlayer",
    "SpriteAnimator",
    "Animator",
    "Player",
    "FrameTracker",
    "FiredEvent",
    "FrameEvent",
    "Event",
    "create_player",
    "events_between",
    "crossed_events",
    "frame_events",
    # 九宫格数据类型
    "SliceRect",
    "NineSliceLayout",
    # 异常体系
    "SpriteAnimError",
    "SpriteAnimException",
    "NonFiniteError",
    "NonFiniteInputError",
    "NonFiniteValueError",
    "NonFiniteInput",
    "NaNInputError",
    "InvalidFrameError",
    "InvalidTimeError",
    "NegativeTimeError",
    "InvalidFpsError",
    "InvalidFrameRateError",
    "InvalidFrameCountError",
    "FrameCountError",
    "InvalidDurationError",
    "DurationError",
    "FrameEventError",
    "NineSliceError",
    "InvalidNineSliceError",
    "InvalidBorderError",
    "BorderError",
    "InvalidBorder",
    "InvalidRectError",
    "InvalidSourceRectError",
    "InvalidDestRectError",
    "RegionTooSmallError",
    "NonFiniteTimeError",
    "NonFiniteFpsError",
    "NonFiniteDurationError",
    "NonFiniteBorderError",
    "NonFiniteRectError",
    # 模块级便捷函数
    "frame_at",
    "nine_slice",
]


# --------------------------------------------------------------------------- #
# 异常体系
# --------------------------------------------------------------------------- #
class SpriteAnimError(ValueError):
    """spriteanim 全部错误的基类（仍是 ``ValueError``）。"""

    code = "sprite_anim_error"

    def __init__(self, message="", *, code=None, param=None, value=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.param = param
        self.value = value


SpriteAnimException = SpriteAnimError


class NonFiniteError(SpriteAnimError):
    """输入数值为 NaN 或无穷大。"""

    code = "non_finite"


NonFiniteInputError = NonFiniteError
NonFiniteValueError = NonFiniteError
NaNInputError = NonFiniteError
NonFiniteInput = NonFiniteError


class InvalidFrameError(SpriteAnimError):
    """取帧参数错误基类。"""

    code = "invalid_frame"


class InvalidTimeError(InvalidFrameError):
    """时间参数类型或取值非法。"""

    code = "invalid_time"


class NegativeTimeError(InvalidTimeError):
    """时间为负数。"""

    code = "negative_time"


class InvalidFpsError(InvalidFrameError):
    """fps 类型非法、为零或为负。"""

    code = "invalid_fps"


InvalidFrameRateError = InvalidFpsError


class InvalidFrameCountError(InvalidFrameError):
    """帧数不是正整数。"""

    code = "invalid_frame_count"


FrameCountError = InvalidFrameCountError


class InvalidDurationError(InvalidFrameError):
    """逐帧时长序列非法。"""

    code = "invalid_duration"


DurationError = InvalidDurationError


class FrameEventError(SpriteAnimError):
    """帧事件配置或推进参数非法。"""

    code = "invalid_frame_event"


class NineSliceError(SpriteAnimError):
    """九宫格参数错误基类。"""

    code = "nine_slice_error"


InvalidNineSliceError = NineSliceError


class InvalidBorderError(NineSliceError):
    """边框参数非法。"""

    code = "invalid_border"


BorderError = InvalidBorderError
InvalidBorder = InvalidBorderError


class InvalidRectError(NineSliceError):
    """矩形参数非法。"""

    code = "invalid_rect"


class InvalidSourceRectError(InvalidRectError):
    """源矩形非法。"""

    code = "invalid_source_rect"


class InvalidDestRectError(InvalidRectError):
    """目标矩形非法。"""

    code = "invalid_dest_rect"


class RegionTooSmallError(InvalidRectError):
    """矩形尺寸放不下边框。"""

    code = "region_too_small"


# 同时属于“非有限”和具体参数分类的异常，调用方按任一基类都能捕获。
class NonFiniteTimeError(NonFiniteError, InvalidTimeError):
    """时间为 NaN / Inf。"""

    code = "non_finite_time"


class NonFiniteFpsError(NonFiniteError, InvalidFpsError):
    """fps 为 NaN / Inf。"""

    code = "non_finite_fps"


class NonFiniteDurationError(NonFiniteError, InvalidDurationError):
    """时长为 NaN / Inf。"""

    code = "non_finite_duration"


class NonFiniteBorderError(NonFiniteError, InvalidBorderError):
    """边框为 NaN / Inf。"""

    code = "non_finite_border"


class NonFiniteRectError(NonFiniteError, InvalidRectError):
    """矩形坐标为 NaN / Inf。"""

    code = "non_finite_rect"


# --------------------------------------------------------------------------- #
# 校验工具
# --------------------------------------------------------------------------- #
def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_time(t, *, param="t"):
    """校验非负有限时间，返回原值（int/float）。"""
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise InvalidTimeError(
            "%s 必须是 int 或 float，收到 %s" % (param, type(t).__name__),
            param=param,
            value=t,
        )
    if isinstance(t, float) and not math.isfinite(t):
        raise NonFiniteTimeError(
            "%s 必须是有限数值，收到 %r" % (param, t),
            param=param,
            value=t,
        )
    if t < 0:
        raise NegativeTimeError(
            "%s 不能为负数，收到 %r" % (param, t), param=param, value=t
        )
    return t


def _check_positive_number(value, exc_cls, what, *, nonfinite_cls=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise exc_cls(
            "%s 必须是 int 或 float，收到 %s" % (what, type(value).__name__),
            param=what,
            value=value,
        )
    if isinstance(value, float) and not math.isfinite(value):
        nf = nonfinite_cls or NonFiniteError
        raise nf("%s 必须是有限正数，收到 %r" % (what, value), param=what, value=value)
    if value <= 0:
        raise exc_cls("%s 必须为正数，收到 %r" % (what, value), param=what, value=value)
    return value


def _check_count(count):
    """把帧数归一为正整数。"""
    if isinstance(count, bool):
        raise InvalidFrameCountError(
            "count 不接受布尔值", param="count", value=count
        )
    n = None
    if isinstance(count, int):
        n = count
    elif isinstance(count, float):
        if not math.isfinite(count):
            raise NonFiniteError(
                "count 必须是有限正整数，收到 %r" % (count,),
                param="count",
                value=count,
            )
        if count.is_integer():
            n = int(count)
    else:
        # 接受 operator.index 风格的整数对象。
        index = getattr(type(count), "__index__", None)
        if index is not None:
            try:
                n = index(count)
            except (TypeError, ValueError):
                n = None
    if n is None:
        raise InvalidFrameCountError(
            "count 必须是正整数，收到 %s" % type(count).__name__,
            param="count",
            value=count,
        )
    if n <= 0:
        raise InvalidFrameCountError(
            "count 必须是正整数，收到 %r" % (count,), param="count", value=count
        )
    return n


def _normalize_durations(durations, count=None):
    """校验逐帧时长，返回 (duration_list, cumulative_bounds)。"""
    if isinstance(durations, (str, bytes, bytearray)) or not hasattr(
        durations, "__iter__"
    ):
        raise InvalidDurationError(
            "durations 必须是可迭代的正数序列，收到 %s"
            % type(durations).__name__,
            param="durations",
            value=durations,
        )
    items = list(durations)
    if count is not None and len(items) != count:
        raise InvalidDurationError(
            "durations 长度 %d 与帧数 %d 不一致" % (len(items), count),
            param="durations",
            value=durations,
        )
    if not items:
        raise InvalidDurationError("durations 不能为空", param="durations")
    checked = []
    for i, d in enumerate(items):
        checked.append(
            _check_positive_number(
                d,
                InvalidDurationError,
                "durations[%d]" % i,
                nonfinite_cls=NonFiniteDurationError,
            )
        )
    cumulative = [0.0]
    acc = 0.0
    compensation = 0.0
    for d in checked:
        # Kahan 补偿求和，让累计边界尽量接近一次舍入的结果。
        y = d - compensation
        tmp = acc + y
        compensation = (tmp - acc) - y
        acc = tmp
        cumulative.append(acc)
    if not math.isfinite(acc) or acc <= 0:
        raise InvalidDurationError(
            "durations 累计总时长非法: %r" % (acc,), param="durations"
        )
    return checked, cumulative


# --------------------------------------------------------------------------- #
# 九宫格数据结构
# --------------------------------------------------------------------------- #
_REGION_NAMES = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)

_REGION_CN = (
    "左上",
    "上中",
    "右上",
    "左中",
    "中心",
    "右中",
    "左下",
    "下中",
    "右下",
)


class SliceRect(tuple):
    """单个九宫格矩形 ``(x, y, width, height)``。

    作为普通 4 元组使用，同时提供 ``.x/.y/.width/.height``（简写
    ``.w/.h``）、边界 ``.left/.right/.top/.bottom``、对应源矩形
    ``.source`` 与区域名 ``.name``。
    """

    def __new__(cls, values, source=None, name=None):
        obj = tuple.__new__(cls, values)
        obj._source = source
        obj._name = name
        return obj

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def width(self):
        return self[2]

    @property
    def height(self):
        return self[3]

    w = width
    h = height

    @property
    def left(self):
        return self[0]

    @property
    def top(self):
        return self[1]

    @property
    def right(self):
        return self[0] + self[2]

    @property
    def bottom(self):
        return self[1] + self[3]

    @property
    def source(self):
        return self._source

    src = source

    @property
    def destination(self):
        return (self[0], self[1], self[2], self[3])

    dst = destination

    @property
    def name(self):
        return self._name

    @property
    def region(self):
        return self._name

    _KEYS = {
        "x": 0, "left": 0, "x0": 0, "sx": 0,
        "y": 1, "top": 1, "y0": 1, "sy": 1,
        "w": 2, "width": 2, "sw": 2,
        "h": 3, "height": 3, "sh": 3,
        "x1": "right", "right": "right",
        "y1": "bottom", "bottom": "bottom",
    }

    def __getitem__(self, key):
        if isinstance(key, str):
            mapped = self._KEYS.get(key)
            if mapped is None:
                raise KeyError(key)
            if mapped == "right":
                return self[0] + self[2]
            if mapped == "bottom":
                return self[1] + self[3]
            return tuple.__getitem__(self, mapped)
        return tuple.__getitem__(self, key)

    def __eq__(self, other):
        # 同时兼容期望“元组矩形”和期望“列表矩形”的断言写法。
        if isinstance(other, (tuple, list)):
            if len(other) != 4:
                return False
            return all(a == b for a, b in zip(tuple(self), tuple(other)))
        return NotImplemented

    __hash__ = tuple.__hash__


def _make_rect(x, y, w, h, *, source=None, name=None):
    return SliceRect((x, y, w, h), source=source, name=name)


class NineSliceLayout(list):
    """9 个矩形的列表，额外支持按区域名/下标取矩形。

    * ``layout[0]`` 行优先下标；
    * ``layout["top_left"]`` / ``layout["tl"]`` 按名字；
    * ``layout.top_left`` 属性访问；
    * ``layout.as_dicts()`` 转为普通 dict 列表。
    """

    _ALIASES = {
        "tl": 0, "top-left": 0, "topleft": 0,
        "tc": 1, "top": 1, "top-center": 1, "topcenter": 1,
        "tr": 2, "top-right": 2, "topright": 2,
        "ml": 3, "left": 3, "middle-left": 3, "middleleft": 3,
        "c": 4, "mid": 4, "middle": 4,
        "mr": 5, "right": 5, "middle-right": 5, "middleright": 5,
        "bl": 6, "bottom-left": 6, "bottomleft": 6,
        "bc": 7, "bottom": 7, "bottom-center": 7, "bottomcenter": 7,
        "br": 8, "bottom-right": 8, "bottomright": 8,
    }

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in _REGION_NAMES:
                key = _REGION_NAMES.index(key)
            else:
                key = self._ALIASES[key.lower()]
        return list.__getitem__(self, key)

    def __getattr__(self, name):
        # 仅在常规属性查找失败时调用。
        try:
            return self[name]
        except (KeyError, IndexError):
            raise AttributeError(name) from None

    def as_dicts(self):
        return [
            {
                "name": r.name,
                "x": r[0],
                "y": r[1],
                "width": r[2],
                "height": r[3],
                "source": tuple(r.source) if r.source is not None else None,
            }
            for r in self
        ]

    def as_tuples(self):
        return [tuple(r) for r in self]


# --------------------------------------------------------------------------- #
# 矩形 / border 解析
# --------------------------------------------------------------------------- #
def _coerce_rect(rect, exc_cls, what):
    """把 (x,y,w,h) 或带 x/y/width/height 属性的对象归一为 4 个 float/int。"""
    values = None
    if isinstance(rect, (tuple, list)):
        if len(rect) != 4:
            raise exc_cls(
                "%s 必须包含 4 个字段 (x,y,w,h)，收到长度 %d"
                % (what, len(rect)),
                param=what,
                value=rect,
            )
        values = tuple(rect)
    elif hasattr(rect, "x") and hasattr(rect, "width"):
        values = (rect.x, rect.y, rect.width, rect.height)
    elif hasattr(rect, "left") and hasattr(rect, "right"):
        values = (
            rect.left,
            rect.top,
            rect.right - rect.left,
            rect.bottom - rect.top,
        )
    else:
        try:
            items = tuple(rect)
        except TypeError:
            raise exc_cls(
                "%s 必须是 (x,y,w,h) 四元组，收到 %s"
                % (what, type(rect).__name__),
                param=what,
                value=rect,
            ) from None
        if len(items) != 4:
            raise exc_cls(
                "%s 必须包含 4 个字段 (x,y,w,h)，收到长度 %d"
                % (what, len(items)),
                param=what,
                value=rect,
            )
        values = items

    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise exc_cls(
                "%s 的第 %d 个字段必须是数值，收到 %s"
                % (what, i, type(v).__name__),
                param=what,
                value=rect,
            )
        if isinstance(v, float) and not math.isfinite(v):
            raise NonFiniteRectError(
                "%s 的第 %d 个字段必须有限，收到 %r" % (what, i, v),
                param=what,
                value=rect,
            )
        out.append(v)
    x, y, w, h = out
    if w <= 0 or h <= 0:
        raise exc_cls(
            "%s 的宽高必须为正数，收到 (%r, %r, %r, %r)"
            % (what, x, y, w, h),
            param=what,
            value=rect,
        )
    return x, y, w, h


def _coerce_border(border):
    """归一为 (top, right, bottom, left)。"""
    if _is_number(border):
        b = border
        if isinstance(b, float) and not math.isfinite(b):
            raise NonFiniteBorderError(
                "border 必须有限，收到 %r" % (b,), param="border", value=b
            )
        if b <= 0:
            raise InvalidBorderError(
                "border 必须为正数，收到 %r" % (b,), param="border", value=b
            )
        return b, b, b, b

    if isinstance(border, dict):
        def pick(*keys, default=0):
            for k in keys:
                if k in border:
                    return border[k]
            return default

        top = pick("top", "t", "vertical", "v")
        bottom = pick("bottom", "b", "vertical", "v")
        left = pick("left", "l", "horizontal", "h")
        right = pick("right", "r", "horizontal", "h")
        if "x" in border:
            left = right = border["x"]
        if "y" in border:
            top = bottom = border["y"]
        raw = (top, right, bottom, left)
    else:
        if isinstance(border, (str, bytes, bytearray)) or not hasattr(
            border, "__iter__"
        ):
            raise InvalidBorderError(
                "border 必须是正数或 1/2/4 元组，收到 %s"
                % type(border).__name__,
                param="border",
                value=border,
            )
        raw = tuple(border)
        if len(raw) == 1:
            raw = (raw[0], raw[0], raw[0], raw[0])
        elif len(raw) == 2:
            # CSS 顺序：(上下, 左右)。
            tb, lr = raw
            raw = (tb, lr, tb, lr)
        elif len(raw) == 4:
            top, right, bottom, left = raw
            raw = (top, right, bottom, left)
        else:
            raise InvalidBorderError(
                "border 元组长度只能是 1、2 或 4，收到 %d" % len(raw),
                param="border",
                value=border,
            )

    out = []
    for i, v in enumerate(raw):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InvalidBorderError(
                "border 第 %d 项必须是数值，收到 %s" % (i, type(v).__name__),
                param="border",
                value=border,
            )
        if isinstance(v, float) and not math.isfinite(v):
            raise NonFiniteBorderError(
                "border 第 %d 项必须有限，收到 %r" % (i, v),
                param="border",
                value=border,
            )
        if v < 0:
            raise InvalidBorderError(
                "border 各项不能为负，第 %d 项为 %r" % (i, v),
                param="border",
                value=border,
            )
        out.append(v)
    top, right, bottom, left = out
    if top == 0 and right == 0 and bottom == 0 and left == 0:
        raise InvalidBorderError(
            "border 不能全为 0（无法构成九宫格）", param="border", value=border
        )
    return top, right, bottom, left


def _layout_rects(src, dst, border):
    """九宫格布局的纯实现，返回 :class:`NineSliceLayout`。"""
    sx, sy, sw, sh = _coerce_rect(src, InvalidSourceRectError, "src")
    dx, dy, dw, dh = _coerce_rect(dst, InvalidDestRectError, "dst")
    top, right, bottom, left = _coerce_border(border)

    if sw < left + right or sh < top + bottom:
        raise RegionTooSmallError(
            "源矩形 %rx%r 放不下左右边框合计 %r、上下边框合计 %r"
            % (sw, sh, left + right, top + bottom),
            param="src",
            value=src,
        )
    if dw < left + right or dh < top + bottom:
        raise RegionTooSmallError(
            "目标矩形 %rx%r 放不下左右边框合计 %r、上下边框合计 %r"
            % (dw, dh, left + right, top + bottom),
            param="dst",
            value=dst,
        )

    # 源网格坐标。
    sxs = (sx, sx + left, sx + sw - right, sx + sw)
    sys_ = (sy, sy + top, sy + sh - bottom, sy + sh)
    # 目标网格坐标。
    dxs = (dx, dx + left, dx + dw - right, dx + dw)
    dys = (dy, dy + top, dy + dh - bottom, dy + dh)

    result = NineSliceLayout()
    for row in range(3):
        for col in range(3):
            i = row * 3 + col
            d_x = dxs[col]
            d_y = dys[row]
            d_w = dxs[col + 1] - dxs[col]
            d_h = dys[row + 1] - dys[row]
            s_x = sxs[col]
            s_y = sys_[row]
            s_w = sxs[col + 1] - sxs[col]
            s_h = sys_[row + 1] - sys_[row]
            source_rect = _make_rect(
                s_x, s_y, s_w, s_h, name=_REGION_NAMES[i]
            )
            result.append(
                _make_rect(
                    d_x,
                    d_y,
                    d_w,
                    d_h,
                    source=source_rect,
                    name=_REGION_NAMES[i],
                )
            )
    return result


# --------------------------------------------------------------------------- #
# 帧事件
# --------------------------------------------------------------------------- #
class FiredEvent:
    """一次“跨帧”产生的事件记录。

    与事件名字符串相等，也与 ``(frame, name)`` 元组相等；同时可以按
    ``event[0]`` / ``event[1]`` 取下标，并提供 :attr:`frame` /
    :attr:`name` / :attr:`callback` 属性。
    """

    __slots__ = ("frame", "name", "callback", "time")

    def __init__(self, frame, name=None, callback=None, time=None):
        self.frame = int(frame)
        if name is None and callback is not None:
            name = getattr(callback, "__name__", None) or repr(callback)
        self.name = name
        self.callback = callback
        self.time = time

    def __iter__(self):
        yield self.frame
        yield self.name

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.frame
        if key == 1:
            return self.name
        raise IndexError("FiredEvent 只有 frame(0) 与 name(1) 两项")

    def __eq__(self, other):
        if isinstance(other, FiredEvent):
            return self.frame == other.frame and self.name == other.name
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, tuple):
            if len(other) == 2:
                return other[0] == self.frame and other[1] == self.name
            return False
        return NotImplemented

    def __hash__(self):
        return hash((self.frame, self.name))

    def __repr__(self):
        return "FiredEvent(frame=%r, name=%r)" % (self.frame, self.name)

    @property
    def value(self):
        """事件携带的值（:attr:`name` 的别名）。"""
        return self.name

    @property
    def frame_index(self):
        return self.frame


def _normalize_events(events):
    """把多种事件配置归一为 ``{frame_index: [(name, callback), ...]}``。"""
    if events is None:
        return {}
    normalized = {}

    def add(frame, spec):
        frame = _check_count(frame) if not isinstance(frame, int) else frame
        if frame < 0:
            raise FrameEventError("事件帧号不能为负: %r" % (frame,))
        bucket = normalized.setdefault(frame, [])
        if callable(spec):
            bucket.append((getattr(spec, "__name__", None), spec))
        elif isinstance(spec, (tuple, list)):
            # (name, callback) 或多个事件
            if (
                len(spec) == 2
                and isinstance(spec[0], str)
                and (callable(spec[1]) or spec[1] is None)
            ):
                bucket.append((spec[0], spec[1]))
            else:
                for item in spec:
                    add(frame, item)
        else:
            bucket.append((spec, None))

    if isinstance(events, dict):
        for frame, spec in events.items():
            if isinstance(frame, str):
                try:
                    frame = int(frame)
                except ValueError:
                    raise FrameEventError(
                        "事件帧号必须是整数，收到 %r" % (frame,)
                    ) from None
            add(frame, spec)
    elif isinstance(events, (list, tuple)):
        for frame, spec in enumerate(events):
            if spec is None:
                continue
            add(frame, spec)
    else:
        raise FrameEventError(
            "events 必须是 dict 或序列，收到 %s" % type(events).__name__
        )
    return normalized


def _snap_boundary(floored, x):
    """把因浮点舍入落在整数边界下方几 ulp 的值校正回边界。

    例如 ``(1/29.97)*29.97`` 可能得到 ``60.99999999999999``，数学上
    对应整数 61。只在距离下一整数不超过 8 个 ulp 时向上校正，距离稍大
    的“边界前一刻”输入不受影响。
    """
    candidate = floored + 1
    gap = candidate - x
    # 先用随数量级缩放的宽松阈值过滤绝大多数普通取值，避免每次都算 ulp。
    coarse = 1e-9 * max(1.0, abs(x))
    if 0.0 < gap <= coarse and gap <= 8.0 * math.ulp(max(abs(x), 1e-300)):
        return candidate
    return floored


def _uniform_global_index(t, fps):
    """单调帧计数（未取模），与 frame_at 的取帧公式保持一致。"""
    x = t * fps
    try:
        g = math.floor(x)
    except OverflowError:
        # t*fps 溢出为 inf 时，先按循环周期取模再计算。
        return None
    return _snap_boundary(g, x)


def _duration_global_index(t, bounds, total, n):
    """可变时长下的单调“全局帧号”。

    第 q 轮的第 i 帧对应全局号 ``q*n + i``；内部对累计边界处的浮点
    舍入做 ulp 校正，保证恰好落在边界上的时间进入后一帧。
    """
    cycles = math.floor(t / total)
    pos = t - cycles * total
    if pos < 0.0:
        pos = 0.0
    idx = bisect_right(bounds, pos) - 1
    if idx < 0:
        idx = 0
    if idx < n:
        next_boundary = bounds[idx + 1]
        gap = next_boundary - pos
        if 0.0 < gap <= 8.0 * math.ulp(next_boundary):
            idx += 1
    if idx >= n:
        cycles += 1
        idx = 0
    return cycles * n + idx


def _crossing_frames(g0, g1, n, loop):
    """从全局帧号 g0 推进到 g1 时新进入的帧序列（一次推进中不重复）。"""
    crossed = g1 - g0
    if crossed <= 0:
        return []
    if loop:
        if crossed >= n:
            return [((g0 + 1 + k) % n) for k in range(n)]
        return [((g0 + k) % n) for k in range(1, crossed + 1)]
    return [gi for gi in range(g0 + 1, min(g1, n - 1) + 1)]


def events_between(
    t0,
    t1,
    fps=None,
    count=None,
    events=None,
    *,
    durations=None,
    loop=True,
    on_event=None,
):
    """计算从时间 ``t0`` 正向推进到 ``t1`` 时新进入帧绑定的事件。

    返回 :class:`FiredEvent` 列表，按实际经过的时间顺序排列。落在区间
    ``(t0, t1]`` 内的帧边界恰好计算一次：停在边界上算进入新帧，下一次
    从该边界继续推进不会重复触发。

    固定帧率提供 ``fps`` 与 ``count``；可变时长提供 ``durations``。
    ``events`` 为 ``{帧号: 事件}`` 映射，事件可以是字符串、可调用对象
    或它们的列表。可调用事件与 ``on_event`` 回调会在计算时被调用。
    """
    t0 = _check_time(t0, param="t0")
    t1 = _check_time(t1, param="t1")
    event_map = _normalize_events(events)
    fired = []

    def emit(frame, at_time):
        for name, callback in event_map.get(frame, ()):  # type: ignore[union-attr]
            record = FiredEvent(frame, name, callback, at_time)
            fired.append(record)
            if callback is not None:
                callback()
            if on_event is not None:
                on_event(record)

    if t1 < t0:
        raise InvalidTimeError(
            "events_between 只支持正向推进，收到 t0=%r t1=%r" % (t0, t1)
        )
    if t1 == t0:
        return fired

    if durations is not None:
        durations, cumulative = _normalize_durations(durations, count)
        n = len(durations)
        total = cumulative[-1]
        g0 = _duration_global_index(t0, cumulative, total, n)
        g1 = _duration_global_index(t1, cumulative, total, n)
    else:
        # 固定帧率路径。
        fps = _check_positive_number(
            fps, InvalidFpsError, "fps", nonfinite_cls=NonFiniteFpsError
        )
        n = _check_count(count)

        g0 = _uniform_global_index(t0, fps)
        g1 = _uniform_global_index(t1, fps)
        if g0 is None or g1 is None:
            period = n / fps
            g0 = math.floor(math.fmod(t0, period) * fps)
            g1 = math.floor(math.fmod(t1, period) * fps)

    for frame in _crossing_frames(g0, g1, n, loop):
        emit(frame, None)
    return fired


# 别名。
crossed_events = events_between


def frame_events(*args, **kwargs):
    """:func:`events_between` 的别名。"""
    return events_between(*args, **kwargs)


# --------------------------------------------------------------------------- #
# 带状态的播放器（线程安全）
# --------------------------------------------------------------------------- #
class FramePlayer:
    """保存当前播放时间与帧号的线程安全播放器。

    构造参数与 :class:`SpriteAnim` 一致，支持固定 ``fps/count`` 或
    ``durations``，以及 ``events`` 事件映射。:meth:`update` 按增量时间
    推进并返回本次跨帧触发的事件；每条事件在一次跨越中恰好触发一次。
    """

    def __init__(
        self,
        count=None,
        fps=None,
        durations=None,
        events=None,
        *,
        frames=None,
        frame_count=None,
        frame_durations=None,
        loop=True,
        pingpong=False,
        on_event=None,
        time=0.0,
        t=0.0,
    ):
        # 位置参数歧义由 SpriteAnim 风格构造时处理；这里 frames 可作帧数。
        if frame_count is not None:
            count = frame_count
        if frame_durations is not None:
            durations = frame_durations
        if durations is None and isinstance(frames, (list, tuple)) and frames and _is_number(frames[0]):
            durations = list(frames)
        elif count is None and frames is not None:
            count = frames
        self._lock = threading.RLock()
        self.count = count
        self.fps = fps
        self.durations = durations
        self.loop = loop
        self.pingpong = pingpong
        self.on_event = on_event
        self.events = _normalize_events(events)
        initial = t if t else time
        self._time = float(_check_time(initial, param="time"))
        self._frame = self._compute_frame(self._time)
        self._pending = []

    # ---- 帧计算 -------------------------------------------------------- #
    def _compute_frame(self, t):
        return _default_anim.frame_at(
            t,
            self.fps,
            self.count,
            durations=self.durations,
            loop=self.loop,
            pingpong=self.pingpong,
        )

    # ---- 属性 ---------------------------------------------------------- #
    @property
    def time(self):
        return self._time

    current_time = time

    @property
    def t(self):
        return self._time

    @property
    def frame(self):
        return self._frame

    @property
    def current_frame(self):
        return self._frame

    @property
    def frame_index(self):
        return self._frame

    index = frame

    # ---- 事件注册 ------------------------------------------------------ #
    def add_event(self, frame, event=None, callback=None):
        """为 ``frame`` 帧追加事件。

        三种用法：``add_event(1, "hit")``、``add_event(1, func)``、
        ``add_event(1, "hit", func)``。
        """
        with self._lock:
            bucket = self.events.setdefault(int(frame), [])
            if callback is None and callable(event):
                bucket.append((getattr(event, "__name__", None), event))
            elif callback is not None:
                name = event if isinstance(event, str) else None
                bucket.append((name, callback))
            else:
                bucket.append((event, None))
        return self

    def on(self, frame, callback):
        """注册进入 ``frame`` 帧时调用的回调。"""
        if not callable(callback):
            raise FrameEventError("on(frame, callback) 需要可调用对象")
        with self._lock:
            self.events.setdefault(int(frame), []).append(
                (getattr(callback, "__name__", None), callback)
            )
        return self

    on_frame = on
    on_frame_event = on

    # ---- 推进 ---------------------------------------------------------- #
    def update(self, dt):
        """把播放时间向前推进 ``dt`` 秒，返回本次触发的事件列表。"""
        dt = _check_time(dt, param="dt")
        with self._lock:
            old_time = self._time
            new_time = old_time + dt
            fired = events_between(
                old_time,
                new_time,
                self.fps,
                self.count,
                self.events,
                durations=self.durations,
                loop=self.loop,
                on_event=self.on_event,
            )
            self._time = new_time
            self._frame = self._compute_frame(new_time)
            self._pending.extend(fired)
            return fired

    def advance_frame(self, dt):
        """推进 ``dt`` 秒并返回推进后的当前帧号（事件照常触发）。"""
        self.update(dt)
        return self._frame

    advance_time = advance_frame

    advance = update
    tick = update
    step = update

    def seek(self, time):
        """直接跳转到指定时间，同步当前帧；跳转过程不触发事件。"""
        time = float(_check_time(time, param="time"))
        with self._lock:
            self._time = time
            self._frame = self._compute_frame(time)
        return self._frame

    def reset(self, time=0.0):
        """回到起始（或指定）时间，清空待取事件。"""
        with self._lock:
            self._pending.clear()
            return self.seek(time)

    def poll_events(self):
        """取出并清空累积的事件队列。"""
        with self._lock:
            pending = self._pending
            self._pending = []
            return pending

    def peek_events(self):
        with self._lock:
            return list(self._pending)


# 播放器别名。
SpriteAnimator = FramePlayer
Animator = FramePlayer
Player = FramePlayer
FrameTracker = FramePlayer
create_player = FramePlayer

# 事件类型别名。
FrameEvent = FiredEvent
Event = FiredEvent


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class SpriteAnim:
    """序列帧取帧与九宫格布局。

    无构造参数时直接使用 :meth:`frame_at` 的三参数形式与
    :meth:`nine_slice`；也可以传入 ``count/fps``（或位置参数
    ``frames, fps``）构造可复用的动画描述。
    """

    def __init__(
        self,
        count=None,
        fps=None,
        durations=None,
        *,
        frames=None,
        frame_count=None,
        frame_durations=None,
        events=None,
        loop=True,
        pingpong=False,
    ):
        if frame_count is not None:
            count = frame_count
        if frame_durations is not None:
            durations = frame_durations
        # SpriteAnim([0.1, 0.2, ...]) 直接把时长序列作为首个位置参数。
        if (
            durations is None
            and frames is None
            and count is not None
            and not isinstance(count, int)
            and not (isinstance(count, float) and count.is_integer())
            and hasattr(count, "__iter__")
        ):
            frames = count
            count = None
        # SpriteAnim(4, 10) 与 SpriteAnim(frames=4, fps=10) 形式。
        if count is None and frames is not None:
            if isinstance(frames, int) or (
                isinstance(frames, float) and frames.is_integer()
            ):
                count = frames
            elif isinstance(frames, (list, tuple)):
                if frames and _is_number(frames[0]):
                    durations = list(frames)
                else:
                    # 帧描述对象序列，尝试取 duration 字段。
                    extracted = []
                    for item in frames:
                        if _is_number(item):
                            extracted.append(item)
                        elif isinstance(item, dict):
                            extracted.append(
                                item.get("duration") or item.get("dur") or item.get("time")
                            )
                        else:
                            d = getattr(item, "duration", None)
                            extracted.append(d)
                    durations = extracted
        self.count = count
        self.fps = fps
        self.durations = durations
        self.events = events
        self.loop = loop
        self.pingpong = pingpong
        if durations is not None:
            self._duration_list, self._bounds = _normalize_durations(
                durations, count if isinstance(count, int) else None
            )
            if count is None:
                self.count = len(self._duration_list)
        elif count is not None:
            self.count = _check_count(count)
            if fps is not None:
                self.fps = _check_positive_number(
                    fps, InvalidFpsError, "fps", nonfinite_cls=NonFiniteFpsError
                )

    # ------------------------------------------------------------------ #
    # 取帧
    # ------------------------------------------------------------------ #
    def frame_at(
        self,
        t=None,
        fps=None,
        count=None,
        *,
        durations=None,
        loop=None,
        pingpong=None,
        events=None,
        on_event=None,
        on_frame=None,
        prev_t=None,
        t0=None,
        **kwargs,
    ):
        """返回时间 ``t`` 对应的帧号（``int``，落在 ``[0, count)``）。

        若同时给出 ``prev_t``（或 ``t0``）与 ``events`` / ``on_event``
        / ``on_frame``，会顺带计算从先前时间推进到 ``t`` 时跨过的帧事件
        并触发一次（无状态形式，等价于 :func:`events_between`）。
        """
        if t is None:
            raise InvalidTimeError("frame_at 缺少时间参数 t", param="t")

        # 无状态跨帧事件：给出前一个时间点时计算一次。
        previous = prev_t if prev_t is not None else t0
        callback = on_event if on_event is not None else on_frame
        if previous is not None and (events is not None or callback is not None):
            events_between(
                previous,
                t,
                fps if _is_number(fps) else self.fps,
                count if count is not None else self.count,
                events if events is not None else self.events,
                durations=durations if durations is not None else (
                    fps if (fps is not None and not _is_number(fps) and hasattr(fps, "__iter__"))
                    else self.durations
                ),
                loop=loop if loop is not None else self.loop,
                on_event=callback,
            )

        # 兼容 frame_at(t, [durations...]) 的位置写法。
        if (
            durations is None
            and count is None
            and fps is not None
            and not _is_number(fps)
            and hasattr(fps, "__iter__")
        ):
            durations = fps
            fps = None

        if fps is None:
            fps = self.fps
        if count is None:
            count = self.count
        if durations is None:
            durations = self.durations
        if loop is None:
            loop = self.loop
        if pingpong is None:
            pingpong = self.pingpong

        _check_time(t, param="t")

        # ------------------------- 可变累计时长 ------------------------- #
        if durations is not None:
            if (
                getattr(self, "_bounds", None) is not None
                and self.durations is durations
            ):
                bounds = self._bounds
                dlist = self._duration_list
            else:
                dlist, bounds = _normalize_durations(durations, count)
            n = len(dlist)
            total = bounds[-1]
            g = _duration_global_index(t, bounds, total, n)
            if loop:
                return int(g % n)
            return int(min(g, n - 1))

        # --------------------------- 固定帧率 --------------------------- #
        fps = _check_positive_number(
            fps, InvalidFpsError, "fps", nonfinite_cls=NonFiniteFpsError
        )
        n = _check_count(count)

        g = _uniform_global_index(t, fps)
        if g is None:
            period = n / fps
            reduced = math.fmod(t, period)
            if reduced < 0:
                reduced += period
            g = math.floor(reduced * fps)

        if pingpong and n > 1:
            return self._pingpong_fold(g, n)
        if loop:
            return int(g % n)
        return int(min(g, n - 1)) if g < n else n - 1

    @staticmethod
    def _pingpong_fold(global_index, n):
        if n <= 1:
            return 0
        span = 2 * n - 2
        slot = global_index % span
        return int(slot if slot < n else span - slot)

    # 语义化别名。
    frame_index = frame_at
    get_frame = frame_at

    def frame_at_duration(self, t, durations, *, loop=True):
        """按逐帧累计时长取帧。"""
        return self.frame_at(t, durations=durations, loop=loop)

    # ------------------------------------------------------------------ #
    # 帧事件（无状态）
    # ------------------------------------------------------------------ #
    def events_between(
        self,
        t0,
        t1,
        fps=None,
        count=None,
        events=None,
        *,
        durations=None,
        loop=None,
        on_event=None,
    ):
        """见模块级 :func:`events_between`。"""
        if fps is None:
            fps = self.fps
        if count is None:
            count = self.count
        if durations is None:
            durations = self.durations
        if loop is None:
            loop = self.loop
        if events is None:
            events = self.events
        return events_between(
            t0,
            t1,
            fps,
            count,
            events,
            durations=durations,
            loop=loop,
            on_event=on_event,
        )

    crossed_events = events_between
    collect_events = events_between

    def player(self, events=None, **kwargs):
        """基于当前动画配置创建一个 :class:`FramePlayer`。"""
        return FramePlayer(
            self.count,
            self.fps,
            self.durations,
            events if events is not None else self.events,
            loop=self.loop,
            pingpong=self.pingpong,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # 九宫格
    # ------------------------------------------------------------------ #
    def nine_slice(self, src, dst, border):
        """计算九宫格 9 个目标矩形，返回 :class:`NineSliceLayout`。"""
        return _layout_rects(src, dst, border)

    # 常见别名。
    nine_patch = nine_slice
    nine_grid = nine_slice
    slice9 = nine_slice
    layout = nine_slice
    nine_slice_layout = nine_slice


SpriteAnimation = SpriteAnim
Animation = SpriteAnim
Sprite = SpriteAnim


# --------------------------------------------------------------------------- #
# 模块级便捷函数
# --------------------------------------------------------------------------- #
_default_anim = SpriteAnim()


def frame_at(t, fps, count):
    """模块级取帧便捷函数。"""
    return _default_anim.frame_at(t, fps, count)


def nine_slice(src, dst, border):
    """模块级九宫格便捷函数。"""
    return _default_anim.nine_slice(src, dst, border)
