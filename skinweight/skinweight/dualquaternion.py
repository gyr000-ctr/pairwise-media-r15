"""对偶四元数（Dual Quaternion）与单位四元数运算。

约定：

- 四元数采用 ``[w, x, y, z]`` 标量在前的布局。
- 对偶四元数由实部 ``real``（单位四元数，描述旋转）与对偶部 ``dual``
  （描述平移）两个 4 分量组成，满足单位对偶四元数条件
  ``dot(real, real) = 1`` 且 ``dot(real, dual) = 0``。
- 刚体变换 ``(R, t)`` 对应的对偶四元数：

      real = q
      dual = 0.5 * t_quat * q

  其中 ``t_quat = [0, tx, ty, tz]``。

全部运算基于 NumPy，按批处理时首维为批次维，不持有任何可写全局状态。
"""

from __future__ import annotations

import numpy as np

from .errors import TransformError

__all__ = ["DualQuaternion", "quat_rotate", "quat_mul", "quat_conj"]


def quat_mul(a, b):
    """Hamilton 乘积。支持最后一维为 4 的批量数组。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q):
    """四元数共轭 ``[w, -x, -y, -z]``。"""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_rotate(q, points):
    """用单位四元数 ``q`` 旋转一批点。

    向量化公式（等价于 ``q * p * q^{-1}``，省掉两次完整四元数乘法）::

        v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    """
    q = np.asarray(q, dtype=float)
    points = np.asarray(points, dtype=float)
    axis = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * np.cross(axis, points)
    return points + w * t + np.cross(axis, t)


def _rotmat_to_quat(m):
    """3x3 旋转矩阵（或批量矩阵）→ ``[w, x, y, z]``，Shepperd 稳定分支法。"""
    m = np.asarray(m, dtype=float)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(m.shape[:-2] + (4,), dtype=float)

    c0 = trace > 0.0
    if np.any(c0):
        s = np.sqrt(trace[c0] + 1.0) * 2.0  # s = 4*w
        q[c0, 0] = 0.25 * s
        q[c0, 1] = (m[c0, 2, 1] - m[c0, 1, 2]) / s
        q[c0, 2] = (m[c0, 0, 2] - m[c0, 2, 0]) / s
        q[c0, 3] = (m[c0, 1, 0] - m[c0, 0, 1]) / s

    c1 = (~c0) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    if np.any(c1):
        s = np.sqrt(1.0 + m[c1, 0, 0] - m[c1, 1, 1] - m[c1, 2, 2]) * 2.0  # s = 4*x
        q[c1, 0] = (m[c1, 2, 1] - m[c1, 1, 2]) / s
        q[c1, 1] = 0.25 * s
        q[c1, 2] = (m[c1, 0, 1] + m[c1, 1, 0]) / s
        q[c1, 3] = (m[c1, 0, 2] + m[c1, 2, 0]) / s

    c2 = (~c0) & (~c1) & (m[..., 1, 1] > m[..., 2, 2])
    if np.any(c2):
        s = np.sqrt(1.0 + m[c2, 1, 1] - m[c2, 0, 0] - m[c2, 2, 2]) * 2.0  # s = 4*y
        q[c2, 0] = (m[c2, 0, 2] - m[c2, 2, 0]) / s
        q[c2, 1] = (m[c2, 0, 1] + m[c2, 1, 0]) / s
        q[c2, 2] = 0.25 * s
        q[c2, 3] = (m[c2, 1, 2] + m[c2, 2, 1]) / s

    c3 = ~(c0 | c1 | c2)
    if np.any(c3):
        s = np.sqrt(1.0 + m[c3, 2, 2] - m[c3, 0, 0] - m[c3, 1, 1]) * 2.0  # s = 4*z
        q[c3, 0] = (m[c3, 1, 0] - m[c3, 0, 1]) / s
        q[c3, 1] = (m[c3, 0, 2] + m[c3, 2, 0]) / s
        q[c3, 2] = (m[c3, 1, 2] + m[c3, 2, 1]) / s
        q[c3, 3] = 0.25 * s

    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n == 0.0):
        raise TransformError("退化的旋转矩阵：无法提取单位四元数")
    return q / n


class DualQuaternion:
    """表示刚体变换的单位对偶四元数。

    Parameters
    ----------
    real:
        旋转四元数 ``[w, x, y, z]``，内部会归一化为单位四元数。
    dual:
        对偶部四元数；缺省时构造纯旋转变换。直接传入时会按单位对偶四元数
        约束（``real·dual = 0``）校正。
    """

    __slots__ = ("real", "dual")

    def __init__(self, real, dual=None):
        real = np.asarray(real, dtype=float)
        if real.shape[-1:] != (4,):
            raise TransformError("实部必须是长度为 4 的四元数 [w, x, y, z]")
        if not np.all(np.isfinite(real)):
            raise TransformError("实部包含非有限分量")
        norm = np.linalg.norm(real, axis=-1, keepdims=True)
        if np.any(norm == 0.0):
            raise TransformError("零四元数无法归一化")
        real = real / norm

        if dual is None:
            dual = np.zeros_like(real)
        else:
            dual = np.asarray(dual, dtype=float)
            if dual.shape != real.shape:
                raise TransformError("对偶部与实部形状不一致")
            if not np.all(np.isfinite(dual)):
                raise TransformError("对偶部包含非有限分量")
            # 投影到满足 real·dual = 0 的切平面，再随实部一起归一。
            dual = dual - np.sum(real * dual, axis=-1, keepdims=True) * real
        self.real = real
        self.dual = dual

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_rigid(cls, rotation, translation=(0.0, 0.0, 0.0)):
        """由旋转四元数与平移向量构造。

        Parameters
        ----------
        rotation:
            单位四元数 ``[w, x, y, z]``（不必预先归一）。
        translation:
            平移向量 ``[tx, ty, tz]``。
        """
        q = np.asarray(rotation, dtype=float)
        t = np.asarray(translation, dtype=float)
        if t.shape[-1:] != (3,) or not np.all(np.isfinite(t)):
            raise TransformError("平移必须是长度为 3 的有限向量")
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        t_quat = np.concatenate([np.zeros(t.shape[:-1] + (1,)), t], axis=-1)
        dual = 0.5 * quat_mul(t_quat, q)
        return cls(q, dual)

    @classmethod
    def from_matrix(cls, matrix):
        """由 4x4（或批量 4x4）刚体齐次矩阵构造。"""
        m = np.asarray(matrix, dtype=float)
        if m.shape[-2:] != (4, 4):
            raise TransformError("from_matrix 需要 4x4 齐次矩阵")
        if not np.all(np.isfinite(m)):
            raise TransformError("矩阵包含非有限分量")
        q = _rotmat_to_quat(m[..., :3, :3])
        # 平移直接取齐次矩阵第四列；旋转由 Shepperd 分支法稳定提取。
        translation = m[..., :3, 3]
        return cls.from_rigid(q, translation)

    # ------------------------------------------------------------------ 性质
    @property
    def rotation(self):
        """旋转四元数（实部）。"""
        return self.real

    def translation(self):
        """反解平移向量：``t = 2 * dual * conj(real)`` 的向量部。"""
        return quat_mul(self.dual, quat_conj(self.real))[..., 1:] * 2.0

    def as_matrix(self):
        """展开为 4x4（或批量 4x4）齐次刚体矩阵。"""
        q = self.real
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        # 单位四元数下 norm² = 1；显式除以 norm² 以兼容批量中的微小数值偏差。
        s = 2.0 / (w * w + x * x + y * y + z * z)
        m = np.zeros(q.shape[:-1] + (4, 4), dtype=float)
        m[..., 0, 0] = 1.0 - s * (y * y + z * z)
        m[..., 0, 1] = s * (x * y - z * w)
        m[..., 0, 2] = s * (x * z + y * w)
        m[..., 1, 0] = s * (x * y + z * w)
        m[..., 1, 1] = 1.0 - s * (x * x + z * z)
        m[..., 1, 2] = s * (y * z - x * w)
        m[..., 2, 0] = s * (x * z - y * w)
        m[..., 2, 1] = s * (y * z + x * w)
        m[..., 2, 2] = 1.0 - s * (x * x + y * y)
        m[..., :3, 3] = self.translation()
        m[..., 3, 3] = 1.0
        return m

    # ------------------------------------------------------------------ 运算
    def multiply(self, other):
        """对偶四元数乘法（``self ∘ other``）。"""
        return DualQuaternion(
            quat_mul(self.real, other.real),
            quat_mul(self.real, other.dual) + quat_mul(self.dual, other.real),
        )

    def normalize(self):
        """返回单位化后的副本（Kavan 2007 的 DQS 归一化）。"""
        lensq = np.sum(self.real * self.real, axis=-1, keepdims=True)
        norm = np.sqrt(lensq)
        real = self.real / norm
        dual = self.dual / norm
        return DualQuaternion(real, dual)

    def transform_points(self, points):
        """对一批点施加本刚体变换：``R·p + t``。"""
        points = np.asarray(points, dtype=float)
        return quat_rotate(self.real, points) + self.translation()

    def is_close(self, other, atol=1e-10):
        """逐分量近似相等（自动处理实部整体取负的表示歧义）。"""
        a_real, b_real = self.real, other.real
        dot = np.sum(a_real * b_real, axis=-1)
        flip = np.where(dot < 0.0, -1.0, 1.0)[..., None]
        return np.allclose(a_real, flip * b_real, atol=atol) and np.allclose(
            self.dual, flip * other.dual, atol=atol
        )

    def __eq__(self, other):
        if not isinstance(other, DualQuaternion):
            return NotImplemented
        return np.array_equal(self.real, other.real) and np.array_equal(self.dual, other.dual)

    def __repr__(self):
        return f"DualQuaternion(real={self.real.tolist()}, dual={self.dual.tolist()})"
