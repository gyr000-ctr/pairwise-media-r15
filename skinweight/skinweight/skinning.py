"""对偶四元数蒙皮（DQS）与线性混合蒙皮（LBS）。

实现要点
--------
1. **限项归一**：每顶点只保留权重最大的前 ``max_influence``（默认 4）个
   骨骼影响并重新归一，归一残差吸收到最大分量，保证求和误差 ≤ 1e-9。
2. **反对偶符号统一**：混合前以每个顶点的首个（最大权重）骨骼为基准，
   凡与基准实部点积为负的对偶四元数整体取负，消除 ``q`` 与 ``-q``
   表示同一旋转带来的双重覆盖（double-cover）歧义。
3. **DQS 归一化**：按 Kavan et al. 2007 的公式
   ``real /= |real|``、``dual /= |real|`` 重新单位化。
4. **可复现且线程安全**：类实例只保存只读的骨骼数据，所有调用都是
   无状态纯函数式计算，NumPy 调用期间释放 GIL，可安全并发调用。
"""

from __future__ import annotations

import numpy as np

from .dualquaternion import DualQuaternion, quat_conj, quat_mul, quat_rotate
from .errors import SkinWeightError, TransformError
from .weights import (
    limit_and_normalize_dense,
)

__all__ = [
    "Skinner",
    "dual_quaternion_skinning",
    "linear_blend_skinning",
]


def _coerce_bones(bones):
    """把多种骨骼表示统一成 (real, dual, matrix) 三个批量数组。"""
    if bones is None or len(bones) == 0:
        raise TransformError("至少需要一块骨骼")
    reals, duals, matrices = [], [], []
    for bone in bones:
        if isinstance(bone, DualQuaternion):
            dq = bone
        elif isinstance(bone, np.ndarray) and bone.shape == (4, 4):
            dq = DualQuaternion.from_matrix(bone)
        elif (
            isinstance(bone, tuple)
            and len(bone) == 2
            and np.asarray(bone[0]).shape == (4,)
            and np.asarray(bone[1]).shape == (3,)
        ):
            dq = DualQuaternion.from_rigid(bone[0], bone[1])
        else:
            raise TransformError(
                "骨骼必须是 DualQuaternion、(旋转四元数, 平移) 二元组或 4x4 矩阵"
            )
        reals.append(dq.real)
        duals.append(dq.dual)
        matrices.append(dq.as_matrix())
    real_arr = np.ascontiguousarray(reals, dtype=float)
    dual_arr = np.ascontiguousarray(duals, dtype=float)
    mat_arr = np.ascontiguousarray(matrices, dtype=float)
    if not (np.all(np.isfinite(real_arr)) and np.all(np.isfinite(dual_arr))):
        raise TransformError("骨骼变换包含非有限分量")
    # 只读存储：实例在多线程间共享也不会被意外改写。
    real_arr.setflags(write=False)
    dual_arr.setflags(write=False)
    mat_arr.setflags(write=False)
    return real_arr, dual_arr, mat_arr


def _prepare_dense(weights, bone_count, max_influence):
    """稠密权重 → (有限权重 (V,K), 骨骼索引 (V,K))，每顶点至多 K 项。"""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[1] != bone_count:
        raise SkinWeightError(
            f"稠密权重形状 {w.shape} 与骨骼数 {bone_count} 不匹配"
        )
    limited, indices = limit_and_normalize_dense(w, max_influence)
    rows = np.arange(indices.shape[0])[:, None]
    picked = limited[rows, indices]
    # 归一残差吸收到每行最大分量，求和严格为 1。
    picked = picked.copy()
    picked[:, 0] += 1.0 - picked.sum(axis=1)
    return picked, indices


class Skinner:
    """持有骨骼层级的蒙皮器（无状态、线程安全）。

    Parameters
    ----------
    bones:
        骨骼序列，元素可以是 :class:`DualQuaternion`、
        ``(旋转四元数 [w,x,y,z], 平移 [x,y,z])`` 二元组或 4x4 齐次矩阵。
    max_influence:
        每顶点保留的最大骨骼影响数，默认 4，必须为正整数。
    """

    def __init__(self, bones, max_influence=4):
        if isinstance(max_influence, bool) or not isinstance(
            max_influence, (int, np.integer)
        ) or max_influence <= 0:
            from .errors import InfluenceLimitError

            raise InfluenceLimitError("max_influence 必须是正整数")
        self.max_influence = int(max_influence)
        self._real, self._dual, self._matrix = _coerce_bones(bones)
        self.bone_count = self._real.shape[0]

    # ------------------------------------------------------------- 内部工具
    def _gather(self, weights, indices):
        if indices is None:
            return _prepare_dense(
                weights, self.bone_count, self.max_influence
            )
        idx = np.asarray(indices, dtype=np.int64)
        w = np.asarray(weights, dtype=float)
        if w.shape != idx.shape:
            raise SkinWeightError("稀疏 weights 与 indices 形状必须一致")
        if not np.all(np.isfinite(w)):
            from .errors import NonFiniteWeightError

            raise NonFiniteWeightError("权重包含 NaN 或 Inf")
        if np.any(w < 0.0):
            from .errors import NegativeWeightError

            raise NegativeWeightError("权重不允许为负")
        totals = w.sum(axis=1)
        if np.any(totals <= 0.0):
            from .errors import ZeroWeightError

            raise ZeroWeightError("存在权重和为零的顶点")
        if np.any(idx < 0) or np.any(idx >= self.bone_count):
            raise IndexError("骨骼索引越界")
        k = min(self.max_influence, w.shape[1])
        order = np.argsort(-w[:, :k], axis=1, kind="stable")
        idx = np.take_along_axis(idx[:, :k], order, axis=1)
        picked = np.take_along_axis(w[:, :k], order, axis=1)
        picked = picked / picked.sum(axis=1, keepdims=True)
        picked[:, 0] += 1.0 - picked.sum(axis=1)
        return picked, idx

    def _blend_dual_quaternions(self, picked, idx):
        """符号统一 + 加权混合 + 单位化，返回混合后的 (real, dual)。"""
        bone_real = self._real[idx]  # (V, K, 4)
        bone_dual = self._dual[idx]
        # 反对偶符号统一：与首影响实部点积为负则整体翻转。
        anchor = bone_real[:, :1, :]  # (V, 1, 4)
        dots = np.sum(bone_real * anchor, axis=-1, keepdims=True)  # (V, K, 1)
        signs = np.where(dots < 0.0, -1.0, 1.0)
        w = picked[:, :, None]  # (V, K, 1)
        blended_real = np.sum(w * signs * bone_real, axis=1)
        blended_dual = np.sum(w * signs * bone_dual, axis=1)
        # Kavan DQS 归一化。
        norm = np.linalg.norm(blended_real, axis=-1, keepdims=True)
        if np.any(norm == 0.0):
            raise SkinWeightError("对偶四元数混合退化为零")
        return blended_real / norm, blended_dual / norm

    # ----------------------------------------------------------------- DQS
    def dqs(self, vertices, weights, indices=None):
        """对偶四元数蒙皮。

        Parameters
        ----------
        vertices:
            静止姿态顶点，形状 ``(V, 3)``。
        weights:
            稠密权重 ``(V, B)``（此时 ``indices`` 为 ``None``），
            或与 ``indices`` 同形的稀疏权重 ``(V, K)``。
        indices:
            稀疏布局下每顶点的骨骼索引 ``(V, K)``。
        """
        verts = self._check_vertices(vertices)
        picked, idx = self._gather(weights, indices)
        real, dual = self._blend_dual_quaternions(picked, idx)
        # 平移：t = 2 * dual * conj(real) 的向量部。
        translation = 2.0 * quat_mul(dual, quat_conj(real))[..., 1:]
        return quat_rotate(real, verts) + translation

    # ----------------------------------------------------------------- LBS
    def lbs(self, vertices, weights, indices=None):
        """线性混合蒙皮：先按权重混合 4x4 变换矩阵，再施加到顶点。"""
        verts = self._check_vertices(vertices)
        picked, idx = self._gather(weights, indices)
        mats = self._matrix[idx]  # (V, K, 4, 4)
        blended = np.sum(picked[:, :, None, None] * mats, axis=1)  # (V, 4, 4)
        rotated = np.einsum("vij,vj->vi", blended[:, :3, :3], verts)
        return rotated + blended[:, :3, 3]

    def lbs_blended_matrices(self, weights, indices=None):
        """返回 LBS 的每顶点混合矩阵 ``(V, 4, 4)``，供体积/行列式分析。"""
        picked, idx = self._gather(weights, indices)
        mats = self._matrix[idx]
        return np.sum(picked[:, :, None, None] * mats, axis=1)

    @staticmethod
    def _check_vertices(vertices):
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3:
            raise TransformError("顶点数组形状必须为 (顶点数, 3)")
        if not np.all(np.isfinite(v)):
            raise TransformError("顶点包含非有限分量")
        return np.ascontiguousarray(v)


# ---------------------------------------------------------------- 模块级函数
def dual_quaternion_skinning(bones, vertices, weights, indices=None, max_influence=4):
    """DQS 一次性函数封装。"""
    return Skinner(bones, max_influence=max_influence).dqs(vertices, weights, indices)


def linear_blend_skinning(bones, vertices, weights, indices=None, max_influence=4):
    """LBS 一次性函数封装。"""
    return Skinner(bones, max_influence=max_influence).lbs(vertices, weights, indices)
