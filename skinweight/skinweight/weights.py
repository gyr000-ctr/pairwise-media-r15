"""蒙皮权重的限项、归一与校验。

提供两层接口：

- :class:`SkinWeight`：面向一维权重列表的轻量接口（保留历史 API）。
- 本模块内的 NumPy 向量化函数：供 DQS/LBS 蒙皮器复用，支持
  稠密（``(V, B)``）与稀疏（每顶点 ``(索引, 权重)`` 列表）两种布局。

错误分类（均为 :class:`ValueError` 子类，可分别捕获）：

- 非有限分量（NaN/Inf）→ :class:`NonFiniteWeightError`
- 负权重               → :class:`NegativeWeightError`
- 零权重和 / 空输入     → :class:`ZeroWeightError`
- ``max_influence`` 非法 → :class:`InfluenceLimitError`
"""

from __future__ import annotations

import numpy as np

from .errors import (
    InfluenceLimitError,
    NegativeWeightError,
    NonFiniteWeightError,
    ZeroWeightError,
)

__all__ = ["SkinWeight", "validate_1d_weights", "limit_and_normalize_dense", "normalize_dense"]


def _check_influence_limit(max_influence):
    if isinstance(max_influence, bool) or not isinstance(max_influence, (int, np.integer)):
        raise InfluenceLimitError("max_influence 必须是正整数")
    if max_influence <= 0:
        raise InfluenceLimitError("max_influence 必须为正整数")
    return int(max_influence)


def validate_1d_weights(weights):
    """校验一维可迭代权重，返回 ``np.ndarray``。

    校验顺序固定为 非有限 → 负 → 零和，保证错误可区分且确定。
    """
    if weights is None:
        raise ZeroWeightError("权重为空")
    try:
        arr = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"权重无法转换为浮点数组: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError("权重必须是一维序列")
    if arr.size == 0:
        raise ZeroWeightError("空权重：顶点没有任何骨骼影响")
    if not np.all(np.isfinite(arr)):
        raise NonFiniteWeightError("权重包含 NaN 或 Inf")
    if np.any(arr < 0.0):
        raise NegativeWeightError("权重不允许为负")
    if np.sum(arr) <= 0.0:
        raise ZeroWeightError("顶点权重和为零")
    return arr


class SkinWeight:
    """一维权重列表的限项与归一（历史 API，纯函数、无状态）。"""

    def normalize(self, weights):
        """归一化权重，返回新 ``list[float]``，满足和为 1（误差 ≤ 1e-9）。

        零权重顶点、负权重、非有限输入分别抛出可区分的异常。
        """
        arr = validate_1d_weights(weights)
        total = arr.sum()
        out = arr / total
        # 把归一残差吸收到最大分量，保证求和在浮点意义下严格为 1.0。
        out = out.copy()
        out[int(np.argmax(out))] += 1.0 - out.sum()
        return out.tolist()

    def limit(self, weights, max_influence=4):
        """保留权重最大的前 ``max_influence`` 项并重新归一。

        返回按权重降序排列的新列表；输入不被修改。
        """
        k = _check_influence_limit(max_influence)
        arr = validate_1d_weights(weights)
        k = min(k, arr.size)
        # argpartition 取前 k 大，再按权重降序稳定排列。
        idx = np.argpartition(arr, k - 1)[-k:]
        idx = idx[np.argsort(-arr[idx], kind="stable")]
        picked = arr[idx]
        out = picked / picked.sum()
        out = out.copy()
        out[0] += 1.0 - out.sum()
        return out.tolist()


# --------------------------------------------------------------------- 向量化
def normalize_dense(matrix):
    """校验并归一化稠密权重矩阵 ``(V, B)``，返回新数组。"""
    w = np.asarray(matrix, dtype=float)
    if w.ndim != 2:
        raise ValueError("稠密权重必须是形状为 (顶点数, 骨骼数) 的二维数组")
    if w.size == 0:
        raise ZeroWeightError("空权重：顶点没有任何骨骼影响")
    if not np.all(np.isfinite(w)):
        raise NonFiniteWeightError("权重包含 NaN 或 Inf")
    if np.any(w < 0.0):
        raise NegativeWeightError("权重不允许为负")
    totals = w.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ZeroWeightError("存在权重和为零的顶点")
    return w / totals[:, None]


def limit_and_normalize_dense(matrix, max_influence=4):
    """每顶点保留前 ``max_influence`` 个最大权重并重新归一（稠密布局）。

    被裁掉的位置置零，返回 ``(limited (V, B), indices (V, K))``，
    其中 ``K = min(max_influence, B)``，索引按权重降序排列。
    """
    k = _check_influence_limit(max_influence)
    w = normalize_dense(matrix)
    v_count, b_count = w.shape
    k = min(k, b_count)
    # 取前 k 大（argpartition），再在每行内部降序排列。
    part = np.argpartition(-w, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(v_count)[:, None]
    part_vals = w[rows, part]
    order = np.argsort(-part_vals, axis=1, kind="stable")
    indices = np.take_along_axis(part, order, axis=1)
    limited = np.zeros_like(w)
    rows = np.arange(v_count)[:, None]
    picked = w[rows, indices]
    sums = picked.sum(axis=1, keepdims=True)
    picked = picked / sums
    # 归一残差吸收到每行最大分量。
    picked[:, 0] += 1.0 - picked.sum(axis=1)
    limited[rows, indices] = picked
    return limited, indices


def validate_sparse(influences, bone_count):
    """校验稀疏权重（每顶点 ``[(bone_index, weight), ...]``）。

    会丢弃零权重项并按权重降序排序，返回紧凑后的列表；同时做范围、
    非有限、负权重与零和校验。
    """
    if influences is None or len(influences) == 0:
        raise ZeroWeightError("空权重：顶点没有任何骨骼影响")
    compacted = []
    for row in influences:
        if row is None or len(row) == 0:
            raise ZeroWeightError("存在没有任何骨骼影响的顶点")
        entries = []
        for item in row:
            try:
                bone, weight = item
            except (TypeError, ValueError) as exc:
                raise ValueError("稀疏权重每项必须是 (骨骼索引, 权重) 二元组") from exc
            weight = float(weight)
            if not np.isfinite(weight):
                raise NonFiniteWeightError("权重包含 NaN 或 Inf")
            if weight < 0.0:
                raise NegativeWeightError("权重不允许为负")
            if not isinstance(bone, (int, np.integer)) or isinstance(bone, bool):
                raise ValueError("骨骼索引必须是整数")
            if not (0 <= int(bone) < bone_count):
                raise IndexError(f"骨骼索引越界: {bone} 不在 [0, {bone_count})")
            if weight > 0.0:
                entries.append((int(bone), weight))
        if not entries:
            raise ZeroWeightError("存在权重和为零的顶点")
        entries.sort(key=lambda e: e[1], reverse=True)
        compacted.append(entries)
    return compacted
