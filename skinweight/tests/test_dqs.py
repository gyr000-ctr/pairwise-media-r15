"""对偶四元数蒙皮（DQS）验收测试。

覆盖约束：
1. 权重限 4 并归一（误差 ≤ 1e-9），输入不被修改；
2. 纯旋转（单骨骼绑定）下 DQS 与 LBS 之差 ≤ 1e-6；
3. 强扭转下 LBS 出现可测体积损失，DQS 保持刚体（体积不变）；
4. 反对偶符号统一（实部点积为负时整体翻转），混合结果与表示无关；
5. 零权重与负权重抛出可区分的异常，非有限输入单独分类；
6. 1 万顶点性能上界 500 ms；
7. 多线程并发安全、结果可复现；
8. 稠密 / 稀疏两种权重布局一致。
"""

import math
import threading

import numpy as np
import pytest

import skinweight
from skinweight import (
    DualQuaternion,
    InfluenceLimitError,
    NegativeWeightError,
    NonFiniteWeightError,
    Skinner,
    SkinWeight,
    ZeroWeightError,
    dual_quaternion_skinning,
    linear_blend_skinning,
)

rng = np.random.default_rng(20261125)


# --------------------------------------------------------------------- 工具
def axis_quat(axis, angle):
    """绕 ``axis`` 旋转 ``angle`` 弧度的单位四元数 [w, x, y, z]。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * axis])


def random_unit_quat():
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def ring(n=96, radius=1.0, x=0.0):
    """x 平面上的正多边形环，用于面积/体积测量。"""
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack(
        [np.full(n, x), radius * np.cos(a), radius * np.sin(a)], axis=1
    )


def polygon_area(points):
    """二维多边形面积（点取 yz 平面投影）。"""
    y, z = points[:, 1], points[:, 2]
    return 0.5 * abs(np.sum(y * np.roll(z, -1) - np.roll(y, -1) * z))


def cylinder_shell_volume(rings):
    """由一串等间距环面围成的圆柱壳体体积（梯形柱近似）。"""
    vol = 0.0
    for i in range(len(rings) - 1):
        a0, a1 = polygon_area(rings[i]), polygon_area(rings[i + 1])
        dx = abs(rings[i + 1][0, 0] - rings[i][0, 0])
        vol += 0.5 * (a0 + a1) * dx
    return vol


# ============================================================ 1. 权重限项归一
class TestWeightConstraints:
    def test_limit_keeps_at_most_four(self):
        w = rng.random((50, 9))
        sk = Skinner([DualQuaternion.from_rigid(random_unit_quat()) for _ in range(9)])
        limited, indices = skinweight.weights.limit_and_normalize_dense(w, 4)
        nonzero_per_row = np.count_nonzero(limited, axis=1)
        assert np.all(nonzero_per_row <= 4)
        assert indices.shape[1] == 4

    def test_normalization_error_below_1e9(self):
        w = rng.random((32, 7))
        limited, _ = skinweight.weights.limit_and_normalize_dense(w, 4)
        row_sums = limited.sum(axis=1)
        assert np.all(np.abs(row_sums - 1.0) <= 1e-9)

    def test_keeps_largest_influences(self):
        w = np.array([[0.4, 0.05, 0.3, 0.1, 0.15]])
        limited, indices = skinweight.weights.limit_and_normalize_dense(w, 4)
        assert set(indices[0].tolist()) == {0, 2, 4, 3}
        assert limited[0, 1] == 0.0

    def test_input_not_mutated(self):
        w = rng.random((8, 6))
        snapshot = w.copy()
        sk = Skinner(
            [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(6)]
        )
        sk.dqs(rng.normal(size=(8, 3)), w)
        assert np.array_equal(w, snapshot)

    def test_skinweight_1d_api_still_normalizes(self):
        out = SkinWeight().limit([0.1, 0.2, 0.3, 0.5, 0.4, 0.2], 4)
        assert len(out) == 4
        assert abs(sum(out) - 1.0) <= 1e-9
        assert all(a >= b for a, b in zip(out, out[1:]))


# ================================================== 2. 纯旋转 DQS 与 LBS 一致
class TestRotationEquivalence:
    def test_single_bone_pure_rotation(self):
        bones = [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(5)]
        sk = Skinner(bones)
        verts = rng.normal(size=(128, 3))
        assignment = rng.integers(0, 5, size=128)
        weights = np.eye(5)[assignment]
        diff = np.abs(sk.dqs(verts, weights) - sk.lbs(verts, weights)).max()
        assert diff <= 1e-6

    def test_single_bone_rigid_with_translation(self):
        bones = [
            DualQuaternion.from_rigid(random_unit_quat(), rng.normal(size=3))
            for _ in range(4)
        ]
        sk = Skinner(bones)
        verts = rng.normal(size=(64, 3))
        weights = np.eye(4)[rng.integers(0, 4, size=64)]
        diff = np.abs(sk.dqs(verts, weights) - sk.lbs(verts, weights)).max()
        assert diff <= 1e-6

    def test_dqs_matches_direct_transform(self):
        q, t = random_unit_quat(), rng.normal(size=3)
        bone = DualQuaternion.from_rigid(q, t)
        verts = rng.normal(size=(32, 3))
        weights = np.ones((32, 1))
        got = Skinner([bone]).dqs(verts, weights)
        expect = bone.transform_points(verts)
        assert np.allclose(got, expect, atol=1e-12)


# ================================================ 3. 强扭转：LBS 体积损失
class TestTwistVolumeLoss:
    @pytest.mark.parametrize("angle_deg", [60.0, 85.0, 120.0])
    def test_lbs_shrinks_dqs_preserves(self, angle_deg):
        angle = math.radians(angle_deg)
        # 两块骨骼绕 x 轴反向扭转，经典 candy-wrapper 场景。
        bones = [
            DualQuaternion.from_rigid(axis_quat([1, 0, 0], angle)),
            DualQuaternion.from_rigid(axis_quat([1, 0, 0], -angle)),
        ]
        sk = Skinner(bones)
        rings = [ring(x=x) for x in (-1.0, 0.0, 1.0)]
        verts = np.vstack(rings)
        weights = np.tile([0.5, 0.5], (verts.shape[0], 1))

        original = cylinder_shell_volume(rings)
        lbs_out = sk.lbs(verts, weights).reshape(len(rings), -1, 3)
        dqs_out = sk.dqs(verts, weights).reshape(len(rings), -1, 3)

        lbs_volume = cylinder_shell_volume(list(lbs_out))
        dqs_volume = cylinder_shell_volume(list(dqs_out))

        # LBS 必须出现可测体积损失（yz 截面各向同性缩放 cos(angle)，
        # 面积/体积比值为 cos²(angle)）。
        assert lbs_volume < original * 0.95
        expected_ratio = math.cos(angle) ** 2
        assert abs(lbs_volume / original - expected_ratio) < 2e-2
        # DQS 保持刚体：体积与原体积一致到浮点精度。
        assert abs(dqs_volume - original) / original < 1e-9

    def test_lbs_blend_matrix_singular_at_90_deg_twist(self):
        # 绕同一轴向相反方向各转约 90°：LBS 混合矩阵在该平面坍缩，det≈0。
        angle = math.radians(89.5)
        bones = [
            DualQuaternion.from_rigid(axis_quat([0, 0, 1], angle)),
            DualQuaternion.from_rigid(axis_quat([0, 0, 1], -angle)),
        ]
        sk = Skinner(bones)
        w = np.array([[0.5, 0.5]])
        blended = sk.lbs_blended_matrices(w)[0, :3, :3]
        assert np.linalg.det(blended) < 0.05  # 近乎坍缩
        # 同样输入下 DQS 输出仍是合法旋转（距离保持）。
        pts = rng.normal(size=(16, 3))
        ww = np.tile([0.5, 0.5], (16, 1))
        out = sk.dqs(pts, ww)
        d0 = np.linalg.norm(pts[0] - pts[1])
        d1 = np.linalg.norm(out[0] - out[1])
        assert abs(d0 - d1) < 1e-12


# ================================================ 4. 反对偶符号统一
class TestAntipodalUnification:
    def test_sign_flip_invariance(self):
        bones = [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(6)]
        # 对部分骨骼取 q -> -q（同一旋转的另一种四元数表示）。
        flipped = [
            DualQuaternion.from_rigid(-b.real) if i % 2 == 0 else b
            for i, b in enumerate(bones)
        ]
        verts = rng.normal(size=(40, 3))
        weights = rng.random((40, 6))
        weights /= weights.sum(axis=1, keepdims=True)
        a = Skinner(bones).dqs(verts, weights)
        b = Skinner(flipped).dqs(verts, weights)
        assert np.allclose(a, b, atol=1e-12)

    def test_naive_blend_without_unification_is_wrong(self):
        # 证明符号统一确实在起作用。两块骨骼的旋转本身几何上接近
        # （+120° 与 +121°），但第二块取了双重覆盖的相反四元数表示，
        # 实部点积为负 ⇒ 朴素混合被错误抵消。
        q1 = axis_quat([1, 0, 0], math.radians(120.0))
        q2 = -axis_quat([1, 0, 0], math.radians(121.0))
        assert np.dot(q1, q2) < -0.99
        bones = [DualQuaternion.from_rigid(q1), DualQuaternion.from_rigid(q2)]
        sk = Skinner(bones)
        verts = ring(n=32)
        weights = np.tile([0.5, 0.5], (verts.shape[0], 1))

        # 朴素混合（不做符号统一）：实部相消，几乎为零，无法给出单位旋转。
        naive_real = 0.5 * sk._real[0] + 0.5 * sk._real[1]
        assert np.linalg.norm(naive_real) < 0.05

        # 引擎内部做符号统一后，混合结果仍是合法单位四元数，面积保持。
        corrected = sk._blend_dual_quaternions(*sk._gather(weights, None))[0]
        assert np.allclose(np.linalg.norm(corrected, axis=1), 1.0)
        correct = sk.dqs(verts, weights)
        assert abs(polygon_area(correct) - polygon_area(verts)) < 1e-9

    def test_dual_part_flips_together_with_real(self):
        # 带平移的骨骼翻转实部时，对偶部必须同步翻转，否则平移会错。
        q, t = random_unit_quat(), rng.normal(size=3)
        bones = [
            DualQuaternion.from_rigid(q, t),
            DualQuaternion.from_rigid(-q, -t * 0.0 + t),  # 同一刚体的 -q 表示
        ]
        sk = Skinner(bones)
        verts = rng.normal(size=(10, 3))
        weights = np.tile([0.5, 0.5], (10, 1))
        out = sk.dqs(verts, weights)
        # 两块骨骼是同一变换，混合后必须等于该变换本身。
        assert np.allclose(out, bones[0].transform_points(verts), atol=1e-12)


# ================================================ 5. 错误分类可区分
class TestErrorClassification:
    def test_zero_weight_distinct_from_negative(self):
        sk = Skinner([DualQuaternion.from_rigid(random_unit_quat()) for _ in range(3)])
        verts = np.zeros((1, 3))
        with pytest.raises(ZeroWeightError):
            sk.dqs(verts, np.zeros((1, 3)))
        with pytest.raises(NegativeWeightError):
            sk.dqs(verts, np.array([[-0.2, 0.6, 0.6]]))

    def test_both_are_value_errors(self):
        for exc in (ZeroWeightError, NegativeWeightError, NonFiniteWeightError):
            assert issubclass(exc, ValueError)

    def test_non_finite_inputs(self):
        sk = Skinner([DualQuaternion.from_rigid(random_unit_quat()) for _ in range(2)])
        verts = np.zeros((1, 3))
        with pytest.raises(NonFiniteWeightError):
            sk.dqs(verts, np.array([[float("nan"), 1.0]]))
        with pytest.raises(NonFiniteWeightError):
            sk.dqs(verts, np.array([[float("inf"), 1.0]]))

    def test_1d_api_error_types(self):
        with pytest.raises(ZeroWeightError):
            SkinWeight().normalize([0.0, 0.0])
        with pytest.raises(NegativeWeightError):
            SkinWeight().normalize([-0.1, 0.5])
        with pytest.raises(NonFiniteWeightError):
            SkinWeight().normalize([float("nan"), 0.5])
        with pytest.raises(InfluenceLimitError):
            SkinWeight().limit([0.5, 0.5], 0)
        with pytest.raises(ZeroWeightError):
            SkinWeight().normalize([])

    def test_negative_reported_before_zero_sum(self):
        # 同时含负值与零和时，负权重分类优先，保证分类确定。
        with pytest.raises(NegativeWeightError):
            SkinWeight().normalize([-0.3, 0.3])


# ================================================ 6. 性能：1 万顶点 ≤ 500ms
class TestPerformance:
    def test_ten_thousand_vertices_budget(self):
        import time

        bone_count = 12
        bones = [
            DualQuaternion.from_rigid(random_unit_quat(), rng.normal(size=3) * 0.1)
            for _ in range(bone_count)
        ]
        sk = Skinner(bones, max_influence=4)
        verts = rng.normal(size=(10_000, 3))
        weights = rng.random((10_000, bone_count))
        weights /= weights.sum(axis=1, keepdims=True)

        # 预热（首次调用含分配与懒初始化）。
        sk.dqs(verts, weights)
        start = time.perf_counter()
        out = sk.dqs(verts, weights)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert out.shape == (10_000, 3)
        assert np.all(np.isfinite(out))
        assert elapsed_ms <= 500.0, f"DQS 耗时 {elapsed_ms:.1f} ms 超过 500 ms 预算"


# ================================================ 7. 并发安全
class TestConcurrencySafety:
    def test_parallel_calls_identical(self):
        bones = [
            DualQuaternion.from_rigid(random_unit_quat(), rng.normal(size=3))
            for _ in range(8)
        ]
        sk = Skinner(bones)
        verts = rng.normal(size=(500, 3))
        weights = rng.random((500, 8))
        weights /= weights.sum(axis=1, keepdims=True)
        expected = sk.dqs(verts, weights)

        results, errors = [], []

        def worker():
            try:
                for _ in range(20):
                    results.append(sk.dqs(verts, weights))
            except Exception as exc:  # pragma: no cover - 并发失败路径
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(results) == 8 * 20
        for r in results:
            assert np.array_equal(r, expected)

    def test_shared_skinner_with_disjoint_inputs(self):
        bones = [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(4)]
        sk = Skinner(bones)
        outs = [None] * 4

        def worker(i):
            v = rng.normal(size=(100, 3))
            w = rng.random((100, 4))
            w /= w.sum(axis=1, keepdims=True)
            outs[i] = sk.dqs(v, w)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(o is not None and np.all(np.isfinite(o)) for o in outs)


# ================================================ 8. 稀疏/稠密布局与模块级 API
class TestSparseAndModuleAPI:
    def test_sparse_matches_dense(self):
        bones = [
            DualQuaternion.from_rigid(random_unit_quat(), rng.normal(size=3))
            for _ in range(5)
        ]
        sk = Skinner(bones)
        verts = rng.normal(size=(30, 3))
        dense = rng.random((30, 5))
        dense /= dense.sum(axis=1, keepdims=True)

        limited, indices = skinweight.weights.limit_and_normalize_dense(dense, 4)
        rows = np.arange(30)[:, None]
        sparse_w = limited[rows, indices]
        dense_out = sk.dqs(verts, dense)
        sparse_out = sk.dqs(verts, sparse_w, indices=indices)
        assert np.allclose(dense_out, sparse_out, atol=1e-12)

    def test_module_level_functions(self):
        bones = [
            DualQuaternion.from_rigid(random_unit_quat()) for _ in range(3)
        ]
        verts = rng.normal(size=(12, 3))
        w = rng.random((12, 3))
        w /= w.sum(axis=1, keepdims=True)
        assert np.allclose(
            dual_quaternion_skinning(bones, verts, w),
            Skinner(bones).dqs(verts, w),
        )
        assert np.allclose(
            linear_blend_skinning(bones, verts, w),
            Skinner(bones).lbs(verts, w),
        )

    def test_matrix_and_tuple_bone_forms(self):
        q, t = random_unit_quat(), rng.normal(size=3)
        dq = DualQuaternion.from_rigid(q, t)
        verts = rng.normal(size=(8, 3))
        w = np.ones((8, 1))
        a = Skinner([dq]).dqs(verts, w)
        b = Skinner([(q, t)]).dqs(verts, w)
        c = Skinner([dq.as_matrix()]).dqs(verts, w)
        assert np.allclose(a, b, atol=1e-12)
        assert np.allclose(a, c, atol=1e-12)


# ================================================ 9. 稀疏布局边界与错误路径
class TestSparseEdgeCases:
    def test_sparse_zero_and_negative_distinct(self):
        sk = Skinner(
            [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(3)]
        )
        verts = np.zeros((1, 3))
        with pytest.raises(ZeroWeightError):
            sk.dqs(verts, np.array([[0.0, 0.0]]), indices=np.array([[0, 1]]))
        with pytest.raises(NegativeWeightError):
            sk.dqs(verts, np.array([[0.7, -0.2]]), indices=np.array([[0, 1]]))

    def test_sparse_out_of_range_index(self):
        sk = Skinner(
            [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(3)]
        )
        verts = np.zeros((1, 3))
        with pytest.raises(IndexError):
            sk.dqs(verts, np.array([[0.6, 0.4]]), indices=np.array([[0, 7]]))

    def test_dense_zero_row_among_good_rows(self):
        sk = Skinner(
            [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(2)]
        )
        verts = np.zeros((2, 3))
        weights = np.array([[0.5, 0.5], [0.0, 0.0]])
        with pytest.raises(ZeroWeightError):
            sk.dqs(verts, weights)

    def test_limit_respects_custom_max_influence(self):
        bones = [DualQuaternion.from_rigid(random_unit_quat()) for _ in range(6)]
        sk = Skinner(bones, max_influence=2)
        weights = rng.random((4, 6))
        weights /= weights.sum(axis=1, keepdims=True)
        picked, idx = sk._gather(weights, None)
        assert idx.shape[1] == 2
        assert picked.shape[1] == 2
        assert np.allclose(picked.sum(axis=1), 1.0)

    def test_bad_bone_and_vertex_shapes(self):
        from skinweight import TransformError

        with pytest.raises(TransformError):
            Skinner([])
        sk = Skinner([DualQuaternion.from_rigid(random_unit_quat())])
        with pytest.raises(TransformError):
            sk.dqs(rng.normal(size=(4, 2)), np.ones((4, 1)))
        with pytest.raises(Exception):
            sk.dqs(rng.normal(size=(4, 3)), np.ones((4, 2)))

    def test_dual_quaternion_norm_invariants(self):
        dq = DualQuaternion.from_rigid(random_unit_quat(), rng.normal(size=3))
        assert abs(np.linalg.norm(dq.real) - 1.0) < 1e-12
        assert abs(np.dot(dq.real, dq.dual)) < 1e-12
        # 归一化幂等。
        again = dq.normalize()
        assert np.allclose(again.real, dq.real, atol=1e-12)
        assert np.allclose(again.dual, dq.dual, atol=1e-12)
