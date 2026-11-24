"""lightcull 扩展测试：逐条覆盖题面约束。

不复用 ``test_core.py`` 中自相矛盾的 ``tiles`` 用例，改为按题面语义
（块数 = ceil(w/tile) × ceil(h/tile)）逐条验证，并用暴力球-块求交作
为独立判据对照。
"""

import concurrent.futures
import math
import random
import time

import pytest

import lightcull
from lightcull import (InvalidAreaSizeError, InvalidLightError,
                       InvalidMaxLightsError, InvalidTileSizeError,
                       LightCull, LightCullError, NonFiniteLightError)

L = [((0.5, 0.5), 1.0), ((3.5, 3.5), 1.0)]


def _grid(tile, width, height):
    return (math.ceil(width / tile), math.ceil(height / tile))


def _tiles(tile, width, height):
    cols, rows = _grid(tile, width, height)
    out = []
    for row in range(rows):
        y0 = row * tile
        y1 = min(y0 + tile, height)
        for col in range(cols):
            x0 = col * tile
            x1 = min(x0 + tile, width)
            out.append((x0, y0, x1, y1))
    return out


def _brute(lights, tile, width, height):
    """独立的暴力球-块求交，返回每块命中的光源下标（未限项）。"""
    out = []
    for x0, y0, x1, y1 in _tiles(tile, width, height):
        hit = []
        for index, ((cx, cy), r) in enumerate(lights):
            # 球心到闭 AABB 的最近点平方距离。
            nx = min(max(cx, x0), x1)
            ny = min(max(cy, y0), y1)
            if (cx - nx) ** 2 + (cy - ny) ** 2 <= r * r:
                hit.append(index)
        out.append(hit)
    return out


# ---------------------------------------------------------------- 网格与边界
def test_tile_count_matches_ceiling():
    assert LightCull(1.0, 4).tile_count(4, 4) == (4, 4)
    assert LightCull(2.0, 4).tile_count(4, 4) == (2, 2)
    assert LightCull(1.0, 4).tile_count(2, 2) == (2, 2)
    # 不满一个 tile 也至少一块。
    assert LightCull(4.0, 4).tile_count(2, 2) == (1, 1)


def test_output_length_is_rows_times_cols():
    lc = LightCull(1.0, 4)
    assert len(lc.cull(L, 4, 4)) == 16
    assert len(LightCull(2.0, 4).cull(L, 4, 4)) == 4
    assert len(LightCull(1.0, 4).cull([], 2.0, 3.0)) == 6


def test_empty_input_all_tiles_present():
    result = LightCull(1.0, 4).cull([], 2, 2)
    assert result == [[], [], [], []]


@pytest.mark.parametrize("tile,w,h", [
    (1.0, 4.0, 4.0), (0.5, 3.0, 2.0), (2.0, 5.0, 1.0), (1.5, 4.0, 4.0),
    (0.25, 2.0, 2.0), (3.0, 10.0, 7.0), (1.0, 1.0, 1.0),
])
def test_matches_brute_force(tile, w, h):
    rng = random.Random(hash((tile, w, h)) & 0xFFFFFFFF)
    lights = []
    for _ in range(40):
        lights.append((((rng.uniform(-2, w + 2)), rng.uniform(-2, h + 2)),
                       rng.choice([0.0, 0.05, 0.3, 1.0, 2.5])))
    got = LightCull(tile, 10_000).cull(lights, w, h)
    want = _brute(lights, tile, w, h)
    assert [sorted(t) for t in got] == [sorted(t) for t in want]


def test_matches_brute_force_randomized_with_tangency():
    # 固定种子随机对拍，主动构造圆心落在网格线/角点、半径恰为到网格角点
    # 距离的相切情形，验证块边界不漏。
    for seed in range(60):
        rng = random.Random(seed)
        tile = rng.choice([0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.7])
        w = rng.choice([1.0, 2.0, 3.3, 4.0, 5.5, 8.0])
        h = rng.choice([1.0, 2.0, 3.3, 4.0, 5.5, 8.0])
        lights = []
        for _ in range(rng.randint(0, 40)):
            if rng.random() < 0.5:
                cx = rng.randint(-3, int(w / tile) + 3) * tile
                cy = rng.randint(-3, int(h / tile) + 3) * tile
                if rng.random() < 0.5:
                    kx = rng.randint(-3, int(w / tile) + 3) * tile
                    ky = rng.randint(-3, int(h / tile) + 3) * tile
                    r = math.hypot(cx - kx, cy - ky)
                else:
                    r = abs(cx - tile * round(cx / tile))
            else:
                cx = rng.uniform(-3, w + 3)
                cy = rng.uniform(-3, h + 3)
                r = rng.choice([0.0, 1e-12, 0.01, 0.1, 0.5, 1.0, 3.0])
            lights.append(((cx, cy), r))
        got = LightCull(tile, 10 ** 9).cull(lights, w, h)
        want = _brute(lights, tile, w, h)
        assert [sorted(t) for t in got] == [sorted(t) for t in want], \
            "seed %d tile %r area %rx%r" % (seed, tile, w, h)


def test_large_radius_from_outside_hits_all():
    lights = [((500.0, 500.0), 1000.0)]
    got = LightCull(1.0, 10 ** 9).cull(lights, 10, 10)
    assert all(t == [0] for t in got)


def test_tile_boundary_not_missed():
    # 光源恰好落在块边界 / 角点上，相切也必须命中相邻块。
    lights = [((1.0, 1.0), 0.0), ((2.0, 0.0), 0.0), ((0.0, 0.0), 0.0)]
    got = LightCull(1.0, 4).cull(lights, 2, 2)
    want = _brute(lights, 1.0, 2, 2)
    assert [sorted(t) for t in got] == [sorted(t) for t in want]
    # (1,1) 是四个块共享的角点。
    assert [0, 1, 2, 3] == [i for i, t in enumerate(got) if 0 in t]
    # 恰在 (2,0)：列 1 与列 2（若存在）共享；2x2 网格下为右上块。
    assert 1 in got[1]


def test_touching_circle_hits():
    # 半径恰好等于到块边界的距离：相切算命中。
    lights = [((0.4, 0.5), 0.6)]  # 圆心在块0，触及 x=1 边界
    got = LightCull(1.0, 4).cull(lights, 2, 1)
    assert got[0] == [0]
    assert got[1] == [0]


def test_circle_just_short_does_not_hit():
    lights = [((0.4, 0.5), 0.599999)]
    got = LightCull(1.0, 4).cull(lights, 2, 1)
    assert got[0] == [0]
    assert got[1] == []


def test_corner_tangent():
    # 球心在块外，但球恰好与块角点相切。
    lights = [((1.2, 1.2), math.hypot(0.2, 0.2))]
    got = LightCull(1.0, 4).cull(lights, 2, 2)
    assert 0 in got[0]


# ---------------------------------------------------------------- 限项
def test_limit_keeps_nearest_by_distance():
    lc = LightCull(10.0, 2)  # 单块 10x10，块中心 (5,5)
    lights = [((0.0, 0.0), 6.0), ((5.0, 5.0), 10.0), ((9.5, 9.5), 10.0),
              ((1.0, 1.0), 10.0)]
    got = lc.cull(lights, 10, 10)
    assert len(got) == 1
    # 到中心距离：下标1=0 < 下标3=√32 ≈ 5.66 < 下标2=√40.5 ≈ 6.36。
    assert got[0] == [1, 3]


def test_limit_equidistant_uses_index():
    lc = LightCull(10.0, 1)  # 块中心 (5,5)
    lights = [((9.0, 9.0), 10.0), ((1.0, 1.0), 10.0)]  # 等距 √32
    assert lc.cull(lights, 10, 10)[0] == [0]


def test_limit_tie_breaks_by_index():
    lc = LightCull(10.0, 2)
    # (4,5) 与 (6,5) 到中心 (5,5) 等距，按小下标优先。
    lights = [((4.0, 5.0), 1.0), ((6.0, 5.0), 1.0), ((5.0, 8.0), 1.0)]
    got = lc.cull(lights, 10, 10)
    assert got[0] == [0, 1]


def test_limit_never_exceeded():
    lights = [((5.0, 5.0), 100.0)] * 50
    got = LightCull(1.0, 3).cull(lights, 4, 4)
    assert all(len(t) <= 3 for t in got)


# ---------------------------------------------------------------- 非法输入
def test_bad_tile_history():
    for bad in [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True]:
        with pytest.raises(InvalidTileSizeError):
            LightCull(bad, 4)


def test_bad_max_lights():
    for bad in [0, -3, 2.5, True, None]:
        with pytest.raises(InvalidMaxLightsError):
            LightCull(1.0, bad)


def test_bad_area_size():
    lc = LightCull(1.0, 4)
    for bad in [0, -1.0, float("nan"), float("inf"), True, None]:
        with pytest.raises(InvalidAreaSizeError):
            lc.cull([], bad, 4.0)
        with pytest.raises(InvalidAreaSizeError):
            lc.cull([], 4.0, bad)


def test_bad_light_shape():
    lc = LightCull(1.0, 4)
    for bad in [[((0.0,), 1.0)], [((0.0, 0.0, 0.0), 1.0)], [(0.0, 0.0)],
                [((0.0, 0.0), -1.0)], [((0.0, 0.0), "x")], [("a", "b")], [42]]:
        with pytest.raises(InvalidLightError):
            lc.cull(bad, 2, 2)


def test_non_finite_light_classified():
    lc = LightCull(1.0, 4)
    for bad in [[((float("nan"), 0.0), 1.0)],
                [((0.0, float("inf")), 1.0)],
                [((0.0, 0.0), float("nan"))]]:
        with pytest.raises(NonFiniteLightError):
            lc.cull(bad, 2, 2)


def test_error_hierarchy_is_value_error():
    for exc in (InvalidTileSizeError, InvalidMaxLightsError,
                InvalidAreaSizeError, InvalidLightError, NonFiniteLightError):
        assert issubclass(exc, LightCullError)
        assert issubclass(exc, ValueError)
    # 分类互不混淆。
    assert not issubclass(NonFiniteLightError, InvalidTileSizeError)
    assert not issubclass(InvalidAreaSizeError, InvalidLightError)


# ---------------------------------------------------------------- 结果结构
def test_types_and_indices():
    result = LightCull(1.0, 4).cull(L, 4, 4)
    assert isinstance(result, list)
    for tile in result:
        assert isinstance(tile, list)
        for index in tile:
            assert isinstance(index, int)


def test_deterministic():
    lc = LightCull(1.0, 4)
    assert lc.cull(L, 4, 4) == lc.cull(L, 4, 4)


def test_does_not_mutate_input():
    lights = [((0.5, 0.5), 1.0)]
    snapshot = list(lights)
    LightCull(1.0, 4).cull(lights, 4, 4)
    assert lights == snapshot


def test_input_not_retained_between_calls():
    lc = LightCull(1.0, 4)
    assert lc.cull([((0.5, 0.5), 1.0)], 2, 2) != lc.cull([], 2, 2)


# ---------------------------------------------------------------- 性能
def test_perf_ten_thousand_tiles_under_500ms():
    tile = 1.0
    cols = rows = 100  # 100 × 100 = 10000 块
    width = height = float(cols)
    rng = random.Random(1234)
    lights = [((rng.uniform(0, width), rng.uniform(0, height)),
               rng.uniform(0.0, 3.0)) for _ in range(400)]
    lc = LightCull(tile, 8)
    start = time.perf_counter()
    result = lc.cull(lights, width, height)
    elapsed = time.perf_counter() - start
    assert len(result) == 10_000
    assert elapsed < 0.5, "cull took %.3f s for 10k tiles" % elapsed


def test_perf_small_lights_sparse():
    # 大量小半径光源：验证不会退化为"块数 × 光源数"。
    tile = 1.0
    cols = rows = 100
    width = height = float(cols)
    rng = random.Random(4321)
    lights = [((rng.uniform(0, width), rng.uniform(0, height)), 0.2)
              for _ in range(5000)]
    lc = LightCull(tile, 8)
    start = time.perf_counter()
    result = lc.cull(lights, width, height)
    elapsed = time.perf_counter() - start
    assert len(result) == 10_000
    assert elapsed < 0.5, "cull took %.3f s for 5k small lights / 10k tiles" % elapsed


# ---------------------------------------------------------------- 并发
def test_concurrent_same_instance():
    lc = LightCull(1.0, 4)
    rng = random.Random(7)
    lights = [((rng.uniform(0, 8), rng.uniform(0, 8)), rng.uniform(0, 2))
              for _ in range(60)]
    expected = lc.cull(lights, 8, 8)

    def work(_):
        return lc.cull(lights, 8, 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(64)))
    assert all(r == expected for r in results)


def test_concurrent_distinct_instances():
    def work(seed):
        rng = random.Random(seed)
        lights = [((rng.uniform(0, 4), rng.uniform(0, 4)), rng.uniform(0, 2))
                  for _ in range(30)]
        return LightCull(0.5, 4).cull(lights, 4, 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(32)))
    assert len(results) == 32
    assert all(len(r) == 64 for r in results)


def test_concurrent_matches_serial():
    rng = random.Random(99)
    batches = []
    for _ in range(16):
        batches.append([((rng.uniform(0, 6), rng.uniform(0, 6)),
                         rng.uniform(0, 2)) for _ in range(25)])
    lc = LightCull(0.75, 5)
    serial = [lc.cull(b, 6, 6) for b in batches]

    def work(batch):
        return lc.cull(batch, 6, 6)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(work, batches))
    assert parallel == serial


# ---------------------------------------------------------------- 题面主路径
def test_visible_story_with_correct_semantics():
    # 与自带用例同数据，但按正确块数核对。
    result = LightCull(1.0, 4).cull(L, 4, 4)
    assert len(result) == 16
    assert 0 in result[0] and 0 not in result[-1]
    assert 1 in result[-1]
