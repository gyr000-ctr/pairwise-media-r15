"""spriteanim 扩展约束的独立回归用例。

覆盖：循环边界不重复、累计时长、九宫格逐像素坐标对照（1e-12）、
帧事件恰好一次、非有限输入分类报错、并发安全与性能上界。
"""

from __future__ import annotations

import math
import threading
import time

import pytest

import spriteanim
from spriteanim import (
    Animator,
    FramePlayer,
    InvalidBorderError,
    InvalidDestRectError,
    InvalidFpsError,
    InvalidFrameCountError,
    NegativeTimeError,
    NonFiniteError,
    RegionTooSmallError,
    SpriteAnim,
    events_between,
)

SA = SpriteAnim()
SRC = (0, 0, 32, 32)
DST = (0, 0, 96, 96)


# --------------------------------------------------------------------------- #
# 序列帧
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fps", [1, 2, 3, 5, 10, 12, 24, 29.97, 30, 60, 144])
@pytest.mark.parametrize("count", [1, 2, 4, 7])
def test_exact_boundaries_never_repeat_last_frame(fps, count):
    for i in range(300):
        # t 恰好是第 i 条帧边界：必须进入第 i 帧，末帧不在循环点重复。
        assert SA.frame_at(i / fps, fps, count) == i % count


def test_period_point_returns_frame_zero():
    for count in range(1, 9):
        period = count / 10.0
        assert SA.frame_at(period, 10.0, count) == 0
        assert SA.frame_at(2 * period, 10.0, count) == 0


def test_frame_type_is_exact_int():
    assert type(SA.frame_at(0.0, 10.0, 4)) is int


def test_large_timestamp_keeps_frame_in_range():
    assert SA.frame_at(1e12, 10.0, 4) == 0
    for t in (1e9, 1e12, 1e15):
        f = SA.frame_at(t + 0.123456, 29.97, 7)
        assert type(f) is int and 0 <= f < 7


def test_non_loop_stops_on_last_frame():
    assert SA.frame_at(100.0, 10.0, 4, loop=False) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(t=float("nan"), fps=10.0, count=4),
        dict(t=float("inf"), fps=10.0, count=4),
        dict(t=0.0, fps=float("nan"), count=4),
        dict(t=0.0, fps=float("inf"), count=4),
    ],
)
def test_non_finite_inputs_are_classified(kwargs):
    with pytest.raises(NonFiniteError):
        SA.frame_at(**kwargs)
    with pytest.raises(ValueError):
        SA.frame_at(**kwargs)


def test_non_finite_fps_is_also_fps_error():
    with pytest.raises(InvalidFpsError):
        SA.frame_at(0.0, float("nan"), 4)


def test_invalid_frame_arguments_are_distinguishable():
    with pytest.raises(InvalidFpsError):
        SA.frame_at(0.0, 0.0, 4)
    with pytest.raises(InvalidFpsError):
        SA.frame_at(0.0, -3.0, 4)
    with pytest.raises(InvalidFrameCountError):
        SA.frame_at(0.0, 10.0, 0)
    with pytest.raises(InvalidFrameCountError):
        SA.frame_at(0.0, 10.0, -1)
    with pytest.raises(NegativeTimeError):
        SA.frame_at(-1e-9, 10.0, 4)


def test_100k_frame_lookups_within_budget():
    n = 100_000
    start = time.perf_counter()
    total = 0
    for i in range(n):
        total += SA.frame_at(i * 0.000123, 10.0, 4)
    elapsed = time.perf_counter() - start
    assert 0 <= total
    # 题面上界 200ms；这里留足机器负载余量做回归保护。
    assert elapsed < 0.5, elapsed


# --------------------------------------------------------------------------- #
# 累计时长
# --------------------------------------------------------------------------- #
def test_cumulative_durations():
    durations = [0.1, 0.2, 0.3]
    expected = [
        (0.0, 0), (0.099, 0),
        (0.1, 1), (0.299, 1),
        (0.3, 2), (0.599, 2),
        (0.6, 0), (0.7, 1), (1.2, 0),
    ]
    for t, frame in expected:
        assert SA.frame_at(t, durations=durations) == frame, t


def test_cumulative_durations_via_constructor():
    anim = SpriteAnim(durations=[1, 1, 2])
    assert anim.frame_at(0.9) == 0
    assert anim.frame_at(1.0) == 1
    assert anim.frame_at(2.0) == 2
    assert anim.frame_at(4.0) == 0


# --------------------------------------------------------------------------- #
# 九宫格
# --------------------------------------------------------------------------- #
def _reference_nine_slice(src, dst, border):
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    if isinstance(border, (int, float)):
        top = right = bottom = left = border
    else:
        top, right, bottom, left = border
    xs = [dx, dx + left, dx + dw - right, dx + dw]
    ys = [dy, dy + top, dy + dh - bottom, dy + dh]
    out = []
    for row in range(3):
        for col in range(3):
            out.append(
                (xs[col], ys[row], xs[col + 1] - xs[col], ys[row + 1] - ys[row])
            )
    return out


def test_nine_slice_pixel_coordinates_within_1e_12():
    cases = [
        ((0, 0, 32, 32), (0, 0, 96, 96), 8),
        ((3, 7, 40, 50), (11, -5, 123.5, 200.25), 9.25),
        ((0, 0, 10, 10), (0, 0, 10, 10), 5),
        ((-20, 40, 64, 48), (5, 6, 77, 333), (4, 6, 8, 10)),
    ]
    for src, dst, border in cases:
        got = SA.nine_slice(src, dst, border)
        exp = _reference_nine_slice(src, dst, border)
        for gr, er in zip(got, exp):
            for gv, ev in zip(gr, er):
                assert abs(gv - ev) <= 1e-12


def test_nine_slice_stretch_roles():
    lay = SA.nine_slice(SRC, DST, 8)
    # 四角不缩放。
    for i in (0, 2, 6, 8):
        assert lay[i][2] == 8 and lay[i][3] == 8
    # 上下边只在横向拉伸。
    assert (lay[1][2], lay[1][3]) == (80, 8)
    assert (lay[7][2], lay[7][3]) == (80, 8)
    # 左右边只在纵向拉伸。
    assert (lay[3][2], lay[3][3]) == (8, 80)
    assert (lay[5][2], lay[5][3]) == (8, 80)
    # 中心双向拉伸。
    assert lay[4] == (8, 8, 80, 80)


def test_nine_slice_covers_destination_without_gap_or_overlap():
    lay = SA.nine_slice(SRC, DST, 8)
    assert sum(r[2] * r[3] for r in lay) == 96 * 96
    assert lay[0][0] == 0 and lay[0][1] == 0
    assert lay[8][0] + lay[8][2] == 96
    assert lay[8][1] + lay[8][3] == 96


def test_nine_slice_error_types_are_distinguishable():
    with pytest.raises(InvalidBorderError):
        SA.nine_slice(SRC, DST, 0)
    with pytest.raises(InvalidBorderError):
        SA.nine_slice(SRC, DST, -1)
    with pytest.raises(RegionTooSmallError):
        SA.nine_slice((0, 0, 4, 4), DST, 8)
    with pytest.raises(RegionTooSmallError):
        SA.nine_slice(SRC, (0, 0, 10, 96), 8)
    with pytest.raises(InvalidDestRectError):
        SA.nine_slice(SRC, (0, 0, 0, 0), 8)
    with pytest.raises(NonFiniteError):
        SA.nine_slice(SRC, DST, float("nan"))


# --------------------------------------------------------------------------- #
# 帧事件：跨帧恰好一次
# --------------------------------------------------------------------------- #
def test_frame_events_fire_exactly_once_per_crossing():
    calls = []
    player = FramePlayer(
        4,
        10.0,
        events={
            0: lambda: calls.append(0),
            1: lambda: calls.append(1),
            2: lambda: calls.append(2),
            3: lambda: calls.append(3),
        },
    )
    for _ in range(4):
        player.update(0.1)
    assert calls == [1, 2, 3, 0]
    for _ in range(4):
        player.update(0.1)
    assert calls == [1, 2, 3, 0, 1, 2, 3, 0]
    player.update(0.0)
    assert len(calls) == 8


def test_big_jump_across_loop_fires_each_event_once():
    calls = []
    player = FramePlayer(
        4, 10.0, events={i: (lambda i=i: calls.append(i)) for i in range(4)}
    )
    fired = player.update(0.4)
    assert calls == [1, 2, 3, 0]
    assert len(fired) == 4


def test_events_between_boundary_semantics():
    events = {0: "e0", 1: "e1", 2: "e2", 3: "e3"}
    assert [e.name for e in events_between(0.0, 0.1, 10.0, 4, events)] == ["e1"]
    assert [e.name for e in events_between(0.1, 0.2, 10.0, 4, events)] == ["e2"]
    assert [
        e.name for e in events_between(0.0, 0.4, 10.0, 4, events)
    ] == ["e1", "e2", "e3", "e0"]


def test_player_aliases_exist():
    assert Animator is FramePlayer
    assert spriteanim.SpriteAnimator is FramePlayer


# --------------------------------------------------------------------------- #
# 并发安全
# --------------------------------------------------------------------------- #
def test_concurrent_frame_and_slice_calls():
    errors = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait()
            anim = SpriteAnim()
            for i in range(1000):
                frame = anim.frame_at(i * 0.013, 29.97, 9)
                assert 0 <= frame < 9
                rects = anim.nine_slice((1, 2, 40, 40), (3, 4, 120.0, 120.0), 7)
                assert len(rects) == 9 and rects[0][2] == 7
        except Exception as exc:  # pragma: no cover - 失败时上报
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
