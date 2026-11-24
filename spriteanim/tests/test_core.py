import math

import pytest

import spriteanim

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

SRC = (0, 0, 32, 32)
DST = (0, 0, 96, 96)

CASES = [
    ("frame_zero", lambda: spriteanim.SpriteAnim().frame_at(0.0, 10.0, 4) == 0),
    ("frame_one", lambda: spriteanim.SpriteAnim().frame_at(0.1, 10.0, 4) == 1),
    ("frame_wrap", lambda: spriteanim.SpriteAnim().frame_at(0.4, 10.0, 4) == 0),
    ("bad_fps", lambda: _raises(lambda: spriteanim.SpriteAnim().frame_at(0.0, 0.0, 4))),
    ("bad_count", lambda: _raises(lambda: spriteanim.SpriteAnim().frame_at(0.0, 10.0, 0))),
    ("nine_len", lambda: len(spriteanim.SpriteAnim().nine_slice(SRC, DST, 8)) == 9),
    ("bad_border", lambda: _raises(lambda: spriteanim.SpriteAnim().nine_slice(SRC, DST, 0))),
    ("bad_dst", lambda: _raises(lambda: spriteanim.SpriteAnim().nine_slice(SRC, (0, 0, 0, 0), 8))),
    ("type", lambda: isinstance(spriteanim.SpriteAnim().nine_slice(SRC, DST, 8), list)),
    ("det", lambda: spriteanim.SpriteAnim().nine_slice(SRC, DST, 8) == spriteanim.SpriteAnim().nine_slice(SRC, DST, 8)),
    ("corners_fixed", lambda: spriteanim.SpriteAnim().nine_slice(SRC, DST, 8)[0][2] == 8),
    ("frame_type", lambda: isinstance(spriteanim.SpriteAnim().frame_at(0.0, 10.0, 4), int)),
    ("negative_t", lambda: _raises(lambda: spriteanim.SpriteAnim().frame_at(-0.1, 10.0, 4))),
    ("no_mut", lambda: spriteanim.SpriteAnim().frame_at(0.1, 10.0, 4) is not None),
    ("bounds", lambda: 0 <= spriteanim.SpriteAnim().frame_at(0.99, 10.0, 4) <= 3),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
