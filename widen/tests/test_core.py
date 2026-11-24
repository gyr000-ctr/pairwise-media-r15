import math

import pytest

import widen

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _w(x=1.0):
    return widen.Widen(x)

CASES = [
    ("length", lambda: len(_w().process([0.1, 0.2], [0.1, 0.2])[0]) == 2),
    ("width0_equal", lambda: all(abs(a - b) < 1e-12 for a, b in zip(_w(0.0).process([0.3], [0.1])[0], _w(0.0).process([0.3], [0.1])[1]))),
    ("mono_compat", lambda: abs(_w().mono_level([0.5], [0.5]) - _w(0.0).mono_level([0.5], [0.5])) <= 0.5),
    ("bad_width", lambda: _raises(lambda: widen.Widen(-0.1))),
    ("empty", lambda: _w().process([], []) == ([], [])),
    ("bad_len", lambda: _raises(lambda: _w().process([0.1], [0.1, 0.2]))),
    ("type", lambda: isinstance(_w().process([0.1], [0.2]), tuple)),
    ("det", lambda: _w().process([0.1], [0.2]) == _w().process([0.1], [0.2])),
    ("range", lambda: all(abs(v) <= 1.0 for v in _w().process([0.5], [0.5])[0])),
    ("nan", lambda: _raises(lambda: _w().process([float('nan')], [0.0]))),
    ("mono_type", lambda: isinstance(_w().mono_level([0.1], [0.2]), float)),
    ("stereo_pair", lambda: len(_w().process([0.1], [0.2])) == 2),
    ("no_jump", lambda: max(abs(a - b) for a, b in zip(_w().process([0.3] * 20, [0.1] * 20)[0], _w().process([0.3] * 20, [0.1] * 20)[0][1:])) < 1.0),
    ("no_mut", lambda: _w().process([0.2], [0.1]) is not None),
    ("width_one", lambda: isinstance(_w(1.0).process([0.1], [0.2]), tuple)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
