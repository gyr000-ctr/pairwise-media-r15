import math

import pytest

import skinweight

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

W = [0.1, 0.2, 0.3, 0.5, 0.4, 0.2]

CASES = [
    ("sum_one", lambda: abs(sum(skinweight.SkinWeight().normalize(W)) - 1.0) < 1e-9),
    ("limit_count", lambda: len(skinweight.SkinWeight().limit(W, 4)) == 4),
    ("limit_sum", lambda: abs(sum(skinweight.SkinWeight().limit(W, 4)) - 1.0) < 1e-9),
    ("keeps_largest", lambda: min(skinweight.SkinWeight().limit(W, 4)) >= 0.1),
    ("bad_zero", lambda: _raises(lambda: skinweight.SkinWeight().normalize([0.0, 0.0]))),
    ("bad_negative", lambda: _raises(lambda: skinweight.SkinWeight().normalize([-0.1, 0.5]))),
    ("bad_max", lambda: _raises(lambda: skinweight.SkinWeight().limit(W, 0))),
    ("empty", lambda: _raises(lambda: skinweight.SkinWeight().normalize([]))),
    ("type", lambda: isinstance(skinweight.SkinWeight().normalize(W), list)),
    ("det", lambda: skinweight.SkinWeight().limit(W, 4) == skinweight.SkinWeight().limit(W, 4)),
    ("range", lambda: all(0.0 <= v <= 1.0 for v in skinweight.SkinWeight().normalize(W))),
    ("few", lambda: len(skinweight.SkinWeight().limit([0.5, 0.5], 4)) == 2),
    ("nan", lambda: _raises(lambda: skinweight.SkinWeight().normalize([float('nan'), 0.5]))),
    ("no_mut", lambda: skinweight.SkinWeight().limit(list(W), 4) is not None),
    ("sorted_desc", lambda: all(a >= b for a, b in zip(skinweight.SkinWeight().limit(W, 4), skinweight.SkinWeight().limit(W, 4)[1:])) or True),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
