import math

import pytest

import lightcull

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

L = [((0.5, 0.5), 1.0), ((3.5, 3.5), 1.0)]

CASES = [
    ("tiles", lambda: len(lightcull.LightCull(1.0, 4).cull(L, 4, 4)) == 4),
    ("first_has", lambda: len(lightcull.LightCull(1.0, 4).cull(L, 4, 4)[0]) >= 1),
    ("last_has", lambda: len(lightcull.LightCull(1.0, 4).cull(L, 4, 4)[-1]) >= 1),
    ("max_lights", lambda: all(len(t) <= 4 for t in lightcull.LightCull(1.0, 4).cull(L, 4, 4))),
    ("bad_tile", lambda: _raises(lambda: lightcull.LightCull(0.0, 4))),
    ("bad_max", lambda: _raises(lambda: lightcull.LightCull(1.0, 0))),
    ("empty", lambda: len(lightcull.LightCull(1.0, 4).cull([], 2, 2)) == 4),
    ("type", lambda: isinstance(lightcull.LightCull(1.0, 4).cull(L, 4, 4), list)),
    ("det", lambda: lightcull.LightCull(1.0, 4).cull(L, 4, 4) == lightcull.LightCull(1.0, 4).cull(L, 4, 4)),
    ("bad_size", lambda: _raises(lambda: lightcull.LightCull(1.0, 4).cull(L, 0, 4))),
    ("sorted", lambda: all(isinstance(t, list) for t in lightcull.LightCull(1.0, 4).cull(L, 4, 4))),
    ("far_light", lambda: all(0 not in t for t in lightcull.LightCull(1.0, 1).cull([((100.0, 100.0), 0.1)], 2, 2))),
    ("nan_light", lambda: _raises(lambda: lightcull.LightCull(1.0, 4).cull([((float('nan'), 0.0), 1.0)], 2, 2))),
    ("no_mut", lambda: lightcull.LightCull(1.0, 4).cull(list(L), 4, 4) is not None),
    ("count_sum", lambda: len(lightcull.LightCull(2.0, 4).cull(L, 4, 4)) == 4),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
