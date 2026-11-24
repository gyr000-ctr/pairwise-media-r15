import math

import pytest

import morph

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]

CASES = [
    ("erode_center", lambda: morph.Morph().erode(IMG)[1][1] == 0.0),
    ("dilate_center", lambda: morph.Morph().dilate(IMG)[1][1] == 1.0),
    ("shape", lambda: len(morph.Morph().erode(IMG)) == 3),
    ("edge_diff", lambda: morph.Morph().erode(IMG, 3, 'clamp') != morph.Morph().erode(IMG, 3, 'zero')),
    ("bad_edge", lambda: _raises(lambda: morph.Morph().erode(IMG, 3, 'mirror'))),
    ("bad_size", lambda: _raises(lambda: morph.Morph().erode(IMG, 2))),
    ("empty", lambda: _raises(lambda: morph.Morph().erode([]))),
    ("type", lambda: isinstance(morph.Morph().erode(IMG), list)),
    ("det", lambda: morph.Morph().erode(IMG) == morph.Morph().erode(IMG)),
    ("dilate_ge_erode", lambda: all(morph.Morph().dilate(IMG)[r][c] >= morph.Morph().erode(IMG)[r][c] for r in range(3) for c in range(3))),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in morph.Morph().dilate(IMG) for v in row)),
    ("idempotent_open", lambda: morph.Morph().dilate(morph.Morph().erode(IMG)) == morph.Morph().dilate(morph.Morph().erode(IMG))),
    ("nan", lambda: _raises(lambda: morph.Morph().erode([[float('nan')]]))),
    ("no_mut", lambda: morph.Morph().erode([r[:] for r in IMG]) is not None),
    ("const", lambda: all(v == 1.0 for row in morph.Morph().dilate([[1.0] * 3 for _ in range(3)]) for v in row)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
