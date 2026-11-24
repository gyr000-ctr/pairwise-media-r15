"""skinweight：权重限项归一与对偶四元数蒙皮（DQS）。

公共 API
--------
- 权重：:class:`SkinWeight`
- 对偶四元数：:class:`DualQuaternion`
- 蒙皮：:class:`Skinner` 及 :func:`dual_quaternion_skinning`、
  :func:`linear_blend_skinning`
- 异常：:class:`ZeroWeightError`、:class:`NegativeWeightError`、
  :class:`NonFiniteWeightError`、:class:`InfluenceLimitError`
"""

from .core import *  # noqa: F401,F403
from .dualquaternion import DualQuaternion, quat_conj, quat_mul, quat_rotate
from .errors import (
    InfluenceLimitError,
    NegativeWeightError,
    NonFiniteWeightError,
    SkinWeightError,
    TransformError,
    ZeroWeightError,
)
from .skinning import (
    Skinner,
    dual_quaternion_skinning,
    linear_blend_skinning,
)
from .weights import SkinWeight

__all__ = [
    "SkinWeight",
    "Skinner",
    "DualQuaternion",
    "quat_mul",
    "quat_conj",
    "quat_rotate",
    "dual_quaternion_skinning",
    "linear_blend_skinning",
    "SkinWeightError",
    "ZeroWeightError",
    "NegativeWeightError",
    "NonFiniteWeightError",
    "InfluenceLimitError",
    "TransformError",
]
