"""skinweight 核心 API。

历史的 :class:`SkinWeight` 权重限项/归一接口保留在
:mod:`skinweight.weights` 中并在此重新导出；对偶四元数蒙皮的主实现见
:mod:`skinweight.skinning`。
"""

from .weights import SkinWeight

__all__ = ["SkinWeight"]
