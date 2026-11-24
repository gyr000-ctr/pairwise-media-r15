"""蒙皮权重与对偶四元数蒙皮的异常类型。

所有异常都继承 :class:`ValueError`，因此调用方既可以按具体语义
（零权重 / 负权重 / 非有限输入）分别捕获，也可以统一按 ``ValueError`` 捕获。
"""


class SkinWeightError(ValueError):
    """权重相关错误的公共基类。"""


class ZeroWeightError(SkinWeightError):
    """顶点权重和为零（整顶点全零权重，或空输入）。"""


class NegativeWeightError(SkinWeightError):
    """权重中出现负值。"""


class NonFiniteWeightError(SkinWeightError):
    """权重中出现 NaN / +Inf / -Inf。"""


class InfluenceLimitError(SkinWeightError):
    """``max_influence`` 参数非法（非正整数等）。"""


class TransformError(ValueError):
    """骨骼变换输入非法。"""
