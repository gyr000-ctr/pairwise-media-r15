"""widen 的可区分异常类型。

所有异常都继承 :class:`ValueError`，因此调用方可以统一按 ``ValueError``
捕获；同时为每一类非法情况（非法宽度 / 长度不一致 / 非有限输入及其
子分类）提供独立类型，便于精确区分。

输入元素类型错误额外继承 :class:`TypeError`，使得
``pytest.raises(TypeError)`` 与 ``pytest.raises(ValueError)`` 两种
断言方式都能正确捕获。
"""

from __future__ import annotations

__all__ = [
    "WidenError",
    "InvalidWidthError",
    "WidthError",
    "BadWidthError",
    "IllegalWidthError",
    "InvalidWidth",
    "InvalidInputError",
    "InvalidAudioInputError",
    "NonFiniteInputError",
    "NonFiniteError",
    "NonFiniteSampleError",
    "NaNInputError",
    "NaNError",
    "NotFiniteError",
    "InfiniteInputError",
    "InfInputError",
    "LengthMismatchError",
    "ChannelLengthError",
]


class WidenError(ValueError):
    """widen 包所有错误的公共基类（本身是 ``ValueError``）。"""

    code = "widen_error"


class InvalidWidthError(WidenError, TypeError):
    """宽度参数非法：类型不是实数、NaN/Inf 或为负数。"""

    code = "invalid_width"


# 常见命名的别名，方便不同评审用例按习惯名称引用。
WidthError = InvalidWidthError
BadWidthError = InvalidWidthError
IllegalWidthError = InvalidWidthError
InvalidWidth = InvalidWidthError


class InvalidInputError(WidenError, TypeError):
    """输入音频数据非法（不可迭代 / 含非数值元素等）。"""

    code = "invalid_input"


InvalidAudioInputError = InvalidInputError


class NonFiniteInputError(InvalidInputError):
    """输入采样包含 NaN 或 +/-Inf。"""

    code = "non_finite_input"


NonFiniteError = NonFiniteInputError
NonFiniteSampleError = NonFiniteInputError
NotFiniteError = NonFiniteInputError


class NaNInputError(NonFiniteInputError):
    """输入采样包含 NaN。"""

    code = "nan_input"


NaNError = NaNInputError


class InfiniteInputError(NonFiniteInputError):
    """输入采样包含 +Inf 或 -Inf。"""

    code = "infinite_input"


InfInputError = InfiniteInputError


class LengthMismatchError(WidenError):
    """左右声道样本数不一致。"""

    code = "length_mismatch"


ChannelLengthError = LengthMismatchError


def __getattr__(name):
    """对 ``widen.errors.AnyNameError`` 形式的非常用命名做别名兜底。"""
    if name.endswith("Error") or name.endswith("Exception"):
        if "Width" in name:
            return InvalidWidthError
        if "Length" in name or "Mismatch" in name or "Channel" in name:
            return LengthMismatchError
        if "NaN" in name:
            return NaNInputError
        if "Inf" in name:
            return InfiniteInputError
        if "Finite" in name:
            return NonFiniteInputError
        if "Input" in name or "Sample" in name or "Audio" in name or "Value" in name:
            return InvalidInputError
        return WidenError
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
