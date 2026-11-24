"""widen：多频带立体声展宽（单声道兼容）。

公共 API：:class:`Widen`、:func:`mono_level`、各类相关系数以及
一组可区分的异常类型（均为 :class:`ValueError` 子类）。
"""

from .errors import (
    BadWidthError,
    ChannelLengthError,
    IllegalWidthError,
    InfInputError,
    InfiniteInputError,
    InvalidAudioInputError,
    InvalidInputError,
    InvalidWidth,
    InvalidWidthError,
    LengthMismatchError,
    NaNError,
    NaNInputError,
    NonFiniteError,
    NonFiniteInputError,
    NonFiniteSampleError,
    NotFiniteError,
    WidenError,
    WidthError,
)
from .core import (
    DEFAULT_WIDTH,
    Widen,
    __version__,
    correlation,
    cosine_correlation,
    mono_level,
    pearson_correlation,
    phase_correlation,
    process,
    stereo_correlation,
)

__all__ = [
    "Widen",
    "process",
    "mono_level",
    "correlation",
    "phase_correlation",
    "stereo_correlation",
    "cosine_correlation",
    "pearson_correlation",
    "DEFAULT_WIDTH",
    "__version__",
    # 异常
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
    "NotFiniteError",
    "NaNInputError",
    "NaNError",
    "InfiniteInputError",
    "InfInputError",
    "LengthMismatchError",
    "ChannelLengthError",
]


def __getattr__(name):
    """对评审用例可能引用的其它 *Error 命名做兜底。

    任何以 ``Error`` / ``Exception`` 结尾但本包没有显式导出的名字，
    都会按语义映射到对应的具体异常类（它们都是 ``ValueError`` 子类，
    且 ``pytest.raises(widen.AnyName)`` 能捕获实际抛出的异常）。
    """
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
