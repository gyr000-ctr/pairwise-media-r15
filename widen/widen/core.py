"""widen 核心 API（初始未实现）。"""


class Widen:
    def __init__(self, width):
        """构造"""
        raise NotImplementedError

    def process(self, left, right):
        """处理"""
        raise NotImplementedError

    def mono_level(self, left, right):
        """下混电平"""
        raise NotImplementedError
