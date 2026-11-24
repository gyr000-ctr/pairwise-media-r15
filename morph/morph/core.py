"""morph 核心 API（初始未实现）。"""


class Morph:
    def erode(self, image, size=3, edge='clamp'):
        """腐蚀"""
        raise NotImplementedError

    def dilate(self, image, size=3, edge='clamp'):
        """膨胀"""
        raise NotImplementedError
