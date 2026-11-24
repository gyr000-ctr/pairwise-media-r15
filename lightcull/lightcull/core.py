"""lightcull 核心 API（初始未实现）。"""


class LightCull:
    def __init__(self, tile, max_lights):
        """构造"""
        raise NotImplementedError

    def cull(self, lights, width, height):
        """分块结果"""
        raise NotImplementedError
