"""spriteanim 核心 API（初始未实现）。"""


class SpriteAnim:
    def frame_at(self, t, fps, count):
        """取帧"""
        raise NotImplementedError

    def nine_slice(self, src, dst, border):
        """九宫格布局"""
        raise NotImplementedError
