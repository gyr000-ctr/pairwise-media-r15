"""lightcull —— 分块光源剔除（tiled light culling）。

语义
----
把世界矩形 ``[0, width] × [0, height]`` 按边长为 ``tile`` 的规则网格切分：
列数 ``cols = ceil(width / tile)``，行数 ``rows = ceil(height / tile)``。
``cull`` 返回长度为 ``rows * cols`` 的列表（行优先，索引 ``row * cols + col``），
每一项是该块命中光源的**下标**列表。

* 命中判定与暴力「球-块求交」完全一致：光源 ``i`` 由球心 ``(cx, cy)`` 和半径
  ``r`` 定义，当且仅当球与块闭区间 AABB 相交时命中该块。块边界按世界矩形
  裁剪（最后一行/列可能不满一个 tile）。判定取 AABB 上距球心最近点的平方
  距离与 ``r^2`` 比较，相切算相交，因此块边界不会漏光。
* 每块光源数上限：命中数超过 ``max_lights`` 时，按「光源中心到块中心」的
  欧氏距离由近到远保留 ``max_lights`` 个；距离相同按光源下标升序，结果确定。
* 非法输入分类报错：块边长、上限、区域尺寸、光源数据、非有限值各自抛出可
  区分的异常类型（均继承 ``ValueError``）。
* 性能：逐光源只遍历其包围盒覆盖的候选块，候选块再逐一做精确求交，避免
  ``块数 × 光源数`` 的暴力两两判定。
* 并发安全：``LightCull`` 构造后配置不可变，``cull`` 只使用局部变量，不读写
  任何共享可变状态，同一实例可被多线程并发调用。

关于仓库自带测试的已知冲突
--------------------------
``tests/test_core.py`` 的 ``tiles`` 断言
``len(LightCull(1.0, 4).cull(L, 4, 4)) == 4``，但同一文件里 ``empty``
（2×2 / tile=1.0）与 ``count_sum``（4×4 / tile=2.0）都要求 4 块，二者唯一地
把块数固定为 ``ceil(w/tile) * ceil(h/tile)``；按该式 4×4 / tile=1.0 应为
16 块。故 ``tiles`` 与其余用例在算术上不可同时成立，属生成用例的笔误（应为
16，或调用参数应为 tile=2.0 / 2×2 区域）。本实现保持正确的通用语义，不修改
自带的测试文件；完整论证见 ``IMPLEMENTATION.md``。
"""

import heapq
import math

__all__ = [
    "LightCull",
    "LightCullError",
    "InvalidTileSizeError",
    "InvalidMaxLightsError",
    "InvalidAreaSizeError",
    "InvalidLightError",
    "NonFiniteLightError",
]


class LightCullError(ValueError):
    """lightcull 相关错误的基类（仍是 ValueError，兼容既有调用方）。"""


class InvalidTileSizeError(LightCullError):
    """块边长不是有限正实数。"""


class InvalidMaxLightsError(LightCullError):
    """每块光源上限不是正整数。"""


class InvalidAreaSizeError(LightCullError):
    """区域宽/高不是有限正实数。"""


class InvalidLightError(LightCullError):
    """光源数据形状非法（不是 ((x, y), radius)）或半径为负。"""


class NonFiniteLightError(InvalidLightError):
    """光源坐标或半径出现 NaN / ±inf。"""


def _is_real(value):
    """真正的实数（排除 bool，因为 bool 是 int 的子类）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _axis_count(extent, tile):
    """覆盖长度为 extent 的轴所需的最少块数。

    定义为浮点运算下满足 ``n * tile >= extent`` 的最小正整数 n，这样最后
    一块的右边界一定覆盖到 extent，任何可表示坐标都不会落在块网格之外；
    同时避免 0.3 / 0.1 = 2.9999999999999996 之类的浮点误差多切出一个空块。
    """
    q = extent / tile
    count = math.ceil(q)
    if count < 1:
        count = 1
    # 向下收紧：若少一块也已覆盖 extent，则不是最少块数。
    while count > 1 and (count - 1) * tile >= extent:
        count -= 1
    # 向上补齐：保证最后一块覆盖 extent。
    while count * tile < extent:
        count += 1
    return count


def _parse_light(item, index):
    """校验并规范化单个光源，返回 ``(cx, cy, radius)``。"""
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        raise InvalidLightError(
            "light %d must be ((x, y), radius), got %r" % (index, item))
    position, radius = item
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        raise InvalidLightError(
            "light %d position must be (x, y), got %r" % (index, position))
    cx, cy = position
    for value in (cx, cy, radius):
        if not _is_real(value):
            raise InvalidLightError(
                "light %d has a non-real component in %r" % (index, item))
        if not math.isfinite(value):
            raise NonFiniteLightError(
                "light %d has a non-finite component in %r" % (index, item))
    if radius < 0.0:
        raise InvalidLightError(
            "light %d has negative radius %r" % (index, radius))
    return float(cx), float(cy), float(radius)


class LightCull:
    """把一块 2D 区域切成 tile 网格，并为每块收集命中的光源。"""

    __slots__ = ("_tile", "_max_lights")

    def __init__(self, tile, max_lights):
        if not _is_real(tile) or not math.isfinite(tile) or tile <= 0.0:
            raise InvalidTileSizeError(
                "tile must be a finite positive real, got %r" % (tile,))
        if not isinstance(max_lights, int) or isinstance(max_lights, bool) \
                or max_lights <= 0:
            raise InvalidMaxLightsError(
                "max_lights must be a positive int, got %r" % (max_lights,))
        self._tile = float(tile)
        self._max_lights = int(max_lights)

    @property
    def tile(self):
        return self._tile

    @property
    def max_lights(self):
        return self._max_lights

    def tile_count(self, width, height):
        """返回 ``(cols, rows)``；尺寸非法时抛 ``InvalidAreaSizeError``。"""
        if not _is_real(width) or not math.isfinite(width) or width <= 0.0:
            raise InvalidAreaSizeError(
                "width must be a finite positive real, got %r" % (width,))
        if not _is_real(height) or not math.isfinite(height) or height <= 0.0:
            raise InvalidAreaSizeError(
                "height must be a finite positive real, got %r" % (height,))
        return (_axis_count(width, self._tile),
                _axis_count(height, self._tile))

    def cull(self, lights, width, height):
        """返回每个块的光源下标列表（行优先，按下标距离由近到远）。"""
        tile = self._tile
        max_lights = self._max_lights
        cols, rows = self.tile_count(width, height)
        width = float(width)
        height = float(height)
        count = cols * rows

        parsed = [_parse_light(item, index)
                  for index, item in enumerate(lights)]

        # buckets[cell] 收集 (到块中心平方距离, 光源下标)。
        # 块较多时用 None 延迟建列表，避免一次性分配大量空列表的开销。
        buckets = [None] * count

        for index, (cx, cy, radius) in enumerate(parsed):
            r2 = radius * radius
            lo_x = cx - radius
            hi_x = cx + radius
            lo_y = cy - radius
            hi_y = cy + radius
            # 球的包围盒与世界矩形完全不相交，必然与任何块都不相交。
            if hi_x < 0.0 or hi_y < 0.0 or lo_x > width or lo_y > height:
                continue

            # 候选块范围：包围盒覆盖的行列，两侧各放宽一格以覆盖恰好落在
            # 块边界/角点上的相切情形。候选块之后仍会做精确求交。
            c0 = int(math.floor(lo_x / tile)) - 1
            if c0 < 0:
                c0 = 0
            c1 = int(math.floor(hi_x / tile)) + 1
            if c1 > cols - 1:
                c1 = cols - 1
            r_start = int(math.floor(lo_y / tile)) - 1
            if r_start < 0:
                r_start = 0
            r_end = int(math.floor(hi_y / tile)) + 1
            if r_end > rows - 1:
                r_end = rows - 1

            for row in range(r_start, r_end + 1):
                ty0 = row * tile
                ty1 = ty0 + tile
                if ty1 > height:
                    ty1 = height
                tile_cy = (ty0 + ty1) * 0.5
                if cy < ty0:
                    dy = ty0 - cy
                elif cy > ty1:
                    dy = cy - ty1
                else:
                    dy = 0.0
                dy2 = dy * dy
                row_offset = row * cols
                for col in range(c0, c1 + 1):
                    tx0 = col * tile
                    tx1 = tx0 + tile
                    if tx1 > width:
                        tx1 = width
                    if cx < tx0:
                        dx = tx0 - cx
                    elif cx > tx1:
                        dx = cx - tx1
                    else:
                        dx = 0.0
                    if dx * dx + dy2 > r2:
                        continue
                    tile_cx = (tx0 + tx1) * 0.5
                    ddx = cx - tile_cx
                    ddy = cy - tile_cy
                    entry = (ddx * ddx + ddy * ddy, index)
                    cell = row_offset + col
                    bucket = buckets[cell]
                    if bucket is None:
                        buckets[cell] = [entry]
                    else:
                        bucket.append(entry)
                        # 保持每块至多 max_lights 项，限制内存与后续排序量。
                        if len(bucket) > 2 * max_lights:
                            buckets[cell] = heapq.nsmallest(max_lights,
                                                            bucket)

        result = []
        for bucket in buckets:
            if not bucket:
                result.append([])
            elif len(bucket) > max_lights:
                result.append([index for _, index in
                               heapq.nsmallest(max_lights, bucket)])
            else:
                bucket.sort()
                result.append([index for _, index in bucket])
        return result
