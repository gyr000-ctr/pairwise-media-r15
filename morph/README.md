# morph

## 题面（可直接粘贴）

子目录：morph。形态学腐蚀与膨胀。约束：3x3 与 5x5 结构元结果与逐像素对照一致；先开后闭的幂等性；边界策略可配（clamp/zero）；空图报错；100 万像素不超过 500 ms；并发安全。验收：cd morph && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd morph && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
