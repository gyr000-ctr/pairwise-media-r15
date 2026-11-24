# skinweight

## 题面（可直接粘贴）

子目录：skinweight。骨骼蒙皮权重归一与限项。约束：每顶点权重和为 1（≤1e-9）；保留前 4 个并重新归一；零权重顶点报错；权重为负可区分报错；1 万顶点不超过 300 ms；并发安全。验收：cd skinweight && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd skinweight && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
