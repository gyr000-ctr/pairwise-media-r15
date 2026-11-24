# lightcull

## 题面（可直接粘贴）

子目录：lightcull。分块光源剔除。约束：每块光源集合与暴力球-块求交一致；块边界不漏；每块光源数不超过上限（超出按距离取近）；非法块尺寸可区分报错；1 万块不超过 500 ms；并发安全。验收：cd lightcull && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd lightcull && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
