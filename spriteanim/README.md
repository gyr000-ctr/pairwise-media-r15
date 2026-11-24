# spriteanim

## 题面（可直接粘贴）

子目录：spriteanim。精灵序列帧与九宫格。约束：帧序列按时间取模且循环边界不重复；九宫格拉伸保持边角不缩放；非法九宫格参数可区分报错；10 万次取帧不超过 100 ms；并发安全。验收：cd spriteanim && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd spriteanim && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
