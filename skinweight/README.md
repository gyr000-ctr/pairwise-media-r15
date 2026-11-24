# skinweight

## 任务

对偶四元数蒙皮（Dual Quaternion Skinning, DQS），并保留原有的蒙皮权重限项/归一能力。

### 约束

- **权重限项归一**：每顶点最多保留 4 个最大权重影响并重新归一，归一残差吸收到
  最大分量，行和误差 ≤ 1e-9；
- **DQS 与 LBS 一致性**：纯旋转（单骨骼绑定）下两者输出之差 ≤ 1e-6；
- **扭转体积损失**：强扭转（candy-wrapper）下 LBS 出现可测的体积/面积损失，
  混合矩阵行列式坍缩；DQS 保持刚体性质（体积不变、距离保持）；
- **反对偶符号统一**：混合前以首影响骨骼实部为基准，凡点积为负的对偶四元数
  整体翻转，消除单位四元数 ``q`` 与 ``-q`` 的双重覆盖歧义；
- **错误可区分**：零权重（:class:`ZeroWeightError`）与负权重
  （:class:`NegativeWeightError`）分别报错，非有限输入为
  :class:`NonFiniteWeightError`，三者均为 :class:`ValueError` 子类；
- **性能**：1 万顶点 DQS 不超过 500 ms（实测约 6 ms，NumPy 向量化）；
- **并发安全**：:class:`Skinner` 只持有只读骨骼数组，调用无共享可变状态，
  可被多线程并发调用且结果逐位一致。

## 运行方式

```bash
cd skinweight && python -m pytest tests -q
```

无外部运行时依赖（仅 NumPy + pytest）。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `skinweight/errors.py` | 可区分的异常类型（零/负/非有限权重、限项非法、变换非法） |
| `skinweight/dualquaternion.py` | 四元数运算与刚体变换的单位对偶四元数（含矩阵互转） |
| `skinweight/weights.py` | `SkinWeight` 一维 API 与稠密/稀疏权重的校验、限项、归一 |
| `skinweight/skinning.py` | `Skinner`：向量化 DQS / LBS、反对偶符号统一、Kavan 归一化 |
| `tests/test_core.py` | 原权重限项归一 API 验收 |
| `tests/test_dqs.py` | DQS 全部约束、错误分类、性能、并发安全验收 |

## 关键公式

刚体变换 ``(R, t)`` 的单位对偶四元数（``t_quat = [0, tx, ty, tz]``）：

```
real = q          （旋转单位四元数 [w, x, y, z]）
dual = 0.5 * t_quat * q
```

逐顶点混合（Kavan et al. 2007）：

```
对每个影响 j：若 dot(real_0, real_j) < 0，则 real_j, dual_j 同时取负
blend = Σ w_j * dq_j
real' = blend_real / |blend_real|,  dual' = blend_dual / |blend_real|
t' = 2 * (dual' * conj(real')).xyz
p'  = rotate(real', p) + t'
```
