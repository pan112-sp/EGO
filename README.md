# EGO — When a correlation metric cannot tell learning from non-learning

**A negative result and a methodological caution, from a spiking sensorimotor learning project.**

[中文版见下方](#中文版)

---

## What this is

A spiking neural network (1000 LIF neurons, **fixed** chaotic reservoir + plastic STDP readout)
was trained to map visual shapes onto vocal-tract motor trajectories, using only local synaptic
plasticity and teacher forcing — no reward, no gradients, no global optimiser.

The project originally reported that the local STDP readout matched an offline ridge-regression
baseline. **Re-analysis showed this conclusion does not hold**, because the evaluation metric
used cannot distinguish a network that learned the task from one that did not.

## Main finding

The evaluation was the Pearson correlation between the produced and the target trajectory.
On this task it has **zero correlation with actual task competence**:

| | |
|---|---|
| mutual correlation among the three target trajectories | **0.836** |
| mutual correlation among the network's three outputs | **0.992** |
| correlation between the old metric and true shape competence, over 20 hyperparameter configs | **r = −0.016** |
| the config the old metric ranked **highest** | **33% = chance** |
| the config the old metric ranked **lowest** | **56% = best in the sweep** |

True shape competence = does the output correlate with its *own* target more than with the
other two (chance = 33%).

## Results under a discriminating metric

| Method | old metric score | old metric rank-1 | **new metric rank-1** | mutual output similarity |
|---|---|---|---|---|
| STDP readout (as-is) | 0.865 | 27% | **40%** | **0.992** |
| STDP readout + PSP filtering (sweep best) | — | 37% | **47%** | 0.986 |
| Ridge regression (baseline) | 0.873 | 67% | **77%** | **0.822** |
| chance | — | 33% | 33% | — |

Shape information is still present throughout the network (decodable at 100% from reservoir
activity and 87% from the final motor output), so the failure is one of **weak target
correspondence**, not of lost information.

We also found that the STDP readout consumed instantaneous spikes while the baselines consumed
PSP-filtered firing rates — an asymmetry that disfavours STDP. Aligning them raised STDP from
40% to 47%, still short of ridge.

**Recommendation:** when targets are mutually correlated, report a confusion matrix or a
rank-based measure, not the correlation with a single target.

## Repository layout

```
reanalysis/          <- the main contribution: re-evaluation + sweep
  reanalyze.py         contrastive metrics, layer-wise decodability   (~215 s)
  sweep_psp.py         PSP fairness + parameter sweep                 (~450 s)
  report_only.py       regenerate REPORT.md from results.json
  results.json         all numbers from reanalyze.py
  REPORT.md            full report (Chinese)
  SWEEP_REPORT.md      sweep report (Chinese)
force_phase3.py      earlier fixed-reservoir code
rigorous_experiments.py   experiments (see the WARNING in exp1_baselines)
rls_fixed.py         ridge / RLS baselines (fair, PSP-filtered)
rls_fix_results.txt  baseline logs
SNUFA_2026_abstract.md    workshop abstract (corrected version)
```

## Reproduce

```bash
pip install -r requirements.txt
python -B reanalysis/reanalyze.py      # ~4 min
python -B reanalysis/sweep_psp.py      # ~8 min
```

Both write their results into `reanalysis/`.

## Limitations

- **One task, one architecture** (3 shapes, 1000 neurons, a simplified vocal-tract model).
- **The sweep covers two dimensions only** (input scale × learning rate); teaching current,
  PSP time constant and network size were held fixed. Whether a configuration outside the
  searched region would close the gap is **unknown**.
- **We do not claim that local rules are fundamentally incapable.** We report only that, in
  this setting and within this search, they did not match the global readout.
- The sweep used 3 seeds and is noisy (3-seed best 56% → 47% under 10-seed revalidation).

## License

MIT

---
---

# 中文版

## 这是什么

一个脉冲神经网络（1000 个 LIF 神经元，**固定**的混沌储备池 + 可塑的 STDP 读出层），
被训练来把视觉形状映射到声带运动轨迹。只使用局部突触可塑性和教师强制——
不用奖惩、不用梯度、不用全局优化器。

项目原本报告的结论是：局部 STDP 读出达到了离线岭回归基线的水平。
**重新评测后，这个结论不成立**——因为所用的评测方式分不出"学会了"和"没学会"。

## 主要发现

原评测是"产出轨迹与目标轨迹的皮尔逊相关"。它在本任务上**与真实任务能力零相关**：

| | |
|---|---|
| 三条目标轨迹彼此的相关 | **0.836** |
| 网络三次输出彼此的相关 | **0.992** |
| 旧度量与真实形状能力在 20 组配置上的相关 | **r = −0.016** |
| 旧度量排名**最高**的配置 | **33% = 随机水平** |
| 旧度量排名**最低**的配置 | **56% = 扫描中最优** |

（"真实形状能力" = 输出与自己那个目标的相关，是否高于另外两个；随机水平 33%）

## 换成能分辨的度量之后

| 方法 | 旧度量分数 | 旧度量 rank-1 | **新度量 rank-1** | 输出互相相似度 |
|---|---|---|---|---|
| STDP 读出（原样） | 0.865 | 27% | **40%** | **0.992** |
| STDP 读出 + PSP 滤波（扫描最优） | — | 37% | **47%** | 0.986 |
| 岭回归（基线） | 0.873 | 67% | **77%** | **0.822** |
| 随机水平 | — | 33% | 33% | — |

形状信息在网络中一直存在（可从储备池活动以 100% 解码、从最终运动输出以 87% 解码），
所以问题在于**目标对应关系弱**，而不是信息丢失。

另外发现：STDP 读出层消耗的是瞬时脉冲，而基线消耗的是 PSP 滤波后的放电率——
这个不一致对 STDP 不利。对齐后 STDP 从 40% 提升到 47%，仍低于岭回归。

**建议**：当多个目标彼此相关时，报告混淆矩阵或排名类度量，而不是单一目标的相关系数。

## 目录结构

```
reanalysis/          <- 现在的主要成果：重新评测 + 参数扫描
  reanalyze.py         对比式度量、逐层可解码性        （约 215 秒）
  sweep_psp.py         PSP 公平化 + 参数扫描           （约 450 秒）
  report_only.py       从 results.json 重生成报告
  results.json         reanalyze.py 的全部数字
  REPORT.md            完整报告
  SWEEP_REPORT.md      扫描报告
force_phase3.py      早期固定储备池代码
rigorous_experiments.py   实验（注意 exp1_baselines 里的 WARNING）
rls_fixed.py         岭回归 / RLS 基线（公平版，带 PSP 滤波）
rls_fix_results.txt  基线日志
SNUFA_2026_abstract.md   投稿摘要（修正版）
```

## 复现

```bash
pip install -r requirements.txt
python -B reanalysis/reanalyze.py      # 约 4 分钟
python -B reanalysis/sweep_psp.py      # 约 8 分钟
```

## 局限

- **只有一个任务、一个架构**（3 个形状、1000 神经元、简化的声带模型）。
- **参数扫描只覆盖两个维度**（输入尺度 × 学习率）；教导电流、PSP 时间常数、
  网络规模都是固定的。**扫描范围之外是否存在能追上的配置，未知。**
- **我们不主张"局部规则原理上做不到"。** 只报告：在此设定和此搜索范围内没有做到。
- 扫描阶段用 3 个种子，噪声较大（3 种子下最优 56%，10 种子复核后 47%）。

## 许可证

MIT
