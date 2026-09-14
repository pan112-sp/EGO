# SNUFA 2026 Abstract

**Presentation type:** Poster

**Presentation title:** When a correlation metric cannot tell learning from non-learning in a spiking sensorimotor network

**Presentation authors:** Pan Haimeng

---

## Abstract

We report a negative result together with a methodological caution. We trained a spiking
network — a fixed reservoir of 1000 LIF neurons with a plastic STDP readout — to map visual
shapes onto vocal-tract motor trajectories. Under the evaluation used in the project, the
Pearson correlation between produced and target trajectory, the local STDP readout appeared
statistically indistinguishable from an offline ridge-regression baseline
(0.865 ± 0.022 vs 0.873 ± 0.010, p = 0.29).

We then examined the evaluation itself. The three target trajectories are mutually correlated
at 0.836, and the network's outputs across the three shapes are mutually correlated at 0.992.
The score therefore sits near 0.86 regardless of whether the network actually discriminates
shapes. Across 20 hyperparameter configurations, this metric correlated with true shape
competence — whether the output matched its own target better than the other two — at
r = −0.016. The configuration the metric ranked highest was at chance; the one it ranked lowest
was the best in the sweep.

A confusion-matrix / rank-based measure exposes the gap: local STDP readout 47%, global ridge
readout 77%, chance 33%. Shape information is nevertheless present throughout the network —
decodable at 100% from reservoir activity and 87% from the final motor output — so the failure
is one of weak target correspondence, not of lost information.

We also found an implementation confound: the STDP readout consumed instantaneous spikes while
the baselines consumed PSP-filtered firing rates. Aligning them raised STDP from 40% to 47%,
still short of ridge.

We recommend rank-based or confusion-matrix measures whenever targets are mutually correlated.

*(261 words)*

---

## 中文对照

我们报告一个负面结果，以及一则方法学上的提醒。我们训练了一个脉冲网络——1000 个 LIF 神经元的
固定储备池，加上一个可塑的 STDP 读出层——把视觉形状映射到声带运动轨迹。在项目原本使用的评测
方式（产出轨迹与目标轨迹的皮尔逊相关）下，局部 STDP 读出与离线岭回归基线在统计上无法区分
（0.865 ± 0.022 对 0.873 ± 0.010，p = 0.29）。

随后我们检查了这个评测本身。三条目标轨迹彼此相关 0.836，而网络在三个形状下的输出彼此相关
0.992。因此无论网络是否真的区分了形状，分数都停在 0.86 附近。在 20 组超参数配置上，该度量与
真实形状能力——输出与自己那个目标的相关是否高于另外两个——的相关系数为 r = −0.016。
该度量排名最高的配置处在随机水平；排名最低的配置反而是整个扫描中最好的。

换成混淆矩阵 / 排名类度量后，差距显现：局部 STDP 读出 47%，全局岭回归读出 77%，随机水平 33%。
尽管如此，形状信息在网络中一直存在——可从储备池活动以 100% 解码，从最终运动输出以 87% 解码
——所以这是"目标对应关系弱"，而不是"信息丢失"。

我们还发现一个实现层面的混淆：STDP 读出层消耗的是瞬时脉冲，而基线消耗的是 PSP 滤波后的放电率。
对齐二者后，STDP 从 40% 提升到 47%，仍低于岭回归。

我们建议：当多个目标彼此相关时，报告排名类或混淆矩阵类度量。

---

## Note on the earlier version

An earlier version of this abstract carried the title *"Local STDP with Physical Guidance
Approaches Global Linear Readout Performance on a Small-Scale Sensorimotor Task"*. Its central
claim — that the local STDP readout matched a global linear readout, including generalization
to unseen shapes — does not survive re-evaluation. See `reanalysis/REPORT.md` and
`reanalysis/SWEEP_REPORT.md` for the corrected analysis.
