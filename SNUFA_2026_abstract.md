# SNUFA 2026 Abstract Submission (Updated)

## Submission Form Fields

- **Corresponding author name:** 潘海盟 (Pan Haimeng)
- **Corresponding author email:** [你的邮箱]
- **Presenting author name:** 潘海盟 (Pan Haimeng)
- **Presentation title:** Local Plasticity Rules Match Global Optimization in Chaotic Spiking Networks for Sensorimotor Learning
- **Presentation authors:** Pan Haimeng

## Abstract (278 words)

How can a neural network learn to map sensory inputs to motor outputs without reward signals or gradient-based weight updates? Biological brains achieve this using local synaptic plasticity, yet most spiking neural network (SNN) learning algorithms rely on mechanisms unavailable to biological synapses: global covariance matrices (FORCE learning), external reward signals (reward-modulated STDP), or gradient backpropagation. We ask whether purely local plasticity rules, combined with teacher-forced guidance, can enable a chaotic spiking network to acquire cross-modal sensorimotor function. We constructed a network of 1000 LIF neurons with chaotic recurrent dynamics (spectral radius 1.5) and a plastic readout layer of 8 output neurons governing a mass-spring-damper vocal tract model. Learning used only spike-timing-dependent plasticity (STDP) and homeostatic plasticity. During training, output neurons were driven by teaching currents derived from the target vocalization trajectory, physically guiding the desired motor pattern while STDP autonomously learned which reservoir neurons correlate with the guided output. During testing, all guidance was removed; the network had to generate the correct vocalization trajectory from visual input alone. Across 10 random seeds, the method achieved a Pearson correlation of 0.865 ± 0.022 between network output and target trajectory, statistically matching both offline ridge regression (0.873 ± 0.010; p=0.29) and online RLS/FORCE with PSP filtering (0.875 ± 0.009; p=0.21) — despite using only local synaptic information (one scalar per synapse) versus global network state (a 1000×1000 covariance matrix). Ablation confirmed STDP as the core mechanism. The network generalized to unseen visual shapes (held-out correlation: 0.763–0.821). These results demonstrate that biologically available local plasticity rules can match globally optimal methods for acquiring sensorimotor mappings through guided experience, without reward signals or gradient-based weight updates.

## Notes

- Word count: 278 (limit: 300)
- Submit at: https://forms.cloud.microsoft/e/2z19jyiWqS
- Deadline: September 25, 2026 (Anywhere on Earth)
- Workshop dates: November 4-5, 2026 (online)
- Abstract will be made public for blinded review and rating after deadline
- Top-rated abstracts get contributed talks (7 slots) or flash talks; others get posters

## What changed from previous version

1. Title: "涌现感觉运动映射" -> "局部规则匹配全局优化"
2. RLS对比: "significantly outperforming (0.072, d=3.03)" -> "statistically matching (0.875, p=0.21)"
3. 新增岭回归基线: offline ridge regression (0.873)作为理论上限
4. 术语修正: "physical guidance" -> "teacher-forced guidance"
5. 核心结论: 从"STDP比FORCE强"改为"局部规则用1标量/突触匹配全局优化的1000x1000矩阵"
