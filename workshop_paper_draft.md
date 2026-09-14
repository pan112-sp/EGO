> # ⚠️ SUPERSEDED — DO NOT CITE
>
> **This draft is retained for historical record only. Its central claim does not hold.**
>
> The claim below — that the local STDP readout "approaches global linear readout performance",
> including generalization to unseen shapes — was based on a correlation metric that, on this
> task, has **zero correlation with actual task competence** (across 20 hyperparameter
> configurations, r = −0.016; the configuration the metric ranked highest was at chance level).
>
> Under a discriminating (confusion-matrix / rank-based) measure, the two methods are far
> apart: local STDP readout **47%**, global ridge readout **77%**, chance **33%**.
>
> See [`reanalysis/REPORT.md`](reanalysis/REPORT.md),
> [`reanalysis/SWEEP_REPORT.md`](reanalysis/SWEEP_REPORT.md), and the corrected abstract in
> [`SNUFA_2026_abstract.md`](SNUFA_2026_abstract.md).
>
> ---
>
> **本稿仅作历史留存，请勿引用。** 其核心结论不成立：支撑该结论的评测方式在本任务上
> 与真实任务能力**零相关**（20 组配置，r = −0.016）。换成对比式度量后，局部 STDP 读出
> 47%，全局岭回归 77%，随机水平 33%。详见 `reanalysis/REPORT.md`。

# Local STDP with Physical Guidance Approaches Global Linear Readout Performance on a Small-Scale Sensorimotor Task

**Abstract**

Can a chaotic spiking neural network learn cross-modal sensorimotor mappings using only local synaptic weight updates, without reward signals or gradient-based optimization? We investigate this question in a small-scale system where visual shapes must be mapped to vocalization trajectories. Our architecture consists of a fixed recurrent reservoir of 1000 leaky integrate-and-fire (LIF) neurons with chaotic dynamics (spectral radius ≈ 5.0), paired with a plastic readout layer of 8 output neurons controlling a mass-spring-damper vocal tract model. Training uses a physical guidance protocol: output neurons are driven by teaching currents proportional to the error between current and target output, physically guiding the correct motor pattern while spike-timing-dependent plasticity (STDP) autonomously learns which reservoir neurons correlate with the guided output. The teaching current contains an error term (target − output) — a form of teacher forcing — but this error drives neuron activity, not weight updates; all synaptic modifications follow local STDP and homeostatic plasticity rules. After training, all guidance is removed and the network must generate vocalization trajectories from visual input alone. Across 10 random seeds, the method achieves a Pearson correlation of 0.865 ± 0.022 between network output and target trajectory. We compare against both offline ridge regression (theoretical upper bound for linear readout) and online RLS-based FORCE learning with PSP filtering. Both global methods achieve slightly higher performance (ridge: 0.873 ± 0.010; RLS: 0.875 ± 0.009), and the difference from STDP does not reach statistical significance (ridge: p = 0.29, Cohen's d = −0.52; RLS: p = 0.21, Cohen's d = −0.61). However, the moderate effect sizes and small sample size mean that statistical equivalence is not established. Critically, STDP achieves this using only local spike-timing information — each synapse accesses only its own pre- and post-synaptic spike times — while ridge regression requires a 1000×1000 covariance matrix and RLS maintains a global inverse covariance estimate. We further show that spike-rate readout, while less stable than membrane potential readout (0.675 ± 0.304 vs. 0.865 ± 0.022), also supports learning, confirming that the mechanism does not fundamentally depend on analog membrane potential access. Ablation experiments identify STDP as the core learning mechanism; removing STDP yields systematic anti-correlation (−0.916), reflecting homeostatic-only suppression of active synapses. The network generalizes to unseen visual shapes (held-out correlation: 0.763–0.821) and to position-shifted (0.794) and size-scaled (0.549) variants of trained shapes. These results suggest that local synaptic weight-update rules, when paired with a global physical guidance signal, can approach the performance of global linear readout optimization on this task, though statistical equivalence and general scalability remain unproven.

---

## 1. Introduction

How do biological brains learn to map sensory inputs to motor outputs? The dominant frameworks in artificial neural networks — gradient descent and reward-based learning — rely on information that biological synapses cannot access: global error gradients or centralized reward signals. Yet biological brains learn sensorimotor skills effortlessly, suggesting that local plasticity rules must be sufficient when paired with the right kind of experience.

This paper investigates whether local synaptic weight-update rules (STDP and homeostatic plasticity), combined with a physical guidance training protocol, can enable a chaotic spiking neural network to acquire a cross-modal sensorimotor mapping (visual shape → vocalization trajectory). We ask how close a local weight-update rule can come to global linear readout optimization methods, rather than claiming equivalence.

The physical guidance protocol works as follows: during training, output neurons are injected with a teaching current that drives them toward the correct firing pattern. This current is proportional to the error between current output and target trajectory. While this error signal is a form of teacher forcing, the critical distinction is that the error drives neuron activity, not weight updates. All synaptic modifications are determined by local STDP rules — each synapse only knows the timing of its own pre- and post-synaptic spikes. The network "experiences" the correct sensorimotor trajectory, and local plasticity does the rest.

This is analogous to how infants learn to speak: caregivers model correct articulations, and the infant's brain learns the association between sensory input and motor patterns through experience. No explicit reward signal tells the infant whether a vocalization is "correct"; rather, the infant's own sensorimotor system experiences the right movements and learns from them.

We make the following contributions:

1. We demonstrate that STDP with physical guidance enables a 1000-neuron chaotic spiking network to learn a visual-to-vocalization mapping with high reliability (0.865 ± 0.022 correlation across 10 seeds).

2. We show that this local-plasticity approach shows no statistically significant difference from both offline ridge regression (0.873 ± 0.010) and online RLS-based FORCE learning (0.875 ± 0.009) on the same task — though moderate effect sizes and limited statistical power mean equivalence is not established.

3. We characterize the readout mechanism: membrane potential readout achieves 0.865 ± 0.022 while spike-rate readout achieves 0.675 ± 0.304, establishing that the learning works with both readout schemes, though stability differs.

4. We provide comprehensive ablation, generalization, scaling, and parameter sensitivity analyses that characterize each mechanism's contribution and the method's current limitations.

---

## 2. Method

### 2.1 Network Architecture

**Reservoir.** N = 1000 LIF (Leaky Integrate-and-Fire) neurons with 10% random recurrent connectivity. Weights are drawn from a Gaussian distribution scaled by g / √(p·N) where g = 5.0 and p = 0.1, yielding an initial spectral radius of approximately 5.0. Neurons are divided into excitatory (80%) and inhibitory (20%) populations. Inhibitory weights are negative, scaled by a factor of w_inh relative to excitatory weights.

**Membrane potential dynamics:**

dV/dt = (V_rest − V) / τ_m + I_syn / C_m + I_bias / C_m

When V > V_th, a spike is emitted, V is reset to V_reset, and the neuron enters a refractory period of τ_ref ms.

**Output layer.** 8 output neurons receive input from all reservoir neurons. Output weights are plastic (STDP + homeostatic plasticity); reservoir weights are fixed. The 8 output neurons control 8 parameters of a mass-spring-damper vocal tract model.

**Visual encoder.** 20×10 = 200 visual input neurons encode shape position on a grid. Each visual neuron fires at a rate proportional to the pixel intensity at its receptive field location.

### 2.2 Learning Rules

**STDP (Spike-Timing-Dependent Plasticity).** If a pre-synaptic spike arrives before a post-synaptic spike (Δt > 0), the synapse is potentiated. If the post-synaptic spike arrives first (Δt < 0), the synapse is depressed.

Δw = A_+ · exp(−Δt / τ_+)  for Δt > 0 (LTP)
Δw = −A_- · exp(Δt / τ_-)  for Δt < 0 (LTD)

Parameters: τ_+ = τ_- = 20 ms, A_+ = 0.01, A_- = 0.01.

**Homeostatic plasticity.** Output neuron target firing rate: 30 Hz. If actual rate exceeds target, all incoming weights are slightly reduced; if below target, they are slightly increased. This maintains stability without introducing reward signals.

### 2.3 Physical Guidance Training Protocol

During training, output neurons receive a teaching current:

I_teach = gain × (target − output)

This current drives output neurons toward the target firing pattern. Crucially, this error signal drives *neuron activity*, not *weight updates*. All synaptic modifications follow STDP — which observes the resulting spike timing patterns and adjusts weights accordingly.

The training protocol uses a gradual transition from full guidance to autonomous operation: the mix parameter goes from 0 (full guidance) to 1 (autonomous) over the course of training, allowing STDP to consolidate learned patterns as guidance is withdrawn.

### 2.4 Baselines

**Offline Ridge Regression.** We collect 9000 time steps of PSP-filtered reservoir states and corresponding target forces, then solve the ridge regression problem W = argmin ‖XW − Y‖² + α‖W‖² in closed form. This represents the theoretical upper bound for linear readout from this reservoir — it has access to all data simultaneously and solves a convex optimization problem. We sweep PSP time constants (τ_psp = 20, 50, 100, 200 ms) and regularization strengths (α = 0.01–100).

**Online RLS (FORCE learning).** Following Sussillo & Abbott (2009), we implement recursive least squares with PSP-filtered spiking input. The RLS algorithm maintains a 1000×1000 inverse covariance matrix P and updates weights online. We use strong numerical stabilization: float64 precision, symmetric P matrix, PSP normalization, and P_max clipping. We test the same PSP time constants (20, 50, 100, 200 ms).

For both baselines, training uses the same precomputed target force trajectories and the same reservoir dynamics as the STDP method. Testing is performed closed-loop: the readout drives the vocal tract model, and correlation between achieved and target positions is measured.

---

## 3. Experiments

### 3.1 Main Result

We compare three methods on the visual-to-vocalization sensorimotor mapping task, across 10 random seeds:

| Method | Correlation (mean ± std) | Information per synapse |
|--------|--------------------------|------------------------|
| STDP + physical guidance (membrane readout) | 0.865 ± 0.022 | 1 scalar (pre/post spike times) |
| Offline Ridge Regression | 0.873 ± 0.010 | 1000×1000 covariance matrix |
| Online RLS / FORCE | 0.875 ± 0.009 | 1000×1000 inverse covariance matrix |

The STDP method shows no statistically significant difference from either baseline (ridge: t = −1.10, p = 0.29, Cohen's d = −0.52; RLS: t = −1.30, p = 0.21, Cohen's d = −0.61). The effect sizes are moderate (STDP is slightly lower), and with only 10 seeds the study has limited statistical power. The absence of a significant difference does not constitute proof of equivalence; formal equivalence testing would require a larger sample size and a pre-specified equivalence bound.

### 3.2 Spike Rate Readout

When using spike-rate readout (100-step window) instead of membrane potential readout, performance drops to 0.675 ± 0.304. Approximately 1 in 10 seeds fails to converge, resulting in high variance. The mechanism remains functional — learning clearly occurs — but stability is reduced.

### 3.3 Ablation

| Condition | Correlation | Interpretation |
|-----------|-------------|----------------|
| Full model (STDP + homeostatic) | 0.865 ± 0.022 | Baseline |
| No STDP (homeostatic only) | −0.916 ± 0.031 | Systematic anti-correlation: homeostatic-only suppression of active synapses drives output opposite to target |
| No homeostatic (STDP only) | 0.218 ± 0.345 | Unstable, most seeds fail |
| No teaching current | 0.041 ± 0.029 | Random performance |

Ablation confirms STDP as the core learning mechanism. The strongly negative correlation in the "no STDP" condition (−0.916) is not merely an absence of learning; it reflects homeostatic plasticity alone systematically suppressing the most active synapses, which in a teacher-forced setup are precisely those encoding the correct pattern. Removing homeostatic plasticity leads to instability and divergence.

### 3.4 Generalization

**Held-out shape generalization.** Training on 2 of 3 shapes and testing on the held-out shape yields held-out correlations ranging from 0.763 to 0.821 depending on which shape is held out. This demonstrates that the network learns something more general than memorization of three specific patterns.

**Position-shifted variants.** Testing with shapes shifted 2 units horizontally and 1 unit vertically yields 0.794 correlation, indicating partial translation invariance.

**Size-scaled variants.** Testing with shapes scaled to 70% of training size yields 0.549 correlation, showing limited but non-trivial size generalization.

### 3.5 Network Size

| Network size | Correlation |
|-------------|-------------|
| 200 neurons | 0.712 ± 0.063 |
| 500 neurons | 0.834 ± 0.031 |
| 1000 neurons | 0.865 ± 0.022 |
| 2000 neurons | 0.495 ± 0.312 |
| 5000 neurons | 0.102 ± 0.187 |

Performance peaks at 1000 neurons and collapses above that, likely because the fixed spectral radius becomes too large in bigger networks. This is a known limitation of fixed-reservoir approaches.

### 3.6 Parameter Sensitivity

The method is most sensitive to STDP learning rate and teaching current gain. Performance is best in a relatively narrow operating range, suggesting that the method currently requires careful tuning.

---

## 4. Discussion

### 4.1 Interpretation of Results

The main finding is that STDP with physical guidance achieves 0.865 correlation, approaching the 0.873 of offline ridge regression and 0.875 of online RLS/FORCE. The gap is small (about 1%), and the difference does not reach statistical significance at p < 0.05 with 10 seeds. However, a non-significant p-value does not establish equivalence. The moderate effect sizes (Cohen's d ≈ −0.5 to −0.6) indicate that STDP is somewhat below the global baselines, and the non-significant result may partly reflect low statistical power. A formal equivalence test (e.g., TOST) with more seeds and a pre-specified equivalence bound would be needed to support stronger claims.

From an information-theoretic perspective, the more interesting observation is that a local weight-update rule using one scalar per synapse can come this close to a global linear readout method using a 1000×1000 matrix. The 1% performance gap comes with a many-orders-of-magnitude reduction in per-synapse information requirements.

### 4.2 Mechanism

The proposed mechanism is straightforward: physical guidance drives output neurons toward the correct pattern; STDP then strengthens synapses from reservoir neurons whose spikes reliably precede the guided output spikes; over time, these strengthened connections allow the reservoir to drive the correct output pattern even after guidance is removed. Homeostatic plasticity provides stability, preventing runaway weight growth or synaptic decay.

The −0.916 anti-correlation in the no-STDP ablation is revealing: it shows that homeostatic plasticity alone systematically weakens active synapses (because high-firing neurons trigger homeostatic down-scaling). In the full model, STDP's potentiation of correctly-timed synapses must overcome this homeostatic pressure — a tug-of-war that ultimately settles at the learned configuration.

### 4.3 Biological Plausibility

The method draws on mechanisms known to exist in biological brains: STDP, homeostatic synaptic scaling, and teacher-forced learning through motor experience. However, several caveats apply:

1. **Membrane potential readout** is not biologically realistic as a decoding mechanism. Real neurons communicate via spikes, not via direct access to each other's membrane potentials. The spike-rate readout result (0.675 ± 0.304) is more biologically grounded but less reliable.

2. **The teaching current** uses an explicit error term (target − output) that in biology would need to be instantiated through sensory feedback or efference copy, not through direct access to a target trajectory.

3. **Fixed reservoir weights** are an abstraction. In real brains, recurrent connections are also plastic. Testing whether the same principles apply in a fully plastic recurrent network is an important direction.

### 4.4 Comparison to Existing Work

**FORCE learning (Sussillo & Abbott, 2009):** The original FORCE was demonstrated in rate-based chaotic networks. We show that when properly implemented with PSP filtering and numerical stabilization, FORCE achieves strong performance (0.875 ± 0.009) on our spiking task, slightly above STDP with physical guidance. The key distinction is not performance level but the information requirements: FORCE uses a global inverse covariance matrix for weight updates, while STDP uses only local spike timing.

**Reward-modulated STDP (Florian 2007, Izhikevich 2007):** These approaches use a global reward signal to gate STDP. Our method avoids reward signals entirely, relying instead on the physical guidance protocol.

**Reservoir computing (Maass et al. 2002, Jaeger 2001):** The fixed-reservoir architecture follows the reservoir computing paradigm. What is new is the demonstration that local STDP weight updates, without any global weight-update rule, can learn the readout weights to high accuracy when paired with a global physical guidance signal.

---

## 5. Limitations

Several limitations should be noted:

1. **Scalability.** Performance collapses above 1000 neurons without parameter retuning. The method currently relies on a carefully tuned operating point (spectral radius ≈ 5.0). Biological systems maintain stability across scale through mechanisms absent from our model (e.g., structural plasticity, neuromodulation, hierarchical organization). This work should be understood as a **proof of concept** demonstrating that local synaptic weight updates can approach global linear readout optimization performance in a controlled setting — not as a general-purpose algorithm ready for large-scale deployment.

2. **Task complexity.** The task involves only 3 visual shapes, 8 vocal tract parameters, and 1000 time-step trajectories. While this suffices for a proof of concept, real sensorimotor learning involves far richer sensory input, more degrees of freedom, noise, delays, and dynamic interactions with the environment. Extending to more complex tasks is essential.

3. **Biological realism of readout.** Membrane potential readout (0.865) substantially outperforms spike-rate readout (0.675). While the mechanism is robust with membrane-potential access, the biological case relies on the spiking output, which is less stable. Investigating more biologically plausible decoding schemes (e.g., population coding, downstream integration) is important future work.

4. **Statistical power.** With only 10 seeds, the study has limited power to detect small-to-moderate effect sizes. The lack of significant difference between STDP and global baselines should not be interpreted as proof of equivalence. A larger-scale study with formal equivalence testing (e.g., TOST) and a pre-specified equivalence bound would be needed for stronger claims.

5. **Fixed reservoir.** The recurrent reservoir weights are fixed during learning. While this follows the reservoir computing paradigm, it is not biologically realistic — real cortical circuits modify recurrent connections during learning. Extending to a fully plastic recurrent network is an important next step.

---

## 6. Conclusion

In a small-scale chaotic spiking network (1000 neurons, 3 shapes, 8 motor parameters), local STDP weight updates paired with a global physical guidance signal achieve performance that shows no statistically significant difference from offline ridge regression and online RLS/FORCE (0.865 vs 0.873/0.875), though effect sizes indicate STDP is slightly lower. This suggests that local synaptic weight-update rules, when paired with appropriate guidance, can approach the performance of global linear readout optimization. However, statistical equivalence is not established, and general scalability to larger networks and more complex tasks remains unknown. The primary contribution is a clear proof-of-concept demonstration that a local plasticity rule can come surprisingly close to global linear readout optimization on a sensorimotor task — using orders of magnitude less information per synapse.
