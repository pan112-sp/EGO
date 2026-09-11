# Local Plasticity Matches Global Optimization in Chaotic Spiking Networks: STDP with Physical Guidance for Cross-Modal Sensorimotor Learning

**Abstract**

Can a chaotic spiking neural network learn cross-modal sensorimotor mappings using only local synaptic plasticity, without reward signals or gradient-based weight optimization? We investigate this question in a system where visual shapes must be mapped to vocalization trajectories. Our architecture consists of a fixed recurrent reservoir of 1000 leaky integrate-and-fire (LIF) neurons with chaotic dynamics (spectral radius ≈ 1.5), paired with a plastic readout layer of 8 output neurons controlling a mass-spring-damper vocal tract model. Training uses a physical guidance protocol: output neurons are driven by teaching currents proportional to the error between current and target output, physically guiding the correct motor pattern while spike-timing-dependent plasticity (STDP) autonomously learns which reservoir neurons correlate with the guided output. The teaching current contains an error term (target − output) — a form of teacher forcing — but this error drives neuron activity, not weight updates; all synaptic modifications follow local STDP and homeostatic plasticity rules. After training, all guidance is removed and the network must generate vocalization trajectories from visual input alone. Across 10 random seeds, the method achieves a Pearson correlation of 0.865 ± 0.022 between network output and target trajectory. We compare against both offline ridge regression (theoretical upper bound for linear readout) and online RLS-based FORCE learning with PSP filtering. Both global methods achieve comparable performance (ridge: 0.873 ± 0.010; RLS: 0.875 ± 0.009), confirming that the task is solvable by linear readout from this reservoir. Critically, STDP achieves statistically equivalent performance using only local spike-timing information — each synapse accesses only its own pre- and post-synaptic spike times — while ridge regression requires a 1000×1000 covariance matrix and RLS maintains a global inverse covariance estimate. We further show that spike-rate readout, while less stable than membrane potential readout (0.675 ± 0.304 vs. 0.865 ± 0.022), also supports learning, confirming that the mechanism does not fundamentally depend on analog membrane potential access. Ablation experiments identify STDP as the core learning mechanism, with homeostatic plasticity providing auxiliary stabilization. The network generalizes to unseen visual shapes (held-out correlation: 0.763–0.821). These results demonstrate that purely local plasticity rules, combined with physical guidance, can match the performance of global optimization methods on chaotic spiking networks, suggesting that biological brains may not need global error signals to achieve comparable sensorimotor learning.

---

## 1. Introduction

How do biological brains learn to map sensory inputs to motor outputs? The dominant frameworks in artificial neural networks — gradient descent and reward-based learning — rely on information that biological synapses cannot access: global error gradients or centralized reward signals. Yet biological brains learn sensorimotor skills effortlessly, suggesting that local plasticity rules must be sufficient when paired with the right kind of experience.

This paper investigates whether purely local plasticity rules (STDP and homeostatic plasticity), combined with a physical guidance training protocol, can enable a chaotic spiking neural network to acquire a cross-modal sensorimotor mapping (visual shape → vocalization trajectory). Crucially, we ask not whether local rules *outperform* global methods, but whether they can *match* them — achieving comparable performance without access to global information.

The physical guidance protocol works as follows: during training, output neurons are injected with a teaching current that drives them toward the correct firing pattern. This current is proportional to the error between current output and target trajectory. While this error signal is a form of teacher forcing, the critical distinction is that the error drives neuron activity, not weight updates. All synaptic modifications are determined by local STDP rules — each synapse only knows the timing of its own pre- and post-synaptic spikes. The network "experiences" the correct sensorimotor trajectory, and local plasticity does the rest.

This is analogous to how infants learn to speak: caregivers model correct articulations, and the infant's brain learns the association between sensory input and motor patterns through experience. No explicit reward signal tells the infant whether a vocalization is "correct"; rather, the infant's own sensorimotor system experiences the right movements and learns from them.

We make the following contributions:

1. We demonstrate that STDP with physical guidance enables a 1000-neuron chaotic spiking network to learn a visual-to-vocalization mapping with high reliability (0.865 ± 0.022 correlation across 10 seeds).

2. We show that this local-plasticity approach achieves performance statistically equivalent to both offline ridge regression (0.873 ± 0.010) and online RLS-based FORCE learning (0.875 ± 0.009) on the same task — using only local spike-timing information rather than global covariance matrices.

3. We characterize the readout mechanism: membrane potential readout achieves 0.865 ± 0.022 while spike-rate readout achieves 0.675 ± 0.304, establishing that the learning works with both readout schemes, though stability differs.

4. We provide comprehensive ablation, generalization, scaling, and parameter sensitivity analyses that characterize each mechanism's contribution and the method's current limitations.

---

## 2. Methods

### 2.1 Network Architecture

The network consists of a fixed recurrent reservoir and a plastic readout layer.

**Recurrent Reservoir.** N = 1000 LIF (Leaky Integrate-and-Fire) neurons with 10% random recurrent connectivity. Weights are drawn from a Gaussian distribution scaled by g / √(p·N) where g = 5.0 and p = 0.1, yielding an initial spectral radius of approximately 5.0. **The recurrent weights remain fixed throughout all training and testing; only readout layer weights (W_out) are plastic.**

The reservoir receives visual input encoding geometric shapes (square, circle, triangle) as spatial spike patterns on a 20×10 = 200 neuron grid, using rate coding.

**Plastic Readout Layer.** 8 LIF output neurons, each corresponding to one vocal tract muscle parameter. Each output neuron receives weighted input from all reservoir neurons via plastic weights W_out (8 × 1000). We compare two readout schemes:

- **Membrane potential readout:** The sub-threshold membrane potentials of output neurons are read as continuous commands, with global mean centering: cmd = (v_out − mean(v_out)) × gain. This provides a clean continuous signal but is less biologically plausible.
- **Spike-rate readout:** Output neuron firing rates are estimated over a sliding window (50 or 100 steps) and used as commands. This is more biologically realistic but introduces temporal smoothing and noise.

### 2.2 Plasticity Mechanisms

Two local plasticity rules operate in the readout layer:

**Spike-Timing-Dependent Plasticity (STDP).** Following Bi & Poo (1998), synaptic weight change depends on the relative timing of pre- and post-synaptic spikes:

- Pre before post (Δt > 0): Δw = A₊ · exp(−Δt/τ₊)  [LTP]
- Post before pre (Δt < 0): Δw = −A₋ · exp(Δt/τ₋)  [LTD]

Parameters: A₊ = A₋ = 0.01, τ₊ = τ₋ = 20 ms. Weights are clipped to [0, w_max] with w_max = 5.0.

**Homeostatic Plasticity.** Each output neuron tracks its firing rate via an exponential moving average and adjusts weight scaling to maintain a target rate (~10 Hz). This prevents neurons from falling silent or saturating.

An eligibility trace mechanism was also implemented but had no measurable effect on this task (see Section 4.2), so it is not included in the final model. We discuss it as a potential extension for tasks with longer sensorimotor delays.

### 2.3 Physical Guidance Protocol

Training proceeds as follows:

1. A visual shape is presented to the reservoir.
2. The target vocalization trajectory is known.
3. The teaching current is computed as I_teach = gain × (target_force(t) − current_output(t)).
4. This current is injected into output neurons, driving them toward the correct firing pattern.
5. STDP modifies W_out based on the relative timing of reservoir (pre) and output (post) spikes.
6. Over training, the network progressively takes over: the mix shifts from 100% target force to 100% network output.

During testing, the teaching current is completely removed. The network must generate the vocalization trajectory from visual input alone.

The teaching current explicitly contains an error term (target minus output). This is a form of teacher forcing — the output neurons are externally clamped toward the target trajectory. However, this error signal does not directly compute weight updates. Instead, it shapes the output neurons' firing patterns, and STDP — a purely local rule — determines which synapses strengthen or weaken based on spike timing.

### 2.4 Vocal Tract Model and Target Trajectories

A mass-spring-damper vocal tract model with 8 muscle parameters generates vocalization trajectories. Each muscle is a second-order dynamical system (mass, spring, damping).

Three geometric shapes (square, circle, triangle) map to distinct 8-parameter vocalization trajectories of 1000 time steps each. Each trajectory has different temporal characteristics: the square produces a rapid oscillation pattern, the circle produces an envelope-shaped trajectory, and the triangle produces a complex multi-frequency pattern.

### 2.5 Baseline Methods

We compare against two global optimization baselines:

**Offline Ridge Regression.** We collect 9000 time steps of PSP-filtered reservoir states and corresponding target forces, then solve the ridge regression problem W = argmin ‖XW − Y‖² + α‖W‖² in closed form. This represents the theoretical upper bound for linear readout from this reservoir — it has access to all data simultaneously and solves a convex optimization problem. We sweep PSP time constants (τ_psp = 20, 50, 100, 200 ms) and regularization strengths (α = 0.01–100).

**Online RLS (FORCE learning).** Following Sussillo & Abbott (2009), we implement recursive least squares with PSP-filtered spiking input. The RLS algorithm maintains a 1000×1000 inverse covariance matrix P and updates weights online. We use strong numerical stabilization: float64 precision, symmetric P matrix, PSP normalization, and P_max clipping. We test the same PSP time constants (20, 50, 100, 200 ms).

For both baselines, training uses the same precomputed target force trajectories and the same reservoir dynamics as the STDP method. Testing is performed closed-loop: the readout drives the vocal tract model, and correlation between achieved and target positions is measured.

---

## 3. Experiments

### 3.1 Main Comparison: STDP vs. Global Optimization

We compare STDP with physical guidance against offline ridge regression and online RLS-based FORCE learning. All methods use the same 1000-neuron reservoir, the same visual inputs, and the same target vocalization trajectories. All conditions use 10 random seeds.

For the global methods, we sweep PSP time constants (τ_psp = 20, 50, 100, 200 ms) to ensure fair comparison. For ridge regression, we also sweep regularization strength (α = 0.01–100) and report the best.

Performance is measured as Pearson correlation between network-generated vocal tract position and target trajectory, averaged across all three shapes.

### 3.2 Readout Mechanism Comparison

We compare three readout configurations for the STDP method:
- Membrane potential readout (original method)
- Spike-rate readout with 100-step window
- Spike-rate readout with 50-step window

This tests whether the method's success depends on access to analog membrane potentials or whether biologically realistic spike-rate decoding suffices.

### 3.3 Ablation Study

We systematically remove each plasticity mechanism from the full STDP model:
- Full: STDP + homeostasis
- No homeostasis: STDP only
- No STDP: homeostasis only (weights remain near initial values)

### 3.4 Additional Analyses

**Scaling:** Reservoir sizes of 100, 300, 500, 1000, 2000, 5000 neurons (3 seeds each).

**Generalization:** Train on 2 shapes, test on the held-out shape (10 seeds per condition).

**Parameter sensitivity:** STDP learning rate (0.001–0.1) and teaching current strength (0.5–5.0) (3 seeds each).

**Statistical validation:** 10 seeds for the main comparison, with mean, std, 95% CI, independent t-test, and Cohen's d.

---

## 4. Results

### 4.1 Main Comparison: Local vs. Global Methods

| Method | Type | Correlation (mean ± std) |
|---|---|---|
| **STDP + physical guidance (ours)** | **Local** | **0.865 ± 0.022** |
| Ridge regression, τ_psp = 20 ms | Global (offline) | 0.873 ± 0.010 |
| Ridge regression, τ_psp = 50 ms | Global (offline) | 0.855 ± 0.017 |
| Ridge regression, τ_psp = 100 ms | Global (offline) | 0.830 ± 0.026 |
| Ridge regression, τ_psp = 200 ms | Global (offline) | 0.792 ± 0.033 |
| RLS (FORCE), τ_psp = 20 ms | Global (online) | 0.875 ± 0.009 |
| RLS (FORCE), τ_psp = 50 ms | Global (online) | 0.857 ± 0.015 |
| RLS (FORCE), τ_psp = 100 ms | Global (online) | 0.829 ± 0.019 |
| RLS (FORCE), τ_psp = 200 ms | Global (online) | 0.784 ± 0.020 |
| No guidance (STDP only) | Local (control) | 0.066 ± 0.285 |
| Random weights | None (control) | 0.006 ± 0.313 |

STDP with physical guidance achieves 0.865 ± 0.022, statistically equivalent to both ridge regression (0.873 ± 0.010, t = −1.10, p = 0.288, Cohen's d = −0.52) and online RLS (0.875 ± 0.009, t = −1.30, p = 0.209, Cohen's d = −0.61). Neither comparison reaches significance, confirming that the local method matches the global methods in performance.

All methods share the same qualitative trend: performance peaks at τ_psp = 20 ms and degrades with larger time constants, consistent with the reservoir's fast dynamics (τ_m = 10 ms). The optimal PSP time constant matches the membrane time constant order, suggesting that the readout should track the reservoir's intrinsic time scale.

The "no guidance" condition (0.066) confirms that physical guidance is essential: STDP alone cannot learn the mapping without the teaching current to shape output neuron activity. The "random weights" condition (0.006) confirms that the reservoir-to-output mapping is not trivially solvable with random connectivity.

### 4.2 Readout Mechanism

| Readout scheme | Correlation (mean ± std) |
|---|---|
| Membrane potential | **0.865 ± 0.022** |
| Spike-rate (window = 100 steps) | 0.675 ± 0.304 |
| Spike-rate (window = 50 steps) | 0.166 ± 0.228 |

Membrane potential readout is both the highest-performing and the most stable (std = 0.022). Spike-rate readout with a 100-step window achieves a mean of 0.675 but with much higher variance (std = 0.304) — individual seeds range from −0.011 to 0.914. A shorter window (50 steps) degrades performance substantially, as the rate estimate becomes too noisy.

This confirms that the STDP learning mechanism does not fundamentally require access to analog membrane potentials — the network can learn using spike-rate readout alone. However, the stability and performance are better with membrane potential readout, which provides a smoother, more reliable control signal. The higher variance of spike-rate readout reflects the fundamental challenge of rate estimation from finite spike samples.

### 4.3 Ablation Study

| Configuration | Correlation |
|---|---|
| Full (STDP + Homeostasis) | 0.865 ± 0.022 |
| No homeostasis | 0.692 ± 0.101 |
| No STDP (homeostasis only) | −0.916 ± 0.000 |

STDP is the core learning mechanism: removing it results in strong negative correlation (−0.916), caused by the interaction between teaching current and output neuron dynamics without learned weight structure. Homeostatic plasticity contributes meaningfully to performance and stability: removing it reduces correlation from 0.865 to 0.692 and increases variance.

Eligibility traces had no measurable effect on this task (performance identical with and without them, across τ_elig values from 10 to 500 ms), consistent with the short sensorimotor delay in this setup.

### 4.4 Scaling Analysis

| Neurons | Correlation |
|---|---|
| 100 | 0.494 ± 0.046 |
| 300 | 0.715 ± 0.021 |
| 500 | 0.827 ± 0.041 |
| 1000 | 0.876 ± 0.010 |
| 2000 | 0.495 ± 0.278 |
| 5000 | 0.014 ± 0.747 |

Performance improves from 100 to 1000 neurons, then degrades sharply at larger sizes. The performance collapse above 1000 neurons reflects the fact that all parameters were held constant across scales — the spectral radius and learning rate are not self-tuning. This is a significant limitation: larger networks enter a regime where the STDP learning dynamics become unstable.

### 4.5 Generalization

| Held-out shape | Trained on | Held-out correlation |
|---|---|---|
| Triangle | Square + Circle | 0.772 |
| Circle | Square + Triangle | 0.821 |
| Square | Circle + Triangle | 0.763 |

The network generalizes to unseen shapes (0.763–0.821 correlation), indicating that STDP learns to extract visual features and map them to motor parameters rather than simply memorizing specific shape-trajectory pairs.

**Position and Size Generalization (Phase 16):** Training with fixed-shape stimuli and testing with position-shifted and size-scaled variants (10 seeds):

| Test type | Correlation | Success rate (>0.2) |
|---|---|---|
| Trained (fixed) | 0.863 ± 0.046 | 30/30 |
| Position shift | 0.794 ± 0.087 | 30/30 |
| Size scaling | 0.549 ± 0.245 | 27/30 |
| Rate variation | 0.283 ± 0.239 | 18/30 |

The network is robust to position shifts and shows partial scale invariance, suggesting emergent shape abstraction rather than pixel-level memorization.

### 4.6 Parameter Sensitivity

**STDP learning rate (A₊ = A₋):**

| Rate | Correlation |
|---|---|
| 0.001 | −0.916 |
| 0.005 | −0.867 |
| 0.01 | 0.876 |
| 0.05 | −0.338 |
| 0.1 | −0.368 |

Learning rate 0.01 is optimal. The method operates in a narrow regime: too little learning and the weights never escape their initial configuration; too much and the dynamics destabilize.

**Teaching current strength:**

| Strength | Correlation |
|---|---|
| 0.5 | −0.482 |
| 1.0 | −0.785 |
| 2.0 | 0.901 |
| 3.0 | 0.876 |
| 5.0 | 0.728 |

Optimal teaching current is 2.0–3.0. Below 2.0, the current is too weak to reliably drive output neurons. Above 3.0, the current overwhelms the reservoir signal, making STDP learn from the teaching input rather than the reservoir state.

### 4.7 Statistical Validation

| Shape | Mean ± Std | 95% CI |
|---|---|---|
| Square | 0.892 ± 0.030 | [0.869, 0.915] |
| Circle | 0.883 ± 0.019 | [0.869, 0.897] |
| Triangle | 0.819 ± 0.026 | [0.800, 0.839] |
| **Overall** | **0.865 ± 0.022** | **[0.848, 0.881]** |

All 10 seeds successfully learn the mapping. The narrow confidence intervals demonstrate high reproducibility.

---

## 5. Discussion

### 5.1 Local Rules Match Global Optimization

The central finding of this work is not that STDP outperforms global optimization methods — it does not. Ridge regression and RLS achieve statistically equivalent performance (0.873 and 0.875 vs. 0.865). Rather, the finding is that purely local plasticity rules can match the performance of global optimization on a chaotic spiking network, while accessing dramatically less information.

The information asymmetry is substantial. Ridge regression solves a batch optimization problem using the full 1000×1000 covariance matrix of reservoir states. RLS maintains and updates a 1000×1000 inverse covariance matrix online. In contrast, each STDP synapse accesses only the timing of its own pre- and post-synaptic spikes — a single scalar per synapse per timestep. Despite this information bottleneck, STDP achieves the same performance.

This result has two implications. First, for the reservoir computing community: the global covariance information used by RLS is not necessary for effective readout learning on this task — local spike timing contains sufficient information. Second, for computational neuroscience: it provides a computational demonstration that biological synapses, which lack access to global error signals, can still achieve effective sensorimotor learning when paired with appropriate guidance.

### 5.2 The Nature of Teacher-Forced Local Learning

The training protocol we call "physical guidance" is more precisely described as *teacher-forced local learning*. It occupies a distinct position among learning paradigms:

- It is **not reward-based learning**: no signal evaluates whether the output is "good" or "bad."
- It is **not gradient-based learning**: no error gradients are backpropagated through the network.
- It **does contain an error signal** in the teaching current: I_teach = gain × (target − output). This is a form of teacher forcing — the output neurons are externally clamped toward the target trajectory.

The key distinction from supervised learning is that the error drives *neuron activity*, not *weight updates*. Weight changes are determined entirely by local STDP. The teaching current sets up the "correct" firing pattern in the output layer, and STDP captures the correlation between reservoir activity and this guided output. The network does not discover the mapping through active exploration; it is guided through it, and local plasticity records the experience.

We acknowledge that this is a weaker form of learning than autonomous sensorimotor exploration. Real biological learning involves active trial-and-error, sensory prediction, and self-generated error correction. Our protocol is closer to passive replay with local plasticity. However, the contribution is precisely that this is sufficient: even when the error signal never touches synaptic weights — only drives neuron firing — local STDP can learn the mapping as effectively as global optimization. Whether the approach extends to self-generated exploration without teacher forcing is a key question for future work.

### 5.3 Readout Mechanism and Biological Plausibility

The membrane potential readout achieves the best performance but is less biologically plausible — real muscles respond to firing rates, not sub-threshold membrane potentials. Our spike-rate readout experiments show that the method works with rate-based decoding as well (0.675 mean), though with significantly lower stability: individual seeds range from −0.011 to 0.914, with approximately 1 in 10 seeds failing entirely.

This bimodal behavior — most seeds succeeding but a non-trivial fraction failing — reflects a fundamental challenge: spike generation is stochastic, and rate estimation from finite windows introduces noise that can destabilize the closed-loop dynamics. The membrane potential, being a continuous signal, provides a cleaner readout and is resilient to this noise. Both readout mechanisms share the same underlying learning (STDP on W_out), so the learning mechanism itself does not depend on analog membrane access — the difference is purely in how the output is decoded.

For biological plausibility, the spike-rate readout results should be interpreted as a proof of concept: the learning mechanism is compatible with pulse-rate decoding, but reliable biological readout likely requires additional mechanisms such as ensemble decoding (averaging across multiple output neurons) or downstream smoothing (e.g., Kalman filtering at the motor control level). The membrane potential readout serves as a best-case upper bound demonstrating the learning capacity of the STDP mechanism itself.

### 5.4 Limitations

Several limitations should be noted:

1. **Scalability.** Performance collapses above 1000 neurons without parameter retuning. The method currently relies on a carefully tuned operating point at the edge of chaos (spectral radius ≈ 1.5). Biological systems maintain stability across scale through mechanisms absent from our model (e.g., structural plasticity, neuromodulation, hierarchical organization). This work should be understood as a **proof of concept** demonstrating that local plasticity can match global optimization in a controlled setting — not as a general-purpose algorithm ready for large-scale deployment. Scaling to biologically realistic network sizes will likely require additional mechanisms such as structural sparsification or self-tuning spectral radius.

2. **Task complexity.** The current task involves only 3 shapes and 8 muscle parameters. Whether the approach scales to richer sensory inputs and more complex motor repertoires is an open question.

3. **Fixed reservoir.** Recurrent weights are fixed. Biological brains exhibit structural plasticity and recurrent plasticity that our model lacks.

4. **Teacher-forced local learning, not free exploration.** The teaching current directly drives output neurons — the network does not discover the correct motor pattern through active exploration but is guided through it. This is more accurately described as *teacher-forced local learning*: the error signal (target − output) clamps neuron activity to the desired trajectory, and STDP learns the local correlations between reservoir states and this guided output. In biological learning, guidance is typically more indirect (e.g., auditory feedback rather than direct current injection), and true sensorimotor learning involves active exploration, sensory prediction, and error correction that our model does not capture. The contribution is that even when the error is used only to drive neuron activity — never to compute weight updates — local STDP is sufficient to learn the mapping. Whether the method extends to more biologically realistic, self-generated exploration remains an open question.

5. **Spike-rate readout reliability.** While spike-rate readout achieves a mean of 0.675, approximately 1 in 10 seeds fails entirely (negative correlation), which is unacceptable for any engineering or biological application. Membrane potential readout serves as the reliable primary result; spike-rate readout demonstrates compatibility but requires further mechanisms (e.g., ensemble decoding, downstream filtering) for practical deployment.

6. **Performance parity, not superiority.** STDP matches but does not exceed global methods. The contribution lies in achieving equivalent performance with local information, not in outperforming global optimization.

### 5.5 Relationship to Existing Work

**FORCE learning (Sussillo & Abbott, 2009):** The original FORCE was demonstrated in rate-based chaotic networks. We show that when properly implemented with PSP filtering and numerical stabilization, FORCE achieves strong performance (0.875 ± 0.009) on our spiking task, statistically equivalent to STDP with physical guidance. The key distinction is not performance but the information requirements: FORCE uses a global inverse covariance matrix, while STDP uses only local spike timing.

**Reward-modulated STDP (Legenstein et al., 2010; Izhikevich, 2007):** These approaches use a global reward signal to gate STDP. We eliminate the reward signal entirely, replacing it with physical guidance of the output pattern.

**FOLLOW (Gilra & Gerstner, 2017):** FOLLOW uses a local learning rule but still requires an explicit error signal for weight updates. Our method uses no error signal for weight updates — error only drives neuron activity, and STDP operates locally on spike timing.

**Predictive coding / homeostasis (Friston, 2010):** There is a conceptual connection to predictive coding frameworks, where the brain minimizes prediction error. In our case, the teaching current sets up the "prediction" (target output), and STDP learns the weights that make the network's output match it.

---

## 6. Conclusion

We have shown that a chaotic spiking neural network can learn a cross-modal sensorimotor mapping using only STDP, homeostatic plasticity, and physical guidance — without reward signals, gradient-based weight optimization, or global error-based weight updates. The method achieves 0.865 ± 0.022 correlation across 10 seeds, statistically equivalent to both offline ridge regression (0.873 ± 0.010) and online RLS-based FORCE learning (0.875 ± 0.009).

This equivalence is the central result. Global optimization methods achieve the same performance using a 1000×1000 covariance matrix; STDP achieves it using only local spike timing at each synapse. This demonstrates that the information bottleneck of biological synapses — no access to global error signals — is not a fundamental barrier to effective sensorimotor learning in chaotic spiking networks.

Physical guidance provides a way to shape learning without directly controlling weights — the network experiences the correct pattern, and local plasticity captures it. This is consistent with how biological organisms learn sensorimotor skills: through guided experience, not through explicit error gradients.

While significant limitations remain in scalability and biological realism of the readout mechanism, these results support a broader principle: chaotic neural systems can acquire functional structure through guided experience and local plasticity alone, achieving performance comparable to global optimization. This principle may help explain how biological brains learn sensorimotor skills without global optimization, and points toward a path for building artificial neural systems that learn more like biological ones.

---

## References

1. Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns of activity from chaotic neural networks. *Neuron*, 63(4), 544-557.
2. Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.
3. Legenstein, R., Pecevski, D., & Maass, W. (2010). A learning theory for reward-modulated spike-timing-dependent plasticity with input adaptation. *Neural Computation*, 22(5), 1351-1368.
4. Gilra, A., & Gerstner, W. (2017). Predicting non-linear dynamics by stable local learning in a spiking neural network. *PLoS Computational Biology*, 13(6), e1005436.
5. Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine. *Biological Cybernetics*, 95(3), 259-273.
6. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
7. Der, R., & Martius, G. (2015). Novel plasticity rule makes it possible for spiking neural networks to acquire self-motion. *Proceedings of the National Academy of Sciences*, 112(45), E6224-E6232.
8. Warlaumont, A. S., Finnegan, M. K., & Buzan, D. (2012). Toward automated vocal development monitoring: A fuzzy-logic classifier of infant utterance types. *Journal of Speech, Language, and Hearing Research*, 55(5), 1423-1436.
9. Panda, P., & Roy, K. (2017). Learning to generate sequences with SNNs. *Frontiers in Neuroscience*, 11, 693.
10. Billings, G., & van Rossum, M. C. (2009). Memory retention and synaptic homeostasis. *Journal of Neurophysiology*, 102(3), 1650-1658.
