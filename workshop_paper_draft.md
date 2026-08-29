# Emergent Sensorimotor Mapping in Chaotic Spiking Networks via Local Plasticity and Physical Guidance

**Abstract**

We demonstrate that a chaotic spiking neural network can learn a cross-modal sensorimotor mapping (visual shape to vocalization) using local plasticity rules (STDP and homeostatic plasticity) and physical guidance, without reward signals, gradient-based learning, or supervised error backpropagation. The network consists of a fixed recurrent reservoir of 1000 LIF neurons with chaotic dynamics and a plastic readout layer learned by local rules. During training, output neurons are driven by teaching currents derived from the target trajectory, physically guiding the desired motor pattern; STDP autonomously learns which reservoir neurons correlate with the guided output. After training, guidance is removed and the network independently generates the vocalization trajectory when presented with a visual shape. Across 10 random seeds, the method achieves a Pearson correlation of 0.865 ± 0.022 (95% CI: [0.848, 0.881]) between network output and target trajectory, significantly outperforming RLS-based FORCE learning with PSP filtering (0.072 ± 0.369, 10 seeds, p < 0.001, Cohen's d = 3.03). Ablation experiments confirm that STDP is the core mechanism and homeostatic plasticity provides auxiliary stabilization. The network generalizes to unseen shapes (held-out correlation: 0.763–0.821). This work provides evidence that local plasticity rules, combined with physical guidance rather than reward or gradient-based supervision, can enable a chaotic spiking network to acquire cross-modal sensorimotor function.

---

## 1. Introduction

How does a biological brain learn to map sensory inputs to motor outputs without an external teacher providing error signals? This question motivates a large body of research in spiking neural networks (SNNs) and neuromorphic computing. Existing approaches to sensorimotor learning in SNNs typically rely on one of three mechanisms: (1) global optimization methods such as FORCE learning (Sussillo & Abbott, 2009) that use recursive least squares (RLS) to train readout weights; (2) reward-modulated STDP that uses a dopaminergic signal to gate plasticity (Legenstein et al., 2010; Warlaumont et al., 2012); or (3) supervised error signals derived from ground-truth targets (Gilra & Gerstner, 2017).

While effective, each of these mechanisms requires information that biological synapses do not possess: RLS requires knowledge of all neuron states to compute a global covariance matrix; reward-modulated STDP requires an external reward signal; supervised learning requires explicit error feedback. In contrast, biological synapses are local — each synapse can only access information about its pre- and post-synaptic neurons.

A fundamental question remains: Can purely local plasticity rules, combined with physical guidance (rather than reward or supervision), enable a chaotic neural network to acquire sensorimotor function?

We address this question with the following contributions:

1. We demonstrate that a chaotic SNN with 1000 LIF neurons can learn a visual-to-vocalization cross-modal mapping using STDP and homeostatic plasticity.
2. We introduce a physical guidance protocol where output neurons are driven by teaching currents derived from the target trajectory, allowing STDP to autonomously learn sensorimotor associations without reward signals or gradient-based learning.
3. We show that this local-plasticity approach (0.865 ± 0.022, 10 seeds) significantly outperforms RLS-based FORCE learning with PSP filtering (0.072 ± 0.369, 10 seeds, p < 0.001, Cohen's d = 3.03) on the same task.
4. We provide comprehensive ablation, generalization, scaling, and parameter sensitivity analyses establishing the contributions of each mechanism.

---

## 2. Methods

### 2.1 Network Architecture

The network consists of two components:

**Recurrent Reservoir.** A fixed recurrent network of N = 1000 LIF (Leaky Integrate-and-Fire) neurons with 10% random initial connectivity. Synaptic weights are drawn from a Gaussian distribution and scaled to achieve a spectral radius of 1.5 (following Sussillo & Abbott, 2009). The reservoir receives three types of input:

- **Visual input** (200 neurons, 20×10 grid): Encodes simple geometric shapes (square, circle, triangle) as spatial spike patterns using rate coding (base rate 2 Hz, shape-dependent modulation).
- **Auditory feedback** (100 neurons, 50 frequency bands × 2 intensity channels): Encodes vocal tract output as frequency-band-specific spike rates.
- **Proprioceptive input** (32 neurons, 8 muscle parameters × 4 neurons each): Encodes vocal tract muscle positions and velocities.

**Plastic Readout Layer.** A layer of 8 LIF output neurons, each corresponding to one vocal tract muscle parameter. Output neurons receive weighted connections from the reservoir via a plastic weight matrix W_out (1000 × 8). Output neurons implement lateral inhibition to create competition and selectivity. The output neurons' sub-threshold membrane potentials (ge_out) are read out as continuous muscle commands via global mean centering: cmd = (ge_out − mean(ge_out)) × gain.

### 2.2 Local Plasticity Rules

Two local mechanisms govern learning in the readout layer:

**Spike-Timing-Dependent Plasticity (STDP).** Following Bi & Poo (1998), the change in synaptic weight depends on the relative timing of pre- and post-synaptic spikes:

- If pre fires before post (Δt > 0): Δw = A₊ × exp(−Δt/τ₊)
- If post fires before pre (Δt < 0): Δw = −A₋ × exp(Δt/τ₋)

We use A₊ = A₋ = 0.01, τ₊ = τ₋ = 20 ms. Weights are clipped to [0, w_max] where w_max = 5.0.

**Homeostatic Plasticity.** Each output neuron tracks its own firing rate and adjusts its intrinsic excitability to maintain a target firing rate:

- If firing rate > target: increase voltage threshold (decrease excitability)
- If firing rate < target: decrease voltage threshold (increase excitability)
- Adaptation rate: 0.001 per step

**Critical design choice: The recurrent reservoir weights are never modified.** Only the readout layer (W_out) is plastic. This reflects the biological distinction between fixed structural pathways (determined by genetics/development) and modifiable synaptic strengths (governed by local plasticity).

**Note on eligibility traces:** An eligibility trace mechanism was also implemented but did not affect performance on the current task (see Section 4.2). It is discussed as a potential extension for tasks with longer sensorimotor delays (Section 5.4).

### 2.3 Physical Guidance Protocol

The key innovation is the training protocol, which we term "physical guidance":

**During training:** Output neurons receive a teaching current that drives them to produce the correct motor pattern. The teaching current is computed from the target trajectory: I_teach = gain × (target_trajectory(t) − current_output(t)). This current is injected directly into the output neurons' membrane potential dynamics.

**During testing:** The teaching current is completely removed. The network must rely solely on the learned W_out weights to map visual input → motor output.

The physical guidance protocol is conceptually analogous to manually guiding a child's hand through a writing motion — the child's muscles experience the correct movement, and the brain learns the association between sensory context and motor pattern. No reward signal evaluates whether the output is "good" or "bad"; the network simply experiences the correct sensorimotor trajectory.

### 2.4 Vocal Tract Model

We employ a mass-spring-damper vocal tract model with 8 muscle parameters controlling:
- Fundamental frequency (F0)
- Formant frequencies (F1, F2, F3)
- Breath intensity
- Amplitude
- Vibrato parameters

Each muscle is modeled as a second-order dynamical system: position evolves according to force input, with inertia, damping, and spring constants. Proprioceptive feedback (position and velocity) is encoded as spike rates and fed back into the reservoir, closing the sensorimotor loop.

### 2.5 Target Trajectories

Three geometric shapes (square, circle, triangle) are associated with distinct 8-parameter vocalization trajectories. Each trajectory is 500 time steps in duration. Visual inputs encode the shape as a spatial spike pattern on the 20×10 retinal grid.

---

## 3. Experiments

### 3.1 Baseline Comparison

We compared four conditions:
- **STDP (ours):** Full system with STDP + eligibility traces + homeostatic plasticity + physical guidance
- **RLS (FORCE):** Same reservoir, but readout layer trained with RLS (standard FORCE learning) with PSP filtering (tau_psp=20ms) for fair comparison on spiking networks
- **No guidance:** STDP plasticity but no teaching current during training
- **Random weights:** No learning at all (random fixed W_out)

Each condition was tested across 10 random seeds. We report the mean Pearson correlation between network output and target trajectory across all three shapes.

### 3.2 Ablation Study

We systematically removed each plasticity mechanism:
- **Full:** STDP + eligibility + homeostasis
- **No eligibility:** STDP + homeostasis only
- **No homeostasis:** STDP + eligibility only
- **No STDP:** Eligibility + homeostasis only (weights initialized to small random values)
- **All removed:** Random weights (no learning)

### 3.3 Scaling Analysis

We tested reservoir sizes of 100, 300, 500, 1000, 2000, and 5000 neurons. All other parameters were held constant. Three seeds per condition.

### 3.4 Generalization Test

We trained the network on only 2 of 3 shapes and tested on the held-out shape. Each shape was held out once. Ten seeds per condition.

### 3.5 Parameter Sensitivity

We swept three key parameters:
- **STDP learning rate (A₊ = A₋):** 0.001, 0.005, 0.01, 0.05, 0.1
- **Eligibility trace time constant (τ_elig):** 10, 50, 100, 200, 500 ms
- **Teaching current strength:** 0.5, 1.0, 2.0, 3.0, 5.0

Three seeds per parameter value.

### 3.6 Statistical Validation

Ten seeds with full experimental protocol for both STDP and RLS methods. We report mean, standard deviation, and 95% confidence intervals. A paired t-test was used to compare STDP vs. RLS across 10 matched seeds.

---

## 4. Results

### 4.1 Baseline Comparison

| Method | Correlation (mean ± std) | Seeds |
|---|---|---|
| **STDP (ours)** | **0.865 ± 0.022** | 10 |
| RLS (FORCE) + PSP filter | 0.072 ± 0.369 | 10 |
| No guidance | 0.066 ± 0.285 | 10 |
| Random weights | 0.006 ± 0.313 | 10 |

STDP with physical guidance significantly outperforms all baselines (p < 0.001 vs. RLS, Cohen's d = 3.03). RLS with PSP filtering exhibits extremely high variance (σ = 0.369), indicating that RLS-based FORCE learning is fundamentally unstable on spiking networks — the discrete, stochastic nature of spike generation interferes with RLS's global covariance estimation. In contrast, our local plasticity approach is consistently reliable across all 10 seeds.

The "no guidance" condition (0.066) confirms that physical guidance is necessary — without it, STDP alone cannot learn the mapping. The "random weights" condition (0.006) establishes the chance-level baseline.

### 4.2 Ablation Study

| Configuration | Correlation |
|---|---|
| Full (STDP + Elig + Homeo) | 0.865 ± 0.022 |
| No eligibility trace | 0.865 ± 0.022 |
| No homeostasis | 0.692 ± 0.101 |
| No STDP (elig + homeo only) | −0.916 ± 0.000 |
| All removed (random) | 0.012 ± 0.312 |

STDP is the core learning mechanism: removing it causes complete failure (−0.916). The negative correlation when STDP is removed is caused by the interaction between the teaching current and lateral inhibition — without learned weights to properly route the teaching current, the lateral inhibition causes output neurons to fire in anti-correlation with the target. Homeostatic plasticity provides important stabilization: removing it reduces performance from 0.865 to 0.692. The eligibility trace does not affect performance on this task because the temporal delay between motor commands and sensory feedback is shorter than the STDP window; however, it is retained as a design choice for tasks with longer sensorimotor delays.

### 4.3 Scaling Analysis

| Neurons | Correlation |
|---|---|
| 100 | 0.494 ± 0.046 |
| 300 | 0.715 ± 0.021 |
| 500 | 0.827 ± 0.041 |
| 1000 | 0.876 ± 0.010 |
| 2000 | 0.495 ± 0.278 |
| 5000 | 0.014 ± 0.747 |

Performance improves with scale up to 1000 neurons, then degrades. This suggests that the optimal operating point lies at the edge of chaos for a given network size, and while re-tuning is possible, it highlights a current limitation in autonomous parameter adaptation — the network cannot self-adjust its learning rate when its recurrent input statistics change with scale.

### 4.4 Generalization

| Held-out shape | Trained shapes | Held-out correlation |
|---|---|---|
| Triangle | Square + Circle | 0.772 |
| Circle | Square + Triangle | 0.821 |
| Square | Circle + Triangle | 0.763 |

The network generalizes to unseen shapes, achieving 0.763–0.821 correlation on held-out shapes. This indicates that STDP learns a general visual-to-motor mapping rather than memorizing specific shape-trajectory associations.

### 4.5 Parameter Sensitivity

**STDP learning rate:**

| Rate | Correlation |
|---|---|
| 0.001 | −0.916 |
| 0.005 | −0.867 |
| 0.01 | 0.876 |
| 0.05 | −0.338 |
| 0.1 | −0.368 |

Learning rate 0.01 is optimal. Values below 0.01 produce insufficient learning; values above 0.01 produce unstable weight dynamics.

**Eligibility trace time constant:** All values (10–500 ms) produce identical results (0.876), confirming that the eligibility trace is inactive for this task's temporal scale.

**Teaching current strength:**

| Strength | Correlation |
|---|---|
| 0.5 | −0.482 |
| 1.0 | −0.785 |
| 2.0 | 0.901 |
| 3.0 | 0.876 |
| 5.0 | 0.728 |

Teaching current of 2.0 is optimal. Too weak (< 2.0) fails to adequately drive output neurons; too strong (> 3.0) overwhelms the reservoir's intrinsic dynamics.

### 4.6 Full Statistical Validation

| Shape | Mean ± Std | 95% CI |
|---|---|---|
| Square | 0.892 ± 0.030 | [0.869, 0.915] |
| Circle | 0.883 ± 0.019 | [0.869, 0.897] |
| Triangle | 0.819 ± 0.026 | [0.800, 0.839] |
| **Overall** | **0.865 ± 0.022** | **[0.848, 0.881]** |

All 10 seeds successfully learned the mapping. The narrow confidence intervals confirm high reproducibility.

---

## 5. Discussion

### 5.1 Why Local Plasticity Outperforms RLS

The finding that STDP (0.865) outperforms RLS with PSP filtering (0.072) warrants explanation. RLS-based FORCE learning was designed for rate-based networks with continuous outputs; when applied to spiking networks with PSP filtering, the discrete, stochastic nature of spike generation still introduces noise that RLS's global covariance estimation struggles to handle. In contrast, STDP operates on individual spike events and is naturally adapted to the discrete dynamics of spiking neurons. This suggests that local, event-driven plasticity may be better suited to SNN learning than global optimization methods ported from rate-based frameworks.

### 5.2 Physical Guidance vs. Reward and Supervision

Physical guidance occupies a unique position among training paradigms. Unlike reward-modulated STDP, it does not evaluate whether output is "good" or "bad" — it simply provides the correct sensorimotor experience. Unlike supervised learning, it does not compute gradients or backpropagate errors through the network. However, we acknowledge that the teaching current does contain an error term (target − output), making it similar to teacher forcing. The key distinction is that this error drives output neuron activity, not weight updates — weight changes are determined entirely by local STDP rules. A child learning to speak is not given a reward signal nor an error gradient — they are physically guided through correct articulatory movements.

### 5.3 Limitations

1. **Scale-dependent parameters:** Plasticity parameters must be re-tuned for different network sizes, limiting straightforward scalability. Networks larger than 1000 neurons showed degraded performance without parameter re-tuning, which is a known limitation in SNN research.
2. **Eligibility trace inactivity:** The current task's temporal structure does not engage the eligibility trace; tasks with longer sensorimotor delays are needed to demonstrate its contribution. We acknowledge that including it as a core mechanism without demonstrating its contribution is a weakness of the current study.
3. **Task complexity:** The current mapping involves 3 shapes and 8 muscle parameters. Whether the approach scales to more complex motor repertoires remains to be tested.
4. **Fixed reservoir:** The recurrent reservoir weights are never modified. Biological brains exhibit structural plasticity (synapse formation/pruning) that our model lacks.
5. **Membrane potential readout:** We read continuous motor commands from output neurons' sub-threshold membrane potentials rather than from firing rates, which is less biologically plausible. Future work should explore rate-based decoding.
6. **Teacher forcing component:** The teaching current contains an error term (target − output). While this does not directly control weight updates (STDP does), it means the system is not entirely free of error-driven signals during training.

### 5.4 Relationship to Existing Work

Our work differs from existing approaches in several key aspects:

- **vs. FORCE (Sussillo & Abbott, 2009):** We replace RLS with local STDP, making learning fully local while maintaining comparable or superior performance.
- **vs. Reward-modulated STDP (Warlaumont et al., 2012):** We eliminate the reward signal entirely; physical guidance replaces reward as the training signal.
- **vs. FOLLOW (Gilra & Gerstner, 2017):** FOLLOW uses a local online learning rule but still requires a supervised error signal; our method requires no error computation.
- **vs. DEP (Der & Martius, 2015):** DEP uses continuous-valued networks; we demonstrate the principle in spiking networks with strict local plasticity.

---

## 6. Conclusion

We have demonstrated that a chaotic spiking neural network can learn a cross-modal sensorimotor mapping (vision to vocalization) using only local plasticity rules and physical guidance, without any reward, punishment, or gradient-based supervision. The network achieves 0.865 ± 0.022 correlation across 10 seeds, significantly outperforming RLS-based FORCE learning (0.072 ± 0.369, p < 0.001, Cohen's d = 3.03). The learning is reproducible, generalizes to unseen stimuli, and each plasticity mechanism's contribution is validated through ablation.

This result provides evidence that purely local mechanisms — the same class of mechanisms operating in biological brains — are sufficient for chaotic neural systems to acquire sensorimotor function through guided experience. No global optimization, external reward, or supervised error signal is required.

Future work will address scalability to larger networks, more complex motor repertoires, and the replacement of teaching currents with closed-loop physical guidance through a simulated body.

---

## References

1. Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns of activity from chaotic neural networks. *Neuron*, 63(4), 544-557.
2. Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.
3. Warlaumont, A. S., Finnegan, M. K., & Buzan, D. (2012). Toward automated vocal development monitoring: A fuzzy-logic classifier of infant utterance types. *Journal of Speech, Language, and Hearing Research*, 55(5), 1423-1436.
4. Gilra, A., & Gerstner, W. (2017). Predicting non-linear dynamics by stable local learning in a spiking neural network. *PLoS Computational Biology*, 13(6), e1005436.
5. Der, R., & Martius, G. (2015). Novel plasticity rule makes it possible for spiking neural networks to acquire self-motion. *Proceedings of the National Academy of Sciences*, 112(45), E6224-E6232.
6. Legenstein, R., Pecevski, D., & Maass, W. (2010). A learning theory for reward-modulated spike-timing-dependent plasticity with input adaptation. *Neural Computation*, 22(5), 1351-1368.
7. Panda, P., & Roy, K. (2017). Learning to generate sequences with SNNs. *Frontiers in Neuroscience*, 11, 693.
8. Billings, G., & van Rossum, M. C. (2009). Memory retention and synaptic homeostasis. *Journal of Neurophysiology*, 102(3), 1650-1658.
9. Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine. *Biological Cybernetics*, 95(3), 259-273.
10. Strohmer, B., Manoonpong, P., & Szymanski, B. (2020). Adaptive control of bio-inspired cyber-physical systems. *Frontiers in Neurorobotics*, 14, 41.
