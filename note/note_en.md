# A Cautionary Note on Evaluating Sensorimotor Learning in Spiking Networks

**— a negative result, and a more reliable yardstick**

*Pan Haimeng · September 2026*

---

## Abstract

I trained a spiking network (fixed reservoir + plastic STDP readout) to learn a cross-modal
mapping from visual shape to vocal-tract motor trajectories. Under the evaluation used in the
project, my method was statistically indistinguishable from an offline ridge-regression baseline
(0.865 ± 0.022 vs. 0.873 ± 0.010, p = 0.29). I then examined the evaluation itself and found that,
on this task, it has **zero correlation with actual task competence**: across 20 hyperparameter
configurations, the metric correlated with the ability to correctly discriminate shapes at
**r = −0.016**. Re-evaluated with a discriminating metric, the two methods differ substantially
(local STDP readout 47%, global ridge readout 77%, chance 33%).

I report this negative result and offer one concrete evaluation recommendation.

---

## 1. Task and setup

Network (**fixed reservoir + plastic readout**):

- 1000 LIF neurons, sparse recurrent connectivity (10%), spectral radius g = 5.0
  (supercritical, chaotic)
- Input: a 20×10 visual grid (distinct binary patterns for three shapes), Poisson spike encoding
- Readout: 8 output neurons corresponding to 8 vocal-tract parameters; learning rule is
  STDP plus synaptic homeostasis
- The readout is trained by **teacher forcing**: a teaching current proportional to the target
  trajectory is injected into the output neurons during training

Three targets — square, circle, triangle — each a 1000-step, 8-dimensional parameter trajectory.

**Baseline:** the same reservoir states, with an offline ridge-regression readout (on PSP-filtered
states; α selected from {0.01 … 100}).

The project originally reported: STDP readout **0.865 ± 0.022**, ridge **0.873 ± 0.010**,
difference not significant (p = 0.29), concluding that "a local plasticity rule reaches the
performance of a global linear readout."

---

## 2. Finding 1: the evaluation loses discriminative power on this task

The original evaluation was:

```
score = pearson_corr( flatten(output trajectory), flatten(target trajectory) )
```

The problem lies in how the targets were constructed. The three target trajectories are
**mutually correlated**:

| | square | circle | triangle |
|---|---|---|---|
| square | 1.000 | 0.883 | 0.823 |
| circle | 0.883 | 1.000 | 0.802 |
| triangle | 0.823 | 0.802 | 1.000 |

**Mean off-diagonal correlation = 0.836.** In other words, a network that merely emits
*some* target-like signal already correlates above 0.8 with all three targets.

At the same time, the STDP network's outputs across the three shapes were mutually correlated
at **0.992** — essentially identical.

Together, these two facts mean the score lands near 0.86 **regardless of whether the network
actually discriminates the shapes**. The metric measures "does the output resemble this family
of target signals," not "is the output matched to the correct target."

---

## 3. Finding 2: the old metric has zero correlation with real competence

Across 20 hyperparameter configurations (input scale × STDP learning rate), I recorded:

- the **old metric score** (mean diagonal), and
- **true shape competence**: for each shape, whether the output's correlation with its *own*
  target ranked highest among the three (chance = 33%).

Result:

```
corr( old metric score, true shape competence ) = -0.016
```

| | old metric score | true competence |
|---|---|---|
| configuration with the **highest** old-metric score | +0.415 | **33% (= chance)** |
| configuration with the **lowest** old-metric score | −0.891 | **56% (= best in sweep)** |

**The configuration the old metric liked best was at chance; the one it liked least was the
best in the sweep.**

The metric is therefore not merely weak — it has **no monotonic relationship** with task
competence. It largely reflects whether the overall shape and sign of the output happen to
agree with the target, not whether the shape correspondence is correct.

---

## 4. Finding 3: with a discriminating metric, the gap appears

I replaced the evaluation with two discriminating measures:

1. a **3×3 confusion matrix** — each shape is presented, correlated against all three targets,
   and we ask whether the diagonal dominates; and
2. **rank-1 accuracy** — the fraction of cases where the correct target ranks highest
   (chance = 33%).

| Method | old metric score | old metric rank-1 | **new metric rank-1** | mutual output similarity |
|---|---|---|---|---|
| STDP readout (as-is) | 0.865 | 27% | **40%** | **0.992** |
| Ridge (baseline) | 0.873 | 67% | **77%** | **0.822** |
| chance | — | 33% | 33% | — |

**Ridge does learn shape discrimination** (77%, far above chance), and its three outputs are
clearly distinct (0.822). **The STDP readout does not** (40%, near chance), and its three
outputs are nearly identical (0.992).

Broken down by shape (10 seeds): the STDP readout is correct 8/10 times on the triangle,
but 0/10 on the square and 2/10 on the circle.

---

## 5. Finding 4: an implementation-level confound

Inspecting the code, I found that the two methods were not given the **same input
representation**:

| Readout | Synaptic input |
|---|---|
| STDP readout | `w_out @ instantaneous spikes` — a 0/1 vector for a single time step |
| Ridge / RLS baseline | `w_out @ PSP-filtered firing rates` — integrated over τ = 20 ms |

Shape information lives in the **firing rate**, whereas the STDP readout consumed single-step
instantaneous spikes. **This asymmetry disfavours STDP.**

I therefore gave the STDP readout the same PSP filtering and re-swept parameters in two
dimensions (input scale × learning rate, 20 configurations), selecting the best configuration
with the new metric:

| | old metric rank-1 | new metric rank-1 | output similarity |
|---|---|---|---|
| STDP (no PSP) | 27% | 40% | 0.992 |
| STDP + PSP (sweep best) | 37% | **47%** | 0.986 |
| Ridge | 67% | **77%** | 0.822 |

**After aligning the input representation and re-tuning, STDP improves from 40% to 47%, but
remains well below ridge at 77%, and its outputs are still nearly identical (0.986).**

---

## 6. Additional observation: shape information is not lost

Importantly, this is **not** a case of "the network learned nothing." A simple linear classifier
decoding "which shape was just shown" from each layer's activity gives:

| Layer | decoding accuracy |
|---|---|
| visual input | 100% |
| reservoir activity | 100% |
| readout output | 84% |
| final motor output (vocal-tract position) | 87% |

Shape information **survives all the way to the final output**. The problem is not the absence
of shape information but the **weak correspondence** between the output and the intended target:
the network produces an output that varies with shape, yet is only weakly matched to the
correct target.

---

## 7. Conclusions and recommendation

**Two conclusions, both negative/methodological:**

1. On this task, the widely used "output–target correlation" has **zero correlation with actual
   task competence** (20 configurations, r = −0.016). Reporting "parity" between methods with
   this metric is unreliable.
2. After aligning input representations and limited re-tuning, the local STDP readout (47%)
   remains well below the global ridge readout (77%), with chance at 33%.

**Recommendation:** when targets/classes are themselves strongly correlated, report a
**confusion matrix or a rank-based measure**, not the correlation with a single target.
The former distinguishes "learned" from "did not learn"; the latter does not.

---

## 8. Limitations

These must be stated alongside the results:

- **One task, one architecture.** Three shapes, 1000 neurons, 1000-step trajectories, and a
  simplified mass–spring–damper vocal-tract model.
- **The parameter sweep covers only two dimensions** (input scale × learning rate); teaching
  current, PSP time constant, and network size were held fixed. **Whether configurations exist
  outside the searched region that would close the gap is unknown.**
- **I do not claim that local rules are fundamentally incapable.** I report only that, in this
  setting and within this search, they did not succeed.
- The sweep used 3 seeds and is therefore noisy (the 3-seed best was 56%, which fell to 47%
  under 10-seed revalidation); the choice of "best configuration" is itself unstable.

---

## 9. Reproducibility

All code, data, and re-evaluation scripts are public. Every number above can be regenerated by
one script (about 8 minutes):

- re-evaluation (contrastive metrics, layer-wise decoding): `reanalysis/reanalyze.py`
- parameter sweep: `reanalysis/sweep_psp.py`
- reports: `reanalysis/REPORT.md`, `reanalysis/SWEEP_REPORT.md`

Repository: `github.com/pan112-sp/EGO`

---

## One-sentence version

> I trained a spiking network on a shape→vocal-tract mapping and originally reported 0.865,
> on par with a ridge baseline; I then found that this evaluation has zero correlation with
> actual task competence (20 configurations, r = −0.016), and under a discriminating metric the
> gap appears: local STDP readout 47%, global ridge 77%, chance 33%.
