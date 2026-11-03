# Where every number on the poster comes from

Poster: *When a correlation metric cannot tell learning from non-learning in a spiking sensorimotor network* (SNUFA 2026).

Each row of the poster maps to a file in this repository. Values are the exact numbers written by the scripts.

| On the poster | Value | Where |
|---|---|---|
| “lazy” network (same output for every shape), old Pearson score | **0.902** | `reanalysis/lazy_network.json` → `lazy.raw_diag` = 0.90196 |
| “lazy” network, rank-1 | **33%** (= chance) | `reanalysis/lazy_network.json` → `lazy.cen_rank1` = 1 of 3 |
| STDP readout, old Pearson score | 0.865 | `reanalysis/results.json` → `stdp_original.raw_diag` = 0.86480 |
| ridge baseline, old Pearson score | 0.873 | `reanalysis/results.json` → `ridge.raw_diag` = 0.87347 |
| STDP readout, rank-1 | **40%** | `reanalysis/results.json` → `stdp_original.cen_rank1_pct` = 0.400 |
| STDP readout + PSP-filtered input, rank-1 | **47%** | `reanalysis/SWEEP_REPORT.md` (10-seed revalidation of the sweep best) |
| ridge baseline, rank-1 | **77%** | `reanalysis/results.json` → `ridge.cen_rank1_pct` = 0.767 |
| ridge on the **same instantaneous spikes** STDP sees | **100%** | `reanalysis/control_ridge_spikes.json` → `cen_rank1_pct` = 1.0 |
| non-negative ridge (NNLS) on the same spikes | **100%** | `reanalysis/control_ridge_nonneg.json` → `cen_rank1_pct` = 1.0 |
| old score vs. true shape competence, 20 configurations | **r = −0.016** | `reanalysis/SWEEP_REPORT.md` |
| three target trajectories mutually correlated | **0.836** | `reanalysis/results.json` → `target_similarity.matrix` (off-diagonal mean) |
| the network's three outputs mutually correlated | **0.992** | `reanalysis/results.json` → `stdp_original.output_output` (off-diagonal mean) |
| shape decodable: visual input / reservoir / readout / motor output | **100 / 100 / 84 / 87 %** | `reanalysis/results.json` → `decodability.{visual_input,reservoir_fired,readout_output,vocal_position_traj}.mean` |

**True shape competence** (used in the r = −0.016 row) = does the output correlate with its *own* target more than with the other two; one correct out of three candidates = 33% = chance.

## Reproduce

```bash
pip install -r requirements.txt

python -B reanalysis/reanalyze.py          # main re-evaluation (~4 min)
python -B reanalysis/lazy_network.py       # the 0.902 lazy-network control
python -B reanalysis/control_ridge_spikes.py   # ridge on instantaneous spikes (100%)
python -B reanalysis/control_ridge_nonneg.py   # non-negative ridge / NNLS (100%)
python -B reanalysis/sweep_psp.py          # PSP fairness + parameter sweep (~8 min)
```

Each script prints its results and writes a `.json` next to itself.

## Honest note on the figure

`make_fig5.py` (the script that draws the lazy/STDP/ridge comparison) hard-codes its bar heights
rather than reading the JSON at plot time. The four values it prints — 0.902, 0.865, 0.873, and
rank-1 33 / 40 / 77 — are exactly the values in the table above; the JSON files are the source of truth.

---

## 中文速查（同一张表）

- 0.902 = 偷懒网络旧分数 → `reanalysis/lazy_network.json`
- 33% / 40% / 77% / 100% = 四种 rank-1 → `results.json` + `control_ridge_spikes.json` + `control_ridge_nonneg.json`
- r = −0.016 = 20 组参数 → `SWEEP_REPORT.md`
- 100 / 100 / 84 / 87 = 逐层可解码 → `results.json` → `decodability`
- 0.836 / 0.992 = 目标互相关 / 输出互相关 → `results.json`
