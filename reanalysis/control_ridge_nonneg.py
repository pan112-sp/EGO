# -*- coding: utf-8 -*-
# 验证原话: 无（探索性 · 拆解"非负读出"这个混淆）
"""对照：岭回归 + 非负约束（NNLS），读与 STDP 相同的瞬时脉冲。
目的：把 40% vs 77% 差距里的「学习规则」和「只能兴奋」拆开。
  若 NNLS 也掉到 ~40% → 非负约束是主因；
  若 NNLS 仍 ~100%   → 学习规则（局部/在线）才是主因。"""
import os
import sys
import json
import time
import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)

import reanalyze as A
import rls_fixed as RLS

NSEED = 10
TEST_STEPS = 500
TSP = 3000


def collect_spikes(reservoir, vis_enc, shape_id, n_steps):
    spikes = np.zeros((n_steps, reservoir.n), dtype=np.float32)
    for t in range(n_steps):
        vis = vis_enc.encode(shape_id, rate=50.0)
        spikes[t] = reservoir.step(vis)
    return spikes


def main():
    acc = A.new_acc()
    targets_pos = [RLS.generate_target_trajectory(s, 1000) for s in range(3)]
    for seed in range(NSEED):
        t0 = time.time()
        vis_enc = RLS.VisualEncoder(seed=seed)
        reservoir = RLS.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = RLS.VocalTract(seed=seed + 200)
        targets_force = [RLS.precompute_target_forces(targets_pos[s]) for s in range(3)]

        all_spk, all_t = [], []
        for s in [0, 1, 2]:
            sp = collect_spikes(reservoir, vis_enc, s, TSP)
            ft = targets_force[s]
            nr = TSP // len(ft) + 1
            all_spk.append(sp)
            all_t.append(np.tile(ft, (nr, 1))[:TSP].astype(np.float64))
        X = np.vstack(all_spk).astype(np.float64)
        Y = np.vstack(all_t)

        # 非负最小二乘（每列独立求解）
        w_out = np.zeros((Y.shape[1], X.shape[1]))
        for j in range(Y.shape[1]):
            w_out[j], _ = nnls(X, Y[:, j])

        # 闭环测试（与 control_ridge_spikes 完全一致）
        reservoir_test = RLS.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        outs = []
        for s in [0, 1, 2]:
            vt.reset()
            pi = []
            for _ in range(TEST_STEPS):
                vis = vis_enc.encode(s, rate=50.0)
                fired = reservoir_test.step(vis)
                out = w_out @ fired.astype(np.float64)
                force = out * 3.0
                np.clip(force, -20.0, 20.0, out=force)
                vt.step(force.astype(np.float32))
                pi.append(vt.position.copy())
            outs.append(np.array(pi))

        res = A.analyze(outs, [t[:TEST_STEPS].copy() for t in targets_pos])
        A.merge(acc, res)
        print("seed %2d  raw_diag=%.3f  raw_rank1=%d/3  cen_rank1=%d/3  (%.0fs)" % (
            seed, res["raw_diag"], res["raw_rank1"], res["cen_rank1"], time.time() - t0),
            flush=True)

    res = A.finalize(acc)
    oo = (res["output_output"][0][1] + res["output_output"][0][2] +
          res["output_output"][1][2]) / 3
    print("")
    print("=== 岭回归 + 非负约束 (NNLS) ===")
    print("raw_diag  = %.4f" % res["raw_diag"])
    print("raw_rank1 = %d/30 = %.0f%%" % (res["raw_rank1"], 100 * res["raw_rank1_pct"]))
    print("cen_rank1 = %d/30 = %.0f%%" % (res["cen_rank1"], 100 * res["cen_rank1_pct"]))
    print("output similarity = %.3f" % oo)
    with open(os.path.join(HERE, "control_ridge_nonneg.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved -> control_ridge_nonneg.json")


if __name__ == "__main__":
    main()
