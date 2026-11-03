# -*- coding: utf-8 -*-
# 验证原话: 无（探索性 · 对称混淆对照：岭回归喂瞬时脉冲）
"""
对照实验：岭回归读「瞬时脉冲」而非「PSP 滤波后的放电率」
========================================================
摘要的 confound 段说：STDP 读瞬时脉冲得 40~47%，岭回归读 PSP 得 77%。
这里做反方向对照：给岭回归和 STDP 一样的瞬时脉冲输入，看它是否也掉到 STDP 水平。

若岭回归+瞬时脉冲 → 明显低于 77%（靠近 40~47%）：
  输入表示是主因，不是学习规则。
若岭回归+瞬时脉冲 → 仍接近 77%：
  说明 STDP 的学习规则才是瓶颈。

用法:  python control_ridge_spikes.py
输出:  control_ridge_spikes.json
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)          # D:\EGO\一期结果
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)

import reanalyze as A                    # analyze / merge / new_acc / finalize / cc / NSEED / TEST_STEPS
import rls_fixed as RLS

NSEED = A.NSEED
TEST_STEPS = A.TEST_STEPS


def collect_reservoir_spikes(reservoir, vis_enc, shape_id, n_steps):
    """收集瞬时脉冲（fired 0/1 向量），不做 PSP 滤波。"""
    spikes = np.zeros((n_steps, reservoir.n), dtype=np.float32)
    for t in range(n_steps):
        vis = vis_enc.encode(shape_id, rate=50.0)
        fired = reservoir.step(vis)
        spikes[t] = fired
    return spikes


def run_ridge_spikes():
    acc = A.new_acc()
    targets_pos = [RLS.generate_target_trajectory(s, 1000) for s in range(3)]
    for seed in range(NSEED):
        t0 = time.time()
        vis_enc = RLS.VisualEncoder(seed=seed)
        reservoir = RLS.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = RLS.VocalTract(seed=seed + 200)
        targets_force = [RLS.precompute_target_forces(targets_pos[s]) for s in range(3)]

        tsp = 3000
        all_spikes, all_t = [], []
        for s in [0, 1, 2]:
            sp = collect_reservoir_spikes(reservoir, vis_enc, s, tsp)
            ft = targets_force[s]
            nr = tsp // len(ft) + 1
            all_spikes.append(sp)
            all_t.append(np.tile(ft, (nr, 1))[:tsp].astype(np.float64))
        X = np.vstack(all_spikes).astype(np.float64)
        Y = np.vstack(all_t)

        # 岭回归（与 run_ridge 相同的 alpha 扫描）
        XtX = X.T @ X
        XtY = X.T @ Y
        I = np.eye(X.shape[1])
        best = (-1.0, None)
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            w = np.linalg.solve(XtX + alpha * I, XtY).T
            tc = A.cc(X @ w.T, Y)
            if tc > best[0]:
                best = (tc, w)
        w_out = best[1]

        # 测试：闭环保留瞬时脉冲（无 PSP）
        reservoir_test = RLS.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        outs = []
        for s in [0, 1, 2]:
            vt.reset()
            pi = []
            for _ in range(TEST_STEPS):
                vis = vis_enc.encode(s, rate=50.0)
                fired = reservoir_test.step(vis)
                out = w_out @ fired.astype(np.float64)     # <-- 瞬时脉冲，无 PSP
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
    return A.finalize(acc)


if __name__ == "__main__":
    res = run_ridge_spikes()
    oo = (res["output_output"][0][1] + res["output_output"][0][2] +
          res["output_output"][1][2]) / 3
    print("")
    print("=== 岭回归 + 瞬时脉冲（对照）===")
    print("raw_diag  = %.4f" % res["raw_diag"])
    print("raw_rank1 = %d/30 = %.0f%%" % (res["raw_rank1"], 100 * res["raw_rank1_pct"]))
    print("cen_rank1 = %d/30 = %.0f%%" % (res["cen_rank1"], 100 * res["cen_rank1_pct"]))
    print("output similarity = %.3f" % oo)
    print("cen_per_shape =", res["cen_per_shape"])
    with open(os.path.join(HERE, "control_ridge_spikes.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved -> control_ridge_spikes.json")
