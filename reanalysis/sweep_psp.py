# -*- coding: utf-8 -*-
"""
长线扫描：STDP 读出层 + PSP 滤波 + 参数扫描
==========================================
问题：如果给 STDP 读出层和基线**一样的输入表示**（PSP 滤波后的放电率），
      并且用**新尺子**去挑参数，它能不能追上岭回归？

两个扫描维度：
  psp_norm : 输入归一化尺度。x = psp / psp_norm
             psp_norm=20 与岭回归/RLS 完全一致（最公平的点），
             往小扫 = 放大输入驱动。
  a        : STDP 学习率（a_plus = a_minus = a）

评测用新尺子（和 reanalysis/REPORT.md 一致）：
  raw_rank1 : 旧尺子下"自己那个目标排第一"的比例（瞎猜 33%）
  cen_rank1 : 去掉共同成分后同样的比例
  oo        : 三次输出互相的相似度（越接近 1 = 越像同一个动作）

不修改任何原有文件。

用法:  python -B sweep_psp.py
输出:  sweep_psp.json / SWEEP_REPORT.md / sweep_log.txt
"""

import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\EGO"
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import rigorous_experiments as R
from reanalyze import analyze, merge, new_acc, finalize, cc

SWEEP_SEEDS = 3          # 扫描用 3 个种子（快）
FINAL_SEEDS = 10         # 最优配置用 10 个种子复核
TEST_STEPS = 500
TAU_PSP = 20.0

PSP_NORMS = [20.0, 10.0, 5.0, 2.0, 1.0]
LEARNING_RATES = [0.005, 0.01, 0.02, 0.05]

_log = []


def log(msg=""):
    print(msg, flush=True)
    _log.append(str(msg))


class STDPReadoutPSP(R.STDPReadout):
    """与 STDPReadout 的差别：突触输入用 PSP 滤波后的放电率。

    生物学依据：真实突触电流本来就有时间过程（PSP），
    所以滤波版本更像生物，而不是更不像。
    """

    def __init__(self, *a, tau_psp=TAU_PSP, psp_norm=20.0, **k):
        super().__init__(*a, **k)
        self.tau_psp = tau_psp
        self.decay_psp = np.exp(-self.dt / tau_psp)
        self.psp = np.zeros(self.n_in, dtype=np.float32)
        self.psp_norm = float(psp_norm)

    def step(self, reservoir_fired, target_current=None):
        self.psp = self.psp * self.decay_psp + reservoir_fired.astype(np.float32)
        x = (self.psp / self.psp_norm).astype(np.float32)

        decay_pre = np.exp(-self.dt / self.tau_stdp).astype(np.float32)
        self.x_pre = self.x_pre * decay_pre + reservoir_fired.astype(np.float32)
        self.y_post = self.y_post * decay_pre + self.fired_out.astype(np.float32)

        if self.use_eligibility:
            decay_elig = np.exp(-self.dt / self.tau_elig).astype(np.float32)
            self.elig *= decay_elig
            self.elig += np.outer(self.fired_out.astype(np.float32),
                                  reservoir_fired.astype(np.float32))

        self.refractory_out = np.maximum(0.0, self.refractory_out - self.dt)
        I_syn = self.w_out @ x                       # <-- 唯一的改动
        if target_current is not None:
            I_syn = I_syn + target_current

        active = self.refractory_out <= 0.0
        dv = (-self.v_out + I_syn) / 10.0 * self.dt
        self.v_out[active] += dv[active]
        self.v_out[self.v_out < 0] = 0
        self.fired_out = (self.v_out >= 1.0).astype(np.float32)
        self.v_out[self.fired_out > 0.5] = 0.0
        self.refractory_out[self.fired_out > 0.5] = 2.0

        if self.use_homeostasis:
            self.rate_out = 0.999 * self.rate_out + 0.001 * self.fired_out
            rate_error = self.rate_out - self.homeo_target / 1000.0
            self.w_out += (-self.homeo_gain * rate_error[:, np.newaxis]).astype(np.float32)

        if self.use_stdp and target_current is not None:
            ltp = np.outer(self.fired_out.astype(np.float32), self.x_pre)
            ltd = np.outer(self.y_post, reservoir_fired.astype(np.float32))
            self.w_out += (self.a_plus * ltp - self.a_minus * ltd).astype(np.float32)

        np.clip(self.w_out, 0.0, self.w_max, out=self.w_out)
        return self.fired_out.copy()

    def get_output(self, reservoir_fired):
        raw = self.w_out @ (self.psp / self.psp_norm).astype(np.float32)
        return raw - raw.mean()


def run_config(psp_norm, lr, seeds):
    """跑一组配置，返回汇总指标。"""
    acc = new_acc()
    for seed in seeds:
        vis_enc = R.VisualEncoder(seed=seed)
        reservoir = R.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = R.VocalTract(seed=seed + 200)
        tgs = [R.generate_target_trajectory(s, 1000) for s in range(3)]
        readout = STDPReadoutPSP(1000, 8, seed=seed + 100,
                                 a_plus=lr, a_minus=lr, psp_norm=psp_norm)
        R.train_stdp(reservoir, readout, vis_enc, vt, tgs, [0, 1, 2], 9000, 3.0)

        outs = []
        for si in range(3):
            vt.reset()
            pi = []
            for _ in range(TEST_STEPS):
                vis = vis_enc.encode(si, rate=50.0)
                fired = reservoir.step(vis)
                readout.step(fired, target_current=None)
                out = readout.get_output(fired)
                force = out * 3.0 / 10.0
                np.clip(force, -20.0, 20.0, out=force)
                vt.step(force)
                pi.append(vt.position.copy())
            outs.append(np.array(pi))

        merge(acc, analyze(outs, [t[:TEST_STEPS].copy() for t in tgs]))
    return finalize(acc)


def main():
    t_all = time.time()
    log("=" * 78)
    log("  长线扫描：STDP 读出层 + PSP 滤波 + 参数扫描")
    log("  扫描 %d 组 (psp_norm x 学习率)，每组 %d 个种子" %
        (len(PSP_NORMS) * len(LEARNING_RATES), SWEEP_SEEDS))
    log("=" * 78)
    log("")
    log("参照（来自 reanalysis/REPORT.md，10 个种子）:")
    log("  STDP 原样 (无 PSP) : raw_rank1 27%   cen_rank1 40%   输出相似度 0.992")
    log("  岭回归 (主流)      : raw_rank1 67%   cen_rank1 77%   输出相似度 0.822")
    log("  瞎猜               : 33%")
    log("")

    results = []
    for psp_norm in PSP_NORMS:
        for lr in LEARNING_RATES:
            t0 = time.time()
            res = run_config(psp_norm, lr, list(range(SWEEP_SEEDS)))
            oo = (res["output_output"][0][1] + res["output_output"][0][2]
                  + res["output_output"][1][2]) / 3
            row = {
                "psp_norm": psp_norm, "lr": lr,
                "raw_rank1_pct": res["raw_rank1_pct"],
                "cen_rank1_pct": res["cen_rank1_pct"],
                "raw_diag": res["raw_diag"],
                "cen_gap": res["cen_diag"] - res["cen_off"],
                "oo": oo,
                "raw_per_shape": res["raw_per_shape"],
                "cen_per_shape": res["cen_per_shape"],
            }
            results.append(row)
            log("psp_norm=%5.1f  lr=%.3f  raw_rank1=%3.0f%%  cen_rank1=%3.0f%%  "
                "旧尺子=%.3f  cen_gap=%+.3f  输出相似度=%.3f  (%.0fs)" % (
                    psp_norm, lr, 100 * res["raw_rank1_pct"], 100 * res["cen_rank1_pct"],
                    res["raw_diag"], row["cen_gap"], oo, time.time() - t0))

    # 选最优：先看新尺子 rank1，再看 cen_gap
    best = max(results, key=lambda r: (r["cen_rank1_pct"], r["cen_gap"]))
    log("")
    log("扫描最优配置: psp_norm=%.1f, lr=%.3f  (cen_rank1=%.0f%%)" %
        (best["psp_norm"], best["lr"], 100 * best["cen_rank1_pct"]))

    log("")
    log("用 %d 个种子复核最优配置..." % FINAL_SEEDS)
    final = run_config(best["psp_norm"], best["lr"], list(range(FINAL_SEEDS)))
    oo_f = (final["output_output"][0][1] + final["output_output"][0][2]
            + final["output_output"][1][2]) / 3

    log("")
    log("=" * 78)
    log("  最终对比（10 个种子）")
    log("=" * 78)
    log("  %-28s %10s %10s %10s" % ("配置", "旧尺子", "新尺子", "输出相似度"))
    log("  " + "-" * 62)
    log("  %-28s %9.0f%% %9.0f%% %10.3f" % ("STDP 原样（无 PSP）", 27, 40, 0.992))
    log("  %-28s %9.0f%% %9.0f%% %10.3f" % (
        "STDP+PSP (最优配置)", 100 * final["raw_rank1_pct"],
        100 * final["cen_rank1_pct"], oo_f))
    log("  %-28s %9.0f%% %9.0f%% %10.3f" % ("岭回归（主流）", 67, 77, 0.822))
    log("  %-28s %9s %9s %10s" % ("瞎猜", "33%", "33%", "—"))
    log("")
    log("  最优配置细节: psp_norm=%.1f, lr=%.3f" % (best["psp_norm"], best["lr"]))
    log("  每形状挑对次数（新尺子）: %s / %d" %
        (final["cen_per_shape"], final["seeds"]))
    log("  每形状挑对次数（旧尺子）: %s / %d" %
        (final["raw_per_shape"], final["seeds"]))

    out = {
        "meta": {
            "sweep_seeds": SWEEP_SEEDS, "final_seeds": FINAL_SEEDS,
            "tau_psp": TAU_PSP, "test_steps": TEST_STEPS,
            "psp_norms": PSP_NORMS, "learning_rates": LEARNING_RATES,
            "total_seconds": time.time() - t_all,
        },
        "reference": {
            "stdp_original": {"raw_rank1_pct": 0.2667, "cen_rank1_pct": 0.40, "oo": 0.992},
            "ridge": {"raw_rank1_pct": 0.6667, "cen_rank1_pct": 0.7667, "oo": 0.822},
            "chance": 1.0 / 3.0,
        },
        "sweep": results,
        "best": best,
        "final": final,
        "final_oo": oo_f,
    }
    with open(os.path.join(HERE, "sweep_psp.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "sweep_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")

    log("")
    log("结果: %s" % os.path.join(HERE, "sweep_psp.json"))
    log("用时 %.0f 秒" % (time.time() - t_all))


if __name__ == "__main__":
    main()
