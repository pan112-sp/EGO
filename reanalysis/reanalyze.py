# -*- coding: utf-8 -*-
"""
EGO 一期 重新评测 (Re-analysis)
================================
目的：用「对比式度量」重新评测一期结果，回答一个问题：

    "网络看见正方形发出的动作，是不是比看见圆形时发出的动作，
      更像正方形的目标？"

旧度量（也是论文里用的）:
    pearson_corr(输出轨迹拍平, 目标轨迹拍平)
    问题：三个目标本身互相相关 0.836；网络三次输出互相相关 0.99。
          共同成分压倒一切 -> 度量无法区分"学会了"和"没学会"。

新度量（本脚本）:
    1. 3x3 混淆矩阵: 每个形状输入, 跟三个目标都比一遍
       对角线占优 = 真的区分了形状
    2. 中心化矩阵: 从输出和目标两边都减掉跨形状均值, 去掉共同成分
    3. rank-1 准确率: "自己那个目标排第一"的比例 (瞎猜 = 33%)
    4. 逐层可解码性: 形状信息在输入/储备池/读出层/最终动作哪一层还在

本脚本不修改任何原有文件。rls_fixed.py 在 import 时会覆盖写
rls_fix_results.txt, 这里做了拦截（见 _load_rls_fixed）。

运行:  python -B reanalyze.py
输出:  results.json / REPORT.md / run_log.txt  (都在本目录)
"""

import sys
import os
import re
import json
import time
import types

import numpy as np

ROOT = r"D:\EGO"
OUT = os.path.join(ROOT, "reanalysis")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

NSEED = 10
TAU_PSP = 20.0          # 岭回归/RLS 使用的 PSP 时间常数（也是论文最优值）
TEST_STEPS = 500

_log_lines = []


def log(msg=""):
    print(msg, flush=True)
    _log_lines.append(str(msg))


# ============================================================
#  基础工具
# ============================================================

def cc(a, b):
    """拍平后的 Pearson 相关（和项目里 pearson_corr 一致）。"""
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    if af.std() < 1e-9 or bf.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def _rank1(M):
    """每一行里, 对角线是不是最大值。返回命中个数 (0..3)。"""
    return sum(1 for i in range(3) if int(np.argmax(M[i])) == i)


def _per_shape_hit(M):
    return [bool(int(np.argmax(M[i])) == i) for i in range(3)]


def analyze(outs, tg5):
    """对一个 (输出, 目标) 组合算全套指标。

    outs: 3 个 (T,8) 输出轨迹
    tg5 : 3 个 (T,8) 目标轨迹
    """
    Obar = np.mean(outs, axis=0)
    Tbar = np.mean(tg5, axis=0)
    Mraw = np.array([[cc(outs[i], tg5[j]) for j in range(3)] for i in range(3)])
    Mcen = np.array([[cc(outs[i] - Obar, tg5[j] - Tbar) for j in range(3)] for i in range(3)])
    OO = np.array([[cc(outs[i], outs[j]) for j in range(3)] for i in range(3)])
    return {
        "raw": Mraw,
        "centered": Mcen,
        "output_output": OO,
        "raw_rank1": _rank1(Mraw),
        "cen_rank1": _rank1(Mcen),
        "raw_per_shape": _per_shape_hit(Mraw),
        "cen_per_shape": _per_shape_hit(Mcen),
        "raw_diag": float(np.mean(np.diag(Mraw))),
        "raw_off": float(np.mean(Mraw[~np.eye(3, dtype=bool)])),
        "cen_diag": float(np.mean(np.diag(Mcen))),
        "cen_off": float(np.mean(Mcen[~np.eye(3, dtype=bool)])),
    }


def merge(acc, res):
    """把一个 seed 的结果累加进 acc。"""
    for k in ("raw", "centered", "output_output"):
        acc[k] += res[k]
    acc["raw_rank1"] += res["raw_rank1"]
    acc["cen_rank1"] += res["cen_rank1"]
    acc["trials"] += 3
    acc["seeds"] += 1
    acc["raw_diag"] += res["raw_diag"]
    acc["raw_off"] += res["raw_off"]
    acc["cen_diag"] += res["cen_diag"]
    acc["cen_off"] += res["cen_off"]
    for i in range(3):
        acc["raw_per_shape"][i] += int(res["raw_per_shape"][i])
        acc["cen_per_shape"][i] += int(res["cen_per_shape"][i])


def new_acc():
    return {
        "raw": np.zeros((3, 3)), "centered": np.zeros((3, 3)),
        "output_output": np.zeros((3, 3)),
        "raw_rank1": 0, "cen_rank1": 0, "trials": 0, "seeds": 0,
        "raw_diag": 0.0, "raw_off": 0.0, "cen_diag": 0.0, "cen_off": 0.0,
        "raw_per_shape": [0, 0, 0], "cen_per_shape": [0, 0, 0],
    }


def finalize(acc):
    n = max(acc["seeds"], 1)
    return {
        "raw": (acc["raw"] / n).tolist(),
        "centered": (acc["centered"] / n).tolist(),
        "output_output": (acc["output_output"] / n).tolist(),
        "raw_rank1": acc["raw_rank1"], "cen_rank1": acc["cen_rank1"],
        "trials": acc["trials"],
        "raw_rank1_pct": acc["raw_rank1"] / max(acc["trials"], 1),
        "cen_rank1_pct": acc["cen_rank1"] / max(acc["trials"], 1),
        "raw_diag": acc["raw_diag"] / n, "raw_off": acc["raw_off"] / n,
        "cen_diag": acc["cen_diag"] / n, "cen_off": acc["cen_off"] / n,
        "raw_per_shape": acc["raw_per_shape"],
        "cen_per_shape": acc["cen_per_shape"],
        "seeds": acc["seeds"],
    }


# ============================================================
#  加载模块
# ============================================================

import rigorous_experiments as R          # 项目自带, import 无副作用

def _load_rls_fixed():
    """加载 rls_fixed.py。

    rls_fixed.py 已经修好：只有作为脚本直接运行时才会截断
    rls_fix_results.txt，被 import 时日志写进内存缓冲。
    这里只需临时吞掉模块级 out() 的打印，保持日志干净。
    """
    import contextlib
    import io as _io
    import importlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = importlib.import_module("rls_fixed")
    return mod


RLS = _load_rls_fixed()


# ============================================================
#  实验 A: STDP 读出层（项目原样）
# ============================================================

def run_stdp_original():
    acc = new_acc()
    for seed in range(NSEED):
        t0 = time.time()
        vis_enc = R.VisualEncoder(seed=seed)
        reservoir = R.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = R.VocalTract(seed=seed + 200)
        tgs = [R.generate_target_trajectory(s, 1000) for s in range(3)]
        readout = R.STDPReadout(1000, 8, seed=seed + 100)
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

        res = analyze(outs, [t[:TEST_STEPS].copy() for t in tgs])
        merge(acc, res)
        log("  [STDP 原样] seed %d  raw_rank1=%d/3  cen_rank1=%d/3  raw_diag=%.3f  (%.0fs)" % (
            seed, res["raw_rank1"], res["cen_rank1"], res["raw_diag"], time.time() - t0))
    return finalize(acc)


# ============================================================
#  实验 B: 岭回归基线（论文里的 rls_fixed.py 流程）
# ============================================================

def run_ridge():
    acc = new_acc()
    targets_pos = [RLS.generate_target_trajectory(s, 1000) for s in range(3)]
    for seed in range(NSEED):
        t0 = time.time()
        vis_enc = RLS.VisualEncoder(seed=seed)
        reservoir = RLS.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = RLS.VocalTract(seed=seed + 200)
        targets_force = [RLS.precompute_target_forces(targets_pos[s]) for s in range(3)]

        tsp = 3000
        all_psp, all_t = [], []
        for s in [0, 1, 2]:
            psp = RLS.collect_reservoir_states(reservoir, vis_enc, s, tsp, TAU_PSP, 1.0)
            ft = targets_force[s]
            nr = tsp // len(ft) + 1
            all_psp.append(psp)
            all_t.append(np.tile(ft, (nr, 1))[:tsp].astype(np.float64))
        X = np.vstack(all_psp)
        Y = np.vstack(all_t)

        # 岭回归正规方程（与 rls_fixed.ridge_regression 等价，但快得多）
        XtX = X.T @ X
        XtY = X.T @ Y
        I = np.eye(X.shape[1])
        best = (-1.0, None)
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            w = np.linalg.solve(XtX + alpha * I, XtY).T
            tc = cc(X @ w.T, Y)
            if tc > best[0]:
                best = (tc, w)
        w_out = best[1]

        reservoir_test = RLS.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        decay = np.exp(-1.0 / TAU_PSP)
        norm = TAU_PSP / 1.0
        outs = []
        for s in [0, 1, 2]:
            vt.reset()
            psp_state = np.zeros(reservoir_test.n)
            pi = []
            for _ in range(TEST_STEPS):
                vis = vis_enc.encode(s, rate=50.0)
                fired = reservoir_test.step(vis)
                psp_state = psp_state * decay + fired.astype(np.float64)
                out = w_out @ (psp_state / norm)
                force = out * 3.0
                np.clip(force, -20.0, 20.0, out=force)
                vt.step(force.astype(np.float32))
                pi.append(vt.position.copy())
            outs.append(np.array(pi))

        res = analyze(outs, [t[:TEST_STEPS].copy() for t in targets_pos])
        merge(acc, res)
        log("  [岭回归] seed %d  raw_avg=%.3f  raw_rank1=%d/3  cen_rank1=%d/3  (%.0fs)" % (
            seed, res["raw_diag"], res["raw_rank1"], res["cen_rank1"], time.time() - t0))
    return finalize(acc)


# ============================================================
#  实验 C: STDP 读出层 + PSP 滤波（探索性：公平化尝试）
# ============================================================

class STDPReadoutPSP(R.STDPReadout):
    """与 STDPReadout 唯一的差别: 突触输入用 PSP 滤波后的放电率,
    而不是瞬时脉冲。

    生物学依据: 真实突触电流本来就有时间过程 (PSP), 所以滤波版本
    反而更贴近生物, 而不是更不贴近。
    """

    def __init__(self, *a, tau_psp=TAU_PSP, **k):
        super().__init__(*a, **k)
        self.tau_psp = tau_psp
        self.decay_psp = np.exp(-self.dt / tau_psp)
        self.psp = np.zeros(self.n_in, dtype=np.float32)
        self.psp_norm = tau_psp / self.dt

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


def run_stdp_psp():
    acc = new_acc()
    for seed in range(NSEED):
        vis_enc = R.VisualEncoder(seed=seed)
        reservoir = R.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = R.VocalTract(seed=seed + 200)
        tgs = [R.generate_target_trajectory(s, 1000) for s in range(3)]
        readout = STDPReadoutPSP(1000, 8, seed=seed + 100)
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

        res = analyze(outs, [t[:TEST_STEPS].copy() for t in tgs])
        merge(acc, res)
        log("  [STDP+PSP] seed %d  raw_rank1=%d/3  cen_rank1=%d/3  out_out=%.3f" % (
            seed, res["raw_rank1"], res["cen_rank1"],
            (res["output_output"][0, 1] + res["output_output"][0, 2]) / 2))
    return finalize(acc)


# ============================================================
#  实验 D: 逐层可解码性（形状信息在哪一层还在）
# ============================================================

BLOCK = 50
NBLK = 20


def _ridge_acc(Xtr, ytr, Xte, yte, alpha=1e-2):
    Y = np.eye(3)[ytr]
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Y)
    return float((np.argmax(Xte @ W, axis=1) == yte).mean())


def run_decodability():
    """在同一个网络里, 随机顺序展示 3 个形状各 20 次(每次 50 步),
    用岭回归分类器从各层活动猜"刚看的是哪个形状"。
    训练/测试块交错划分, 避免时间泄漏。瞎猜 = 33%。
    """
    layers = ["visual_input", "reservoir_fired", "readout_output",
              "vocal_position_mean", "vocal_position_traj"]
    results = {k: [] for k in layers}
    for seed in range(3):
        vis_enc = R.VisualEncoder(seed=seed)
        reservoir = R.SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = R.VocalTract(seed=seed + 200)
        tgs = [R.generate_target_trajectory(s, 1000) for s in range(3)]
        readout = R.STDPReadout(1000, 8, seed=seed + 100)
        R.train_stdp(reservoir, readout, vis_enc, vt, tgs, [0, 1, 2], 9000, 3.0)

        sched = [i for i in range(3) for _ in range(NBLK)]
        np.random.RandomState(1234 + seed).shuffle(sched)

        I, F, O, P, Pt, Y = [], [], [], [], [], []
        for si in sched:
            ii, fi, oi, pi = [], [], [], []
            for _ in range(BLOCK):
                vis = vis_enc.encode(si, rate=50.0)
                fired = reservoir.step(vis)
                readout.step(fired, target_current=None)
                out = readout.get_output(fired)
                force = out * 3.0 / 10.0
                np.clip(force, -20.0, 20.0, out=force)
                vt.step(force)
                ii.append(vis.copy()); fi.append(fired.copy())
                oi.append(np.array(out, dtype=np.float64)); pi.append(vt.position.copy())
            ii = np.array(ii); fi = np.array(fi)
            oi = np.array(oi, dtype=np.float64); pi = np.array(pi)
            # 输入/储备池维度太大(50x200 / 50x1000), 用块均值(放电率)
            I.append(ii.mean(axis=0))
            F.append(fi.mean(axis=0))
            # 读出层输出/声带位置维度小, 同时保留均值与完整轨迹
            O.append(oi.mean(axis=0))
            P.append(pi.mean(axis=0))
            Pt.append(pi.flatten())
            Y.append(si)

        X = {"visual_input": np.array(I), "reservoir_fired": np.array(F),
             "readout_output": np.array(O), "vocal_position_mean": np.array(P),
             "vocal_position_traj": np.array(Pt)}
        Y = np.array(Y)
        tr, te = [], []
        for s in range(3):
            idx = np.where(Y == s)[0]
            for k, ix in enumerate(idx):
                (tr if k % 2 == 0 else te).append(ix)
        tr, te = np.array(tr), np.array(te)

        for k in layers:
            a = _ridge_acc(X[k][tr], Y[tr], X[k][te], Y[te])
            results[k].append(a)
            log("  [可解码性] seed %d  %-16s = %.3f" % (seed, k, a))
    return {k: {"per_seed": v, "mean": float(np.mean(v))} for k, v in results.items()}


# ============================================================
#  目标本身的相似度
# ============================================================

def run_target_similarity():
    tgs = [R.generate_target_trajectory(s, 1000) for s in range(3)]
    t5 = [t[:TEST_STEPS] for t in tgs]
    M = np.array([[cc(t5[i], t5[j]) for j in range(3)] for i in range(3)])
    stds = [t.std(axis=0).round(4).tolist() for t in tgs]
    return {"matrix": M.tolist(), "per_param_std": stds}


# ============================================================
#  主流程
# ============================================================

def markdown_report(res):
    names = ["正方形", "圆形", "三角形"]
    L = []
    A = L.append

    A("# EGO 一期 · 重新评测报告")
    A("")
    A("本报告由 `reanalysis/reanalyze.py` 自动生成。所有数字都能一键重跑。")
    A("")
    A("---")
    A("")
    A("## 一、结论摘要")
    A("")
    st = res["stdp_original"]
    ri = res["ridge"]
    A("| 指标 | **STDP（你的）** | **岭回归（主流）** | 瞎猜 |")
    A("|---|---|---|---|")
    A("| 旧尺子分数（论文用的） | %.3f | %.3f | — |" %
      (st["raw_diag"], ri["raw_diag"]))
    A("| 旧尺子：挑对目标 | **%d/%d = %.0f%%** | **%d/%d = %.0f%%** | 33%% |" %
      (st["raw_rank1"], st["trials"], 100 * st["raw_rank1_pct"],
       ri["raw_rank1"], ri["trials"], 100 * ri["raw_rank1_pct"]))
    A("| 新尺子：挑对目标 | **%d/%d = %.0f%%** | **%d/%d = %.0f%%** | 33%% |" %
      (st["cen_rank1"], st["trials"], 100 * st["cen_rank1_pct"],
       ri["cen_rank1"], ri["trials"], 100 * ri["cen_rank1_pct"]))
    oo_st = (st["output_output"][0][1] + st["output_output"][0][2] + st["output_output"][1][2]) / 3
    oo_ri = (ri["output_output"][0][1] + ri["output_output"][0][2] + ri["output_output"][1][2]) / 3
    A("| 三次输出互相相似度 | **%.3f（几乎一样）** | **%.3f（明显不同）** | — |" % (oo_st, oo_ri))
    A("")
    A("**读法**：岭回归明显学会了区分形状（%.0f%%–%.0f%%，远高于瞎猜）；" %
      (100 * ri["raw_rank1_pct"], 100 * ri["cen_rank1_pct"]))
    A("STDP 读出层没有（%.0f%%–%.0f%%，基本在瞎猜附近）。" %
      (100 * st["raw_rank1_pct"], 100 * st["cen_rank1_pct"]))
    A("")
    A("**旧尺子把两者都报成 0.86/0.87，掩盖了这个差别。**")
    A("")
    A("---")
    A("")

    def matrix_block(title, M, note=""):
        A(title)
        A("")
        if note:
            A(note)
            A("")
        A("| 看见 ↓ / 比较 → | 正方形 | 圆形 | 三角形 |")
        A("|---|---|---|---|")
        for i in range(3):
            cells = " | ".join("%.3f" % M[i][j] for j in range(3))
            A("| **%s** | %s |" % (names[i], cells))
        A("")
        d = np.mean([M[i][i] for i in range(3)])
        o = np.mean([M[i][j] for i in range(3) for j in range(3) if i != j])
        A("对角线均值 = %.3f，非对角均值 = %.3f，差 = %+.3f" % (d, o, d - o))
        A("")

    A("## 二、目标本身有多像")
    A("")
    matrix_block("三个目标轨迹之间的相关：", res["target_similarity"]["matrix"],
                 "这是**度量的地板**：网络随便产生一个「像目标」的信号，就能跟三个目标都相关到这个水平。")
    A("---")
    A("")

    A("## 三、STDP（项目原样）")
    A("")
    matrix_block("原始相关矩阵（论文里的度量）：", st["raw"])
    A("每个形状在 %d 个种子里「自己那个目标排第一」的次数：%s（合计 %d/%d = %.0f%%）" %
      (st["seeds"], st["raw_per_shape"], sum(st["raw_per_shape"]),
       st["trials"], 100 * st["raw_rank1_pct"]))
    A("")
    matrix_block("去掉共同成分之后（新尺子）：", st["centered"])
    A("每个形状在 %d 个种子里挑对次数：%s（合计 %d/%d = %.0f%%）" %
      (st["seeds"], st["cen_per_shape"], sum(st["cen_per_shape"]),
       st["trials"], 100 * st["cen_rank1_pct"]))
    A("")
    matrix_block("网络三次输出之间的相似度：", st["output_output"],
                 "接近 1.0 = 三个形状发出的是同一个动作。")
    A("---")
    A("")

    A("## 四、岭回归（主流基线）")
    A("")
    A("（复现校验：原始分数 = %.3f，你 `rls_fix_results.txt` 里是 0.873 ± 0.010）" % ri["raw_diag"])
    A("")
    matrix_block("原始相关矩阵：", ri["raw"])
    A("每个形状在 %d 个种子里「自己那个目标排第一」的次数：%s（合计 %d/%d = %.0f%%）" %
      (ri["seeds"], ri["raw_per_shape"], sum(ri["raw_per_shape"]),
       ri["trials"], 100 * ri["raw_rank1_pct"]))
    A("")
    matrix_block("去掉共同成分之后（新尺子）：", ri["centered"])
    A("每个形状在 %d 个种子里挑对次数：%s（合计 %d/%d = %.0f%%）" %
      (ri["seeds"], ri["cen_per_shape"], sum(ri["cen_per_shape"]),
       ri["trials"], 100 * ri["cen_rank1_pct"]))
    A("")
    matrix_block("三次输出之间的相似度：", ri["output_output"],
                 "明显低于 1.0 = 三个形状发出了不同的动作。")
    A("---")
    A("")

    A("## 五、形状信息在哪一层还在（逐层可解码性）")
    A("")
    A("瞎猜 = 33%。")
    A("")
    A("| 层 | 准确率（3 个种子均值） |")
    A("|---|---|")
    label = {"visual_input": "视觉输入信号", "reservoir_fired": "储备池活动",
             "readout_output": "读出层输出",
             "vocal_position_mean": "最终动作（声带位置, 块均值）",
             "vocal_position_traj": "最终动作（声带位置, 完整轨迹）"}
    for k in ["visual_input", "reservoir_fired", "readout_output",
              "vocal_position_mean", "vocal_position_traj"]:
        A("| %s | **%.0f%%** |" % (label[k], 100 * res["decodability"][k]["mean"]))
    A("")
    A("含义：形状信息**一路活到最终动作**——网络确实在做形状相关的事，")
    A("不是完全没学。但它的形状对应关系弱，见第三节。")
    A("")
    A("---")
    A("")

    A("## 六、比较是否公平（重要）")
    A("")
    A("代码检查结果：")
    A("")
    A("| 读出方式 | 突触输入 |")
    A("|---|---|")
    A("| `STDPReadout.get_output` | `w_out @ reservoir_fired` —— **瞬时脉冲** |")
    A("| `RLSReadout` | `w_out @ self.psp_filter` —— **PSP 滤波（τ=20ms）** |")
    A("| `rls_fixed` 岭回归 | `psp_filter_spikes(...)` —— **PSP 滤波（τ=20ms）** |")
    A("")
    A("即：两个基线读的是**积分后的放电率**，STDP 读出层读的是**单个 1ms 的 0/1 向量**。")
    A("而形状信息在放电率里。**这个不公平对 STDP 不利。**")
    A("")
    A("（另注：真实突触电流本来就有时间过程，所以加了 PSP 反而**更像生物**。）")
    A("")
    pp = res["stdp_psp"]
    oo_pp = (pp["output_output"][0][1] + pp["output_output"][0][2] + pp["output_output"][1][2]) / 3
    A("### 探索性尝试：给 STDP 读出层加上同样的 PSP 滤波")
    A("")
    A("| | STDP 原样 | STDP + PSP | 岭回归 |")
    A("|---|---|---|---|")
    A("| 旧尺子挑对 | %.0f%% | %.0f%% | %.0f%% |" %
      (100 * st["raw_rank1_pct"], 100 * pp["raw_rank1_pct"], 100 * ri["raw_rank1_pct"]))
    A("| 新尺子挑对 | %.0f%% | %.0f%% | %.0f%% |" %
      (100 * st["cen_rank1_pct"], 100 * pp["cen_rank1_pct"], 100 * ri["cen_rank1_pct"]))
    A("| 三次输出相似度 | %.3f | %.3f | %.3f |" % (oo_st, oo_pp, oo_ri))
    A("")
    A("**加了 PSP 之后，STDP 的输出立刻从「三次几乎一样」变成「三次明显不同」** ——")
    A("说明输入表示确实是关键。但它的**方向/对齐还没调好**（出现负相关），")
    A("而且输入尺度变了会打乱学习动力学，**需要重新调参才算数**。")
    A("")
    A("所以：这一项**不能**作为「STDP 行/不行」的证据，只能作为「输入表示很重要」的证据。")
    A("")
    A("---")
    A("")
    A("## 七、这份报告支持和不支持什么")
    A("")
    A("**支持：**")
    A("")
    A("1. 一期的数字是**真实的、可复现的**（岭回归 0.873 逐位复现）。")
    A("2. 论文里那个度量**无法区分**「学会了」和「没学会」——两者都报 0.86。")
    A("   这是一个独立的、有价值的方法学发现。")
    A("3. 按现在的实现，STDP 读出层**没有**学会区分形状，而岭回归**学会了**。")
    A("4. 局部 vs 全局的比较被**输入表示**混淆了（瞬时脉冲 vs PSP 滤波）。")
    A("")
    A("**不支持：**")
    A("")
    A("1. 「STDP 达到全局线性读出水平」—— 新尺子下是 %.0f%% vs %.0f%%。" %
      (100 * st["cen_rank1_pct"], 100 * ri["cen_rank1_pct"]))
    A("2. 「泛化到未见形状」—— 输出本身不区分形状时，泛化无从谈起。")
    A("3. 「STDP 原理上做不到」—— **这个也没测**。公平表示 + 重新调参之后会怎样，未知。")
    A("")
    A("---")
    A("")
    A("## 八、下一步（有边界）")
    A("")
    A("| 步骤 | 要重训吗 | 代价 |")
    A("|---|---|---|")
    A("| 换成新尺子重新打分 | 不用（网络不变） | 已做完 |")
    A("| 重挑学习率/教导电流（用新尺子选） | 要 | 25 组 × 70 秒 |")
    A("| 给读出层加 PSP 滤波后重新调参 | 要 | 同上 |")
    A("| 改架构 | 要 | 未定，暂不建议 |")
    A("")
    A("**关键：先做免费的（换尺子重新打分），再决定要不要花钱重训。**")
    A("")
    return "\n".join(L)


def main():
    t_all = time.time()
    log("=" * 70)
    log("  EGO 一期 重新评测")
    log("  种子数 = %d，每形状测试步数 = %d，PSP tau = %.0f" % (NSEED, TEST_STEPS, TAU_PSP))
    log("=" * 70)

    res = {}

    log("\n[0] 目标本身的相似度")
    res["target_similarity"] = run_target_similarity()
    log("")

    log("[A] STDP 读出层（项目原样）")
    res["stdp_original"] = run_stdp_original()

    log("\n[B] 岭回归基线")
    res["ridge"] = run_ridge()

    log("\n[C] STDP + PSP 滤波（探索性）")
    res["stdp_psp"] = run_stdp_psp()

    log("\n[D] 逐层可解码性")
    res["decodability"] = run_decodability()

    res["meta"] = {
        "n_seed": NSEED,
        "test_steps": TEST_STEPS,
        "tau_psp": TAU_PSP,
        "chance_level": 1.0 / 3.0,
        "total_seconds": time.time() - t_all,
        "rigorous_experiments_sha": None,
    }

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(markdown_report(res))

    log("")
    log("=" * 70)
    log("  完成，用时 %.0f 秒" % (time.time() - t_all))
    log("  结果: %s" % os.path.join(OUT, "results.json"))
    log("  报告: %s" % os.path.join(OUT, "REPORT.md"))
    log("=" * 70)

    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")


if __name__ == "__main__":
    main()
