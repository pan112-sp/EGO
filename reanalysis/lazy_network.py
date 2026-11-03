# -*- coding: utf-8 -*-
# 验证原话: 无（探索性 · 海报"偷懒网络"对照，验证旧尺子会把恒定输出评高分）
"""
偷懒网络对照：网络对三个形状都输出同一个轨迹（= 正方形目标 T0），完全不分形状。
验证：旧尺子（Pearson 对角线均值）会给它多少分？新尺子（rank-1）会给它多少分？
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)

import numpy as np
import rigorous_experiments as R
from reanalyze import analyze

TEST_STEPS = 500


def main():
    tgs = [R.generate_target_trajectory(s, 1000) for s in range(3)]
    t5 = [t[:TEST_STEPS].copy() for t in tgs]
    # 偷懒网络：三个形状都输出「正方形目标 T0」
    outs = [t5[0].copy(), t5[0].copy(), t5[0].copy()]
    res = analyze(outs, t5)

    print("=== 偷懒网络（三个形状都输出同一个 T0）===")
    print("旧尺子 raw_diag  = %.4f  (= 对角线均值)" % res["raw_diag"])
    print("旧尺子 raw_rank1 = %d/3 = %.0f%%" % (res["raw_rank1"], 100 * res["raw_rank1"] / 3))
    print("新尺子 cen_rank1 = %d/3 = %.0f%%" % (res["cen_rank1"], 100 * res["cen_rank1"] / 3))
    print("raw 矩阵:")
    print(np.round(res["raw"], 3))
    # 对照：真实网络
    print()
    print("对照：真实 STDP 旧尺子 0.865 / 新尺子 40%")
    print("对照：真实岭回归 旧尺子 0.873 / 新尺子 77%")
    out = {
        "lazy": {
            "raw_diag": res["raw_diag"],
            "raw_rank1": res["raw_rank1"],
            "cen_rank1": res["cen_rank1"],
            "raw_matrix": res["raw"].tolist() if hasattr(res["raw"], "tolist") else res["raw"],
        }
    }
    with open(os.path.join(HERE, "lazy_network.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved -> lazy_network.json")


if __name__ == "__main__":
    main()
