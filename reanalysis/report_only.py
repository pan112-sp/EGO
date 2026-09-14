# -*- coding: utf-8 -*-
"""只从 results.json 重新生成 REPORT.md，不重跑实验。

用法:  python -B report_only.py
"""
import os
import json
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reanalyze  # 复用同一套报告模板，保证格式一致

path = os.path.join(HERE, "results.json")
with open(path, encoding="utf-8") as f:
    res = json.load(f)

md = reanalyze.markdown_report(res)
with open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("REPORT.md regenerated from results.json")
