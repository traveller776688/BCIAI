#!/usr/bin/env python3
"""最终裁决: DAME-C5 vs 同协议基线 (8被试 LOSO 原始波形, seed42, 15轮)
用法: python final_verdict.py [baseline_json] [dame_json]
默认: results/eeg_v5_base8_results.json, results/eeg_v5_iter3_results.json
输出: markdown 对比表 + 配对差异 + 结论行
"""
import json, sys
import numpy as np

BASELINE_JSON = sys.argv[1] if len(sys.argv) > 1 else "results/eeg_v5_base8_results.json"
DAME_JSON = sys.argv[2] if len(sys.argv) > 2 else "results/eeg_v5_iter3_results.json"

def folds_of(d, model, kind):
    """提取某模型某协议的逐折 acc (按 s0..s7 排序)"""
    ks = sorted(
        (k for k in d if k.startswith(f"{model}_{kind}_s") and isinstance(d[k], (int, float))),
        key=lambda k: int(k.split("_s")[1].split("_")[0]))
    return [d[k] for k in ks]

base = json.load(open(BASELINE_JSON))
dame = json.load(open(DAME_JSON))

BASELINES = [
    ("EEGNet", "plain"), ("TSception", "plain"), ("DGCNN", "plain"),
    ("EEGConformer", "plain"), ("LMDA-Net", "plain"),
    ("DANN", "dann"), ("DeepCORAL", "coral"),
]

dame_folds = folds_of(dame, "DAME-C5", "dame")
dame_mean, dame_std = np.mean(dame_folds), np.std(dame_folds, ddof=1)
print(f"DAME-C5 (iter3, 8被试): {dame_mean*100:.2f} ± {dame_std*100:.2f} (n={len(dame_folds)})\n")

print("| 模型 | acc% (n折) | Δ vs DAME-C5 | 结论 |")
print("|------|-----------|--------------|------|")
rows = []
for mname, kind in BASELINES:
    fs = folds_of(base, mname, kind)
    if not fs:
        print(f"| {mname} | — (未完成) | — | — |")
        continue
    mean, std = np.mean(fs), np.std(fs, ddof=1)
    delta = mean - dame_mean
    n_common = min(len(fs), len(dame_folds))
    # 配对: 相同被试折 (s0..sN 对应同一被试)
    paired_d = np.array(fs[:n_common]) - np.array(dame_folds[:n_common])
    n_win = int((paired_d > 0).sum())
    verdict = "DAME领先" if delta < 0 else ("持平(差<1点)" if delta < 1.0 else "基线领先")
    print(f"| {mname} | {mean*100:.2f} ± {std*100:.2f} (n={len(fs)}) | "
          f"{delta*100:+.2f} | {verdict} (DAME赢 {n_common-n_win}/{n_common} 折) |")
    rows.append((mname, mean, std, delta))

if rows:
    best_base = min(rows, key=lambda r: -r[1])  # 最强基线
    print(f"\n最强基线: {best_base[0]} {best_base[1]*100:.2f}% → "
          f"DAME-C5 领先 {-(best_base[3])*100:.2f} 点")
    beats_all = all(r[3] < 0 for r in rows)
    print(f"\n结论: DAME-C5 {'领先全部已跑完基线 [YES]' if beats_all else '未全领先'}")
    print("(8被试×1种子×15轮快速协议; 显著性检验需最终15×3全量, 已推迟至实验室显卡)")
