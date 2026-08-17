#!/usr/bin/env python3
"""不变性面板 v2：聚合 iter4 半程(3折)有效指标 + stab 红旗记录
输出: results/invariance_panel.md + results/figures_v5/invariance_panel.png
2026-08-14 v2: 移除 stab_acc 面板 — stab_acc ≡ 1−stab_prior (float64逐位相等) 且三折 stab_per_class 逐位相同
  → 多数类平凡预测(79.7%=稳定对占比, L2陷阱再现) 或 npz 记录bug, 两种解释都推翻"强信号"结论
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDS = ["s0", "s1", "s2"]
rows = []
for s in FOLDS:
    d = np.load(f"results/diag_v5/DAME-C5_dame_{s}_seed42.npz")
    rows.append({
        "fold": s,
        "stab_acc": float(d["stab_acc"]),
        "stab_prior": float(d["stab_prior"]),
        "drift_cos": float(d["cross_trial_drift_cos"]),
        "self_mse": float(d["self_pred_mse"]),
    })

# ---- markdown ----
md = []
md.append("# 不变性面板 v2（iter4 半程，3 折，2026-08-14 聚合）\n")
md.append("「以不变应万变」三层证据的数值面板。数据来源：`results/diag_v5/DAME-C5_dame_s{0,1,2}_seed42.npz` + `results/v5_iter4.log`。\n")
md.append("## 有效指标\n")
md.append("| 折 | driftCos | selfMSE |")
md.append("|---|---|---|")
for r in rows:
    md.append(f"| {r['fold']} | {r['drift_cos']:.3f} | {r['self_mse']:.4f} |")
md.append("")
md.append("**读法**：driftCos = 相邻窗口表示余弦相似度（0.97+ 表示跨窗口稳定）；selfMSE = Z 对耦合本征的自洽读出误差（~1e-4 量级说明 Z 编码耦合结构而非噪声）——这两项支撑「表示层不变性」。")
md.append("")
md.append("## 🔴 红旗：stab 指标已证伪（2026-08-14）\n")
md.append("- **stab_acc ≡ 1 − stab_prior**：三折均为 0.797101 vs 0.202899，float64 逐位互补——稳定性头极可能一直预测多数类（稳定对占比恰为 79.7%），即 L2 平凡基线陷阱在稳定性任务上再现；")
md.append("- **三折 stab_per_class 逐位相同**（[0.8235, 0.7647, 0.8889, 0.7059]）——三个不同被试不可能相同，npz 记录层亦存 bug；")
md.append("- 两种解释（多数类平凡解 / 记录bug）都推翻此前「stab=79.7% vs 20.3% 首现强可证伪信号」的结论，蓝图与记忆中该表述已同步修正；")
md.append("- **修复方案（全量时）**：稳定性配对采样改为 50/50 平衡（多数类预测失效）+ 逐折独立落盘 + 报告 balanced accuracy 而非原始 acc。")
md.append("")
md.append("## 缺口（待全量补齐）\n")
md.append("① NoPred 主配置 8 折 selfMSE 全为 nan（探针挂在预测头内）——collect_diag 需对主配置单开 selfMSE 探针；② 不动点残差 ‖Z^(t+1)−Z^(t)‖ 逐迭代衰减曲线未存盘（Theorem 2 的 γ 几何收敛实测）——加一行采集；③ NoMutual/Base 均值逐位相同（33.99442546069622，见 jsonl v5_final_verdict 注记）待核验。")
open("results/invariance_panel.md", "w", encoding="utf-8").write("\n".join(md))
print("wrote results/invariance_panel.md")

# ---- figure: 2 valid panels ----
BLUE = "#2c6fbb"
fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
x = np.arange(len(FOLDS))

ax = axes[0]
ax.plot(x, [r["drift_cos"] for r in rows], "o-", color=BLUE, ms=6)
for xi, r in zip(x, rows):
    ax.annotate(f"{r['drift_cos']:.3f}", (xi, r["drift_cos"]), textcoords="offset points", xytext=(6, -4), fontsize=8)
ax.set_ylabel("driftCos"); ax.set_ylim(0.955, 1.0)
ax.set_title("cross-window representation drift (driftCos ~ 0.97+)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(x, [r["self_mse"] for r in rows], "s-", color=BLUE, ms=6)
for xi, r in zip(x, rows):
    ax.annotate(f"{r['self_mse']:.4f}", (xi, r["self_mse"]), textcoords="offset points", xytext=(6, 2), fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("selfMSE (log)")
ax.set_title("Z self-readout error (~1e-4: Z encodes coupling eigen)", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"fold {i+1}" for i in range(len(FOLDS))])
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("results/figures_v5/invariance_panel.png", dpi=150, bbox_inches="tight")
print("wrote results/figures_v5/invariance_panel.png")
