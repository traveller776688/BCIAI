#!/usr/bin/env python3
"""场域感知面板: 路由器 ω (场域亲和度) 的分离性证据
数据来源: results/diag_v5/C5-NoPred_dame_s*_seed*.npz (v5_router_loso / v5_router_xsess)
+ results/eeg_v5_router_{loso,xsess}_results.json (折级 acc)
输出: results/field_sensing_panel.md + results/figures_v5/field_sensing_panel.png
可证伪判决: LOSO(跨被试, 陌生场域) ω 应显著低于 跨会话(熟悉场域) ω
2026-08-14
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DDIR = "results/diag_v5"
LOSO_JSON = "results/eeg_v5_router_loso_results.json"
XSESS_JSON = "results/eeg_v5_router_xsess_results.json"
SEEDS = ["42", "123", "789"]

LOSO_RED = "#c0504d"
XSESS_GREEN = "#2e8b57"
GRAY = "#8c8c8c"


def load_folds(tag_prefix, json_path, seeds, session=False):
    """返回 [(label, acc, omega, d0)] 逐折(×种子)"""
    rows = []
    accs = json.load(open(json_path)) if os.path.exists(json_path) else {}
    for seed in seeds:
        if session:
            keys = [f"C5-NoPred_dame_sess3_seed{seed}"]
        else:
            keys = sorted(k for k in accs if k.startswith("C5-NoPred_dame_s")
                          and k.endswith(f"_seed{seed}"))
        for key in keys:
            npz = os.path.join(DDIR, f"{key}.npz")
            if not os.path.exists(npz):
                continue
            d = np.load(npz)
            try:
                omega = float(d["omega_mean"])
            except KeyError:
                continue
            try:
                d0 = float(d["omega_d0"])
            except KeyError:
                d0 = float("nan")
            rows.append({
                "key": key,
                "acc": accs.get(key, float("nan")),
                "omega": omega,
                "d0": d0,
            })
    return rows


def main():
    loso = load_folds("loso", LOSO_JSON, SEEDS)
    xsess = load_folds("xsess", XSESS_JSON, SEEDS, session=True)
    if not loso or not xsess:
        print(f"[field_panel] 数据不全 (loso={len(loso)}, xsess={len(xsess)}), 先跑完路由器验证")
        return

    loso_om = [r["omega"] for r in loso]
    xs_om = [r["omega"] for r in xsess]
    sep = np.mean(xs_om) - np.mean(loso_om)

    # ---- markdown ----
    md = []
    md.append("# 场域感知面板（路由器 ω 分离性，2026-08-14）\n")
    md.append("路由器 = 快慢通路失配度: 慢参考 = 训练集每脑区耦合强度指纹 EMA；快读数 = 当前样本耦合指纹；")
    md.append("ω = σ(a·(d₀−d))，d₀ 在训练分布 95 分位标定（无测试泄漏）。判决：跨被试（陌生脑袋）ω 应显著低，跨会话（同一颗脑袋的另一天）ω 应高。\n")
    md.append("## 逐折 ω\n")
    md.append("| 协议 | 折(种子) | ω | d₀ | acc% |")
    md.append("|---|---|---|---|---|")
    for r in loso:
        md.append(f"| LOSO(跨被试) | {r['key']} | {r['omega']:.3f} | {r['d0']:.3f} | {r['acc'] * 100:.2f} |")
    for r in xsess:
        md.append(f"| 跨会话 | {r['key']} | {r['omega']:.3f} | {r['d0']:.3f} | {r['acc'] * 100:.2f} |")
    md.append("")
    md.append(f"**分离性**: 跨会话 ω 均值 − LOSO ω 均值 = **{sep:+.3f}** "
              f"(LOSO {np.mean(loso_om):.3f}±{np.std(loso_om):.3f} vs 跨会话 {np.mean(xs_om):.3f}±{np.std(xs_om):.3f})。")
    verdict = "✓ 分离成立" if sep > 0.1 else ("△ 弱分离" if sep > 0 else "✗ 未分离（路由器被证伪）")
    md.append(f"**判决**: {verdict}（判据: 分离度 > +0.10）。\n")
    md.append("## 读法\n")
    md.append("- ω = 模型自感"这是我熟悉的场域吗"——0 = 陌生（社会沉默，走水循环不变载体），1 = 熟悉（社会策略登场）；")
    md.append("- d₀ 逐折不同（每折训练分布不同），但 ω 已把 d 归一成可比的亲和度；")
    md.append("- 跨被试 6 折 × 3 种子的 ω 全部应低于跨会话——不是协议标签在选路，是耦合统计在选路。")
    open("results/field_sensing_panel.md", "w", encoding="utf-8").write("\n".join(md))
    print("wrote results/field_sensing_panel.md")

    # ---- figure: 2 panels ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    ax = axes[0]
    xs = np.arange(len(loso))
    ax.bar(xs - 0.2, loso_om, 0.38, color=LOSO_RED, label="LOSO (cross-subject)")
    xs2 = np.arange(len(xsess))
    ax.bar(xs2 + 0.2, xs_om, 0.38, color=XSESS_GREEN, label="cross-session")
    ax.axhline(0.5, color=GRAY, ls="--", lw=1)
    ax.set_xticks(np.arange(max(len(loso), len(xsess))))
    ax.set_xticklabels([f"s{seed}" for seed in SEEDS[:max(len(loso), len(xsess))]])
    ax.set_ylabel("omega (field affinity)")
    ax.set_ylim(0, 1.05)
    ax.set_title("omega per fold: LOSO low vs cross-session high", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.scatter(loso_om, [r["acc"] * 100 for r in loso], color=LOSO_RED, label="LOSO", s=28)
    ax.scatter(xs_om, [r["acc"] * 100 for r in xsess], color=XSESS_GREEN, label="cross-session", s=28)
    ax.axvline(0.5, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("omega (field affinity)")
    ax.set_ylabel("acc (%)")
    ax.set_title("omega vs accuracy per fold", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("results/figures_v5/field_sensing_panel.png", dpi=150, bbox_inches="tight")
    print("wrote results/figures_v5/field_sensing_panel.png")


if __name__ == "__main__":
    main()
