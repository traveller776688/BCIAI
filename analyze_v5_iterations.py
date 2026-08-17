#!/usr/bin/env python3
"""
V5 迭代分析台 — 两条腿:
  1. --summary    : 读 results/v5_iterations.jsonl → 各轮进化表 + 消融增量轨迹 (回溯)
  2. --diag TAG ARM: 读 results/diag_v5/{ARM}_dame_s*_seed*.npz → 论文级诊断图:
     a) 每类情绪PLV耦合矩阵 (DMN前后枢纽PFC↔CP耦合随情绪重组)
     b) 策略指纹跨折一致性 (专家涌现/策略不动如山)
     c) 社区涌现 → 脑区→功能网络分配表
     d) 预迁移: 跨trial情绪稳定性预测 (stab_acc vs 真实保持率 + 每类稳定性 + 耦合漂移)
输出: results/figures_v5/*.png + stdout 摘要
"""
import json, os, glob, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
FIG = os.path.join(RES, "figures_v5")
os.makedirs(FIG, exist_ok=True)

REGION_NAMES = ['PFC', 'FL', 'FR', 'FC', 'C', 'TL', 'TR', 'CP', 'P', 'PO', 'O', 'CB']
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']
DMN_FRONT = 0   # PFC
DMN_BACK = 7    # CP (PCC)

def make_pairs(R):
    return [(i, j) for i in range(R) for j in range(i + 1, R)]

def summary():
    path = os.path.join(RES, "v5_iterations.jsonl")
    if not os.path.exists(path):
        print("no iterations yet"); return
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"{'iter':<12} {'subj':>4} {'seed':>4} {'ep':>3} | " +
          " | ".join(f"{m:<14}" for m in rows[-1]["models"] if m in rows[-1]["models"]) )
    # header: use union of model names in order of first appearance
    order = []
    for r in rows:
        for m in r["models"]:
            if m not in order:
                order.append(m)
    hdr = " ".join(f"{m:>13}" for m in order)
    print(f"{'tag':<12} {'cfg':>12} |{hdr} | verdict")
    print("-" * (40 + 14 * len(order)))
    for r in rows:
        cells = []
        for m in order:
            s = r["models"].get(m)
            cells.append(f"{s['mean']*100:>6.2f}+/-{s['std']*100:<5.1f}" if s else f"{'--':>13}")
        cfg = f"{r['subjects']}s{r['seeds']}x{r['epochs']}e"
        print(f"{r['tag']:<12} {cfg:>12} |" + " ".join(cells) + f" | {r['verdict']}")

def diag(tag, arm, seed=None, folds=None):
    files = sorted(glob.glob(os.path.join(RES, "diag_v5", f"{arm}_dame_s*_seed*.npz")))
    if seed is not None:
        files = [f for f in files if f"_seed{seed}.npz" in f]
    if folds is not None:
        allowed = {f"_s{s}_" for s in folds}
        files = [f for f in files if any(a in f for a in allowed)]
    if not files:
        print(f"no diag for {arm}"); return
    print(f"{arm}: {len(files)} fold-diag files")
    P = 66
    pairs = make_pairs(12)
    # --- a) per-class PLV matrix (mean over Tc) ---
    plv_all = []      # (fold, class, F, P)
    gates_all = []    # (fold, N)
    comms_all = []    # (fold, N)
    stab_rows = []    # (fold, stab_acc, stab_prior)
    stab_pc = []      # (fold, 4)
    drift_cos = []    # (fold,)
    for f in files:
        d = np.load(f)
        plv_all.append(d["plv_per_class"].mean(-1))
        gates_all.append(d["gates_mean"])
        if d["community_ids"] is not None:
            comms_all.append(d["community_ids"])
        if "stab_acc" in d and np.isfinite(d["stab_acc"]):
            stab_rows.append((d["stab_acc"], d["stab_prior"]))
            stab_pc.append(d["stab_per_class"])
        if "cross_trial_drift_cos" in d and np.isfinite(d["cross_trial_drift_cos"]):
            drift_cos.append(d["cross_trial_drift_cos"])
    plv_all = np.stack(plv_all)          # (FOLD, 4, 5, P)
    gates_all = np.stack(gates_all)      # (FOLD, N)

    # a1) 每类情绪的PLV邻接 (跨折平均, 每频段)
    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    for c in range(4):
        for f in range(5):
            adj = np.zeros((12, 12))
            vals = plv_all[:, c, f, :].mean(0)
            for k, (i, j) in enumerate(pairs):
                adj[i, j] = adj[j, i] = vals[k]
            im = axes[c, f].imshow(adj, cmap="magma", vmin=0, vmax=0.5)
            axes[c, f].set_xticks(range(12)); axes[c, f].set_xticklabels(REGION_NAMES, fontsize=7, rotation=60)
            axes[c, f].set_yticks(range(12)); axes[c, f].set_yticklabels(REGION_NAMES, fontsize=7)
            if c == 0:
                axes[c, f].set_title(BAND_NAMES[f])
            if f == 0:
                axes[c, f].set_ylabel(CLASS_NAMES[c])
            # DMN前后枢纽连线标出
            axes[c, f].add_patch(plt.Rectangle((DMN_BACK - .5, DMN_FRONT - .5), 1, 1,
                                               fill=False, edgecolor='lime', lw=2))
    plt.suptitle(f"{arm}@{tag}: 每类情绪脑区PLV耦合矩阵 (跨{len(files)}折平均)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"{tag}_{arm}_plv_per_class.png"), dpi=110)
    plt.close()

    # a2) DMN前后枢纽(PFC↔CP)耦合 随情绪/频段
    k_dmn = pairs.index((DMN_FRONT, DMN_BACK))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(5)
    w = 0.18
    for c in range(4):
        mu = plv_all[:, c, :, k_dmn].mean(0)
        sd = plv_all[:, c, :, k_dmn].std(0)
        ax.bar(x + (c - 1.5) * w, mu, w, yerr=sd, label=CLASS_NAMES[c], capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("PLV (PFC↔CP)")
    ax.set_title(f"{arm}@{tag}: DMN前后枢纽耦合随情绪×频段重组")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"{tag}_{arm}_dmn_frontback.png"), dpi=110)
    plt.close()

    # --- b) 策略指纹跨折一致性 ---
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(gates_all, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("脑区专家 #"); ax.set_ylabel("折")
    ax.set_xticks(range(12)); ax.set_xticklabels(REGION_NAMES)
    ax.set_title(f"{arm}@{tag}: 门控指纹跨折 (稳定列=策略不动如山)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"{tag}_{arm}_gate_fingerprint.png"), dpi=110)
    plt.close()
    # 一致性数值: 各专家门控均值跨折的变异系数 / 两两相关
    C = np.corrcoef(gates_all.T)
    off = ~np.eye(C.shape[0], dtype=bool)
    print(f"  gates: mean={gates_all.mean():.3f} | 跨折两两相关均值={C[off].mean():.3f}")
    print(f"  专家活跃度: " + " ".join(f"{REGION_NAMES[i]}:{gates_all[:, i].mean():.2f}"
                                      for i in range(12)))

    # --- c) 社区涌现 → 脑区→功能网络 ---
    if comms_all:
        print("  社区分配 (各折):")
        for fi, cm in enumerate(comms_all):
            groups = {}
            for r, cid in enumerate(cm):
                groups.setdefault(int(cid), []).append(REGION_NAMES[r])
            print(f"    fold{fi}: " + " | ".join(f"G{k}:{','.join(v)}" for k, v in groups.items()))
        cm = np.stack(comms_all)   # (FOLD, R)
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(cm) + 1)))
        ncomm = int(cm.max()) + 1
        im = ax.imshow(cm, aspect="auto", cmap="tab10", vmin=0, vmax=max(ncomm - 1, 1))
        ax.set_xticks(range(12)); ax.set_xticklabels(REGION_NAMES)
        ax.set_yticks(range(len(cm)))
        ax.set_yticklabels([f"fold{i + 1}" for i in range(len(cm))])
        for i in range(len(cm)):
            for j in range(12):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        fontsize=7, color="white")
        ax.set_title(f"{arm}@{tag}: 社区涌现 (脑区→功能网络, 自发分配)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"{tag}_{arm}_communities.png"), dpi=110)
        plt.close()

    # --- d) 预迁移诊断: 跨trial情绪稳定性预测 (可证伪: vs 真实保持率) ---
    if stab_rows:
        sa = np.array(stab_rows)
        print(f"  稳定性预测: stab={sa[:, 0].mean()*100:.1f}% vs prior(真实保持率)={sa[:, 1].mean()*100:.1f}% "
              f"(超prior折数 {int((sa[:, 0] > sa[:, 1]).sum())}/{len(sa)})")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(range(len(sa)), sa[:, 0] * 100, color="tab:blue", label="stab_acc")
        axes[0].plot(range(len(sa)), sa[:, 1] * 100, "ro-", label="prior(真实保持率)")
        axes[0].axhline(50, ls="--", c="gray", lw=1)
        axes[0].set_xlabel("fold"); axes[0].set_ylabel("%")
        axes[0].set_title("跨trial情绪稳定性预测 vs 基线")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        if stab_pc:
            pc = np.stack(stab_pc)
            axes[1].bar(range(4), pc.mean(0) * 100,
                        yerr=pc.std(0) * 100, capsize=4, color="tab:orange")
            axes[1].set_xticks(range(4)); axes[1].set_xticklabels(CLASS_NAMES)
            axes[1].axhline(50, ls="--", c="gray", lw=1)
            axes[1].set_ylabel("%"); axes[1].set_title("每类情绪的稳定性预测准确率")
            axes[1].grid(alpha=0.3)
        plt.suptitle(f"{arm}@{tag}: 预迁移 — 情绪时序稳定性预测")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"{tag}_{arm}_stability_pred.png"), dpi=110)
        plt.close()
    if drift_cos:
        print(f"  跨trial耦合漂移参考: driftCos={np.mean(drift_cos):.3f} "
              f"(情绪状态切换时的耦合重组幅度, DMN文献对照)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--diag", type=str, default=None, help="ARM name, e.g. DAME-C5")
    ap.add_argument("--tag", type=str, default="v5_iter2")
    ap.add_argument("--seed", type=int, default=None, help="只统计该种子的折文件")
    ap.add_argument("--folds", type=str, default=None, help="逗号分隔折号, 如 2,3,4,5,6,7")
    args = ap.parse_args()
    if args.summary:
        summary()
    if args.diag:
        folds = [int(x) for x in args.folds.split(",")] if args.folds else None
        diag(args.tag, args.diag, seed=args.seed, folds=folds)
