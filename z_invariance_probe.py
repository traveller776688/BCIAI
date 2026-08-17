#!/usr/bin/env python3
"""
Z 不变性直探针 (2026-08-17) — "大气层"假设的实证检验
====================================================
问题: 水循环蒸馏出的情绪本征 Z (K=32) 是否真的不携带场域身份 (被试/会话)?
      论文 III-D 把 Z 定义为"场不变载体"——这是建模赋值, 从未直接测量。
      本探针补上这块砖, 使论文从设计陈述升级为实证闭环。

协议: 与论文完全一致的 LOSO 管线 (复用 eeg_v5_coupling_experiment 的训练原语,
      逐字复刻 loso_v5 的训练协议: 同种子/同优化器/同调度器/同 set_epoch)。
      每个留出被试的窗口, 其 Z 提取自"从未见过该被试"的模型 — 这正是部署场景。

四个读数 (判读准则预先登记, 防止事后解释):
  1. 被试身份线性探针 (N类, 机会=1/N):   指纹 vs Z_pre/Z_init/Z_star
  2. 会话身份线性探针 (3类, 机会=33.3%): 同上
  3. 情绪本征探针 (4类, 机会=25%):       Z 是否保留情绪信息 (有效性控制 —
     Z 若连情绪都丢了, "不变性"就是空洞的: 把信息全丢了自然什么也解不出来)
  4. 跨会话漂移比: 留出被试会话间均值余弦距离 (指纹 vs Z) — "海陆差异"在 Z 空间是否缩小

判读:
  指纹可解码身份 且 Z_star ≈ 机会  → 大气层成立 (蒸馏脱域);
  Z_star 仍可解码身份              → 论文 III-D 措辞必须降级, 诚实记录;
  Z_pre 可解码 而 Z_star 不可      → 是训练蒸馏掉了身份信息 (机制定位成立, 非架构巧合)。

成本: 15被试 × 1种子 × 15epoch ≈ 40-75 分钟 (RTX 5070, 每折 127-290s)。
局限: 线性探针是保守下界 (非线性可能更强, 但线性不可解 → 按 Prop 2 的精神,
      部署方用任何恒等嵌入 f 都难以读到; 非线性留作扩展)。
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import eeg_v5_coupling_experiment as E

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEVICE = E.DEVICE


# =========================================================================
# 1. 指纹收集 — 输入侧场域身份载体 (与附录E指纹族同款: 边级PLV + 区级功率)
#    用全新的未训练 RegionCouplingV2 前端, 纯信号处理, 与训练模型无关
# =========================================================================
@torch.no_grad()
def collect_fingerprints(front, X_test, bs=64):
    """X_test: (n, 62, 800) → fp (n, 330+60), fp_edge (n,330), fp_pow (n,60)"""
    edge_l, pow_l = [], []
    for i in range(0, X_test.size(0), bs):
        xb = X_test[i:i + bs].to(DEVICE)
        _, _, plv = front(xb)                                  # (B, F, P, Tc)
        edge = plv.mean(-1).transpose(1, 2).reshape(plv.size(0), -1)
        reg = torch.stack([xb[:, g].mean(1) for g in E.REGION_GROUPS], dim=1)
        pw = []
        for (lo, hi) in E.BANDS.values():
            bb = E.bandpass_fft(reg, lo, hi)
            pw.append((bb ** 2).mean(-1))
        powf = torch.log1p(torch.stack(pw, 1)).reshape(xb.size(0), -1)
        edge_l.append(edge.cpu())
        pow_l.append(powf.cpu())
    edge = torch.cat(edge_l).numpy()
    powf = torch.cat(pow_l).numpy()
    return np.concatenate([edge, powf], axis=1), edge, powf


@torch.no_grad()
def collect_z(model, X_test, bs=256):
    """eval模式下提取 (Z_star, Z_init) — 确定性 (μ 无采样噪声)."""
    zs, zi = [], []
    model.eval()
    for i in range(0, X_test.size(0), bs):
        out = model(X_test[i:i + bs].to(DEVICE))
        zs.append(out["Z_star"].cpu())
        zi.append(out["Z_init"].cpu())
    return torch.cat(zs).numpy(), torch.cat(zi).numpy()


# =========================================================================
# 2. LOSO 收集 — 训练协议逐字复刻 loso_v5, 额外收集每折测试窗的表示
# =========================================================================
def loso_collect(X, y, subj, wid, pair_idx, seed, epochs):
    Ns = int(subj.max().item()) + 1
    front = E.RegionCouplingV2(E.REGION_GROUPS).to(DEVICE)
    front.eval()

    rep = {k: [] for k in
           ["fp", "fp_edge", "fp_pow", "Z_pre", "Z_init_pre", "Z_star", "Z_init"]}
    y_l, subj_l, sess_l = [], [], []
    fold_accs = {}

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for s in range(Ns):
        t0 = time.time()
        tm = subj == s
        trm = ~tm
        X_train, y_train = X[trm], y[trm]
        X_test, y_test = X[tm], y[tm]

        train_glob = torch.nonzero(trm).flatten().tolist()
        g2l = {g: li for li, g in enumerate(train_glob)}
        pair_map = {g2l[a]: g2l[b] for a, b in pair_idx.tolist()
                    if a in g2l and b in g2l}

        plv_adj = E.compute_plv_adjacency(E.REGION_GROUPS, X_train)
        model = E.DAME_Coupling(plv_adj=plv_adj).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=E.LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2)

        # 输入指纹 (模型无关) + 未训练模型的 Z (架构泄漏基线)
        fp, fpe, fpp = collect_fingerprints(front, X_test)
        zpre, zipre = collect_z(model, X_test)

        # 训练 (与 loso_v5 逐字一致: 看门狗/set_epoch/train_epoch_dame/sched)
        for ep in range(epochs):
            try:
                import faulthandler as _fh
                _fh.cancel_dump_traceback_later()
                _fh.dump_traceback_later(600, exit=True)
            except Exception:
                pass
            model.set_epoch(ep, epochs)
            loss = E.train_epoch_dame(model, X_train, y_train, pair_map, opt)
            sched.step()

        model.mutual.calibrate_omega()   # field_router 关闭时早退, 与主线一致
        acc, _ = E.evaluate(model, X_test, y_test, is_dame=True)
        zstar, zinit = collect_z(model, X_test)

        # 落盘本折
        rep["fp"].append(fp); rep["fp_edge"].append(fpe); rep["fp_pow"].append(fpp)
        rep["Z_pre"].append(zpre); rep["Z_init_pre"].append(zipre)
        rep["Z_star"].append(zstar); rep["Z_init"].append(zinit)
        y_l.append(y_test.cpu().numpy())
        subj_l.append(subj[tm].cpu().numpy())
        sess_l.append(wid[tm][:, 1].cpu().numpy())
        fold_accs[f"s{s}"] = acc
        print(f"  [{s + 1}/{Ns}] acc={acc:.4f} n_test={X_test.size(0)} "
              f"({time.time() - t0:.0f}s)", flush=True)

        del model, opt
        torch.cuda.empty_cache()

    del front
    torch.cuda.empty_cache()
    out = {k: np.concatenate(v) for k, v in rep.items()}
    out["y"] = np.concatenate(y_l)
    out["subj"] = np.concatenate(subj_l)
    out["sess"] = np.concatenate(sess_l)
    return out, fold_accs


# =========================================================================
# 3. 线性探针
# =========================================================================
def probe_acc(Xrep, labels, groups, n_splits, seed=0):
    """两种划分: i.i.d.分层 (池内可解码性) + 按被试留出 (身份模式可迁移性)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1.0, class_weight="balanced", tol=1e-3))
    res = {}
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Xrep, labels):
        clf.fit(Xrep[tr], labels[tr])
        accs.append(float(clf.score(Xrep[te], labels[te])))
    res["iid"] = {"folds": accs,
                  "mean": float(np.mean(accs)),
                  "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0}
    try:
        gkf = GroupKFold(n_splits)
        accs = []
        for tr, te in gkf.split(Xrep, labels, groups=groups):
            clf.fit(Xrep[tr], labels[tr])
            accs.append(float(clf.score(Xrep[te], labels[te])))
        res["group"] = {"folds": accs,
                        "mean": float(np.mean(accs)),
                        "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0}
    except ValueError:
        res["group"] = None
    return res


def cross_session_drift(rep, subj, sess):
    """每个被试的会话间均值余弦距离 (至少4窗的会话才计入)."""
    dists = []
    for s in np.unique(subj):
        cells = []
        for ss in [1, 2, 3]:
            m = (subj == s) & (sess == ss)
            if m.sum() >= 4:
                cells.append(rep[m].mean(0))
        if len(cells) < 2:
            continue
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a = cells[i] / (np.linalg.norm(cells[i]) + 1e-8)
                b = cells[j] / (np.linalg.norm(cells[j]) + 1e-8)
                dists.append(float(1.0 - float(a @ b)))
    return dists


def drift_stats(dists):
    return {"mean": float(np.mean(dists)), "median": float(np.median(dists)),
            "min": float(np.min(dists)), "max": float(np.max(dists)),
            "n_pairs": len(dists)}


# =========================================================================
# 4. Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=E.EPOCHS)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 2被试 2epoch, 验证端到端")
    args = ap.parse_args()
    if args.smoke:
        args.subjects, args.epochs = 2, 2
        print("[SMOKE] 2 subjects, 2 epochs")

    t0 = time.time()
    X, y, subj, pair_idx, wid = E.load_raw_with_pairs(n_subjects=args.subjects)
    print(f"[probe] data: {X.shape[0]} windows, {subj.unique().numel()} subjects, "
          f"{time.time() - t0:.0f}s", flush=True)

    rep, fold_accs = loso_collect(X, y, subj, wid, pair_idx, args.seed, args.epochs)
    n = len(rep["y"])
    print(f"[probe] collected: {n} windows "
          f"(fold accs mean={np.mean(list(fold_accs.values())):.4f})", flush=True)

    n_subj = int(rep["subj"].max()) + 1
    reps_subj = ["fp", "fp_edge", "fp_pow", "Z_pre", "Z_init_pre", "Z_star", "Z_init"]
    reps_sess = reps_subj
    reps_emo = ["Z_pre", "Z_init_pre", "Z_star", "Z_init"]
    nsplits = min(5, n_subj)
    if n_subj < 3:
        nsplits = 2

    result = {
        "protocol": {"type": "LOSO", "subjects": n_subj, "seed": args.seed,
                     "epochs": args.epochs, "model": "DAME-C5",
                     "norm": "per-subject all-session (LOSO standard)"},
        "n_windows": n,
        "class_counts": {int(c): int((rep["y"] == c).sum()) for c in np.unique(rep["y"])},
        "subject_counts": {int(c): int((rep["subj"] == c).sum())
                           for c in np.unique(rep["subj"])},
        "fold_accs": fold_accs,
        "fold_acc_mean": float(np.mean(list(fold_accs.values()))),
        "chance": {"subject": 1.0 / n_subj, "session": 1.0 / 3, "emotion": 0.25},
        "subject_probe": {}, "session_probe": {}, "emotion_probe": {},
        "cross_session_drift": {},
    }

    # 被试身份探针 (主检验)
    print("\n=== 被试身份线性探针 ===", flush=True)
    for r in reps_subj:
        p = probe_acc(rep[r], rep["subj"], rep["subj"], nsplits)
        result["subject_probe"][r] = p
        iid = p["iid"]
        g = p["group"]["mean"] if p["group"] else float("nan")
        print(f"  {r:<10} iid={iid['mean']:.3f}±{iid['std']:.3f} "
              f"group={g:.3f} (chance={1.0 / n_subj:.3f})", flush=True)

    # 会话身份探针
    print("\n=== 会话身份线性探针 ===", flush=True)
    for r in reps_sess:
        p = probe_acc(rep[r], rep["sess"], rep["subj"], nsplits)
        result["session_probe"][r] = p
        iid = p["iid"]
        g = p["group"]["mean"] if p["group"] else float("nan")
        print(f"  {r:<10} iid={iid['mean']:.3f}±{iid['std']:.3f} "
              f"group={g:.3f} (chance={1.0 / 3:.3f})", flush=True)

    # 情绪本征探针 (有效性控制: Z 必须保留情绪)
    print("\n=== 情绪本征探针 (有效性控制) ===", flush=True)
    for r in reps_emo:
        p = probe_acc(rep[r], rep["y"], rep["subj"], nsplits)
        result["emotion_probe"][r] = p
        iid = p["iid"]
        g = p["group"]["mean"] if p["group"] else float("nan")
        print(f"  {r:<10} iid={iid['mean']:.3f}±{iid['std']:.3f} "
              f"group={g:.3f} (chance={0.25:.3f})", flush=True)

    # 跨会话漂移比 ("海陆差异"在 Z 空间是否缩小)
    print("\n=== 跨会话漂移 (会话间均值余弦距离) ===", flush=True)
    for r in ["fp", "Z_pre", "Z_init_pre", "Z_star", "Z_init"]:
        d = cross_session_drift(rep[r], rep["subj"], rep["sess"])
        st = drift_stats(d)
        result["cross_session_drift"][r] = st
        print(f"  {r:<10} mean={st['mean']:.4f} median={st['median']:.4f} "
              f"(n={st['n_pairs']})", flush=True)
    if result["cross_session_drift"]["fp"]["mean"] > 1e-9:
        ratio = (result["cross_session_drift"]["Z_star"]["mean"]
                 / result["cross_session_drift"]["fp"]["mean"])
        result["drift_ratio_Z_star_over_fp"] = ratio
        print(f"\n  Z_star/fp 漂移比 = {ratio:.3f}  "
              f"({'<1: Z 空间海陆差异缩小' if ratio < 1 else '>=1: 未缩小'})",
              flush=True)

    # 落盘
    np.savez(os.path.join(E.RESULTS_DIR, "diag_v5",
                          f"zprobe_arrays_{n_subj}s_seed{args.seed}.npz"),
             fp=rep["fp"], fp_edge=rep["fp_edge"], fp_pow=rep["fp_pow"],
             Z_pre=rep["Z_pre"], Z_init_pre=rep["Z_init_pre"],
             Z_star=rep["Z_star"], Z_init=rep["Z_init"],
             y=rep["y"], subj=rep["subj"], sess=rep["sess"])
    out_path = os.path.join(E.RESULTS_DIR, "z_invariance_probe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {out_path} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
