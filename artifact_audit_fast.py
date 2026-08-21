# -*- coding: utf-8 -*-
"""
伪迹剔除快速审计 (2026-08-20, 论文 VII(4) 登记的伪迹审计第一步)
========================================================
设计 (cheapest-first, 用户指令: 分钟级 + GPU):
  L1 数据级 (GPU分块PLV, 分钟): 复现 VI-E P1/P2/P3, 对比清洗前后
      P1 = PFC–CP 高唤醒(恐惧2+快乐3) vs 低唤醒(中性0+悲伤1), 5频段, t(3)
      P2 = PFC–CP 恐惧(2) vs 中性(0) α段
      P3 = 全66对全频段 PLV 全局均值 (结构守恒)
  L2 分类级 (GPU, ~10min): 4被试 LOSO × 1种子 × 15轮
      C5-NoPred (论文主配置) + EEGNet × {原始, 清洗} 共4臂

清洗管道 (与主管线 eeg_v5_coupling_experiment.clean_subject_trials 同一实现,
2026-08-20 起主管线 --artifact-clean 复用本审计的管道):
  800Hz 原始 → 1-50Hz 带通 (firwin) → decimate(4)→200Hz
  → ICA(fastica, 会话1拟合, decim=3) → ICLabel 剔除 (argmax≠brain 且置信≥0.5;
  裸 argmax 会删 20+/30 组件=过度清洗, 第三轮教训) [ICLabel 失败回退 EOG 相关启发式]
  → 应用到3会话 → 逐被试 z-score (与主代码同口径)
  → 同窗口切片 (800样本/400步长) — 窗口结构逐一对齐 X_raw

预注册判读:
  清洗生效: 每被试剔除组件 ≥ 0 且全组剔除方差占比 > 5% (否则清洗无作用, 报告作废)
  P1 存活: β段清洗后同号 (方向一致)
  P2 存活: α段恐惧-中性同号
  P3 守恒: 全局均值变动 |Δ| < 0.05
  L2: |Δclean| ≤ 2点 = 无实质变化; 跌>2 = 伪迹泄漏嫌疑; 升>2 = 伪迹噪声假设
不污染: 清洗实现已上收主管线, 审计脚本本身只保留数据级对比与判读; 结果落
  results/artifacts_audit/
"""
import os, sys, json, time, traceback
import numpy as np
import scipy.stats as st
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import eeg_v5_coupling_experiment as m          # 复用 bandpass_fft/hilbert_analytic/plv_coupling/loso_v5/clean_subject_trials

AUDIT_DIR = os.path.join(m.RESULTS_DIR, "artifacts_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

N_SUBJ = 4
SEEDS = [42]
EPOCHS = 15
EMO = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
PFC_IDX, CP_IDX = 0, 7                          # REGION_NAMES 顺序: PFC=0, CP=7 (DMN 前后枢纽)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

report = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "n_subjects": N_SUBJ,
          "seeds": SEEDS, "epochs": EPOCHS, "cleaning": {}}


# =====================================================================
# 1. 读原始 800Hz (清洗实现本身在主管线 m.clean_subject_trials, 单一来源)
# =====================================================================
def load_trials_800(subj_id):
    """读 3 会话 × 24 trial 的 800Hz 原始, 返回 [(62, T) float64] × 72"""
    from scipy.io import loadmat
    trials = []
    for session in [1, 2, 3]:
        sess_dir = os.path.join(m.RAW_DIR, str(session))
        fname = next((f for f in os.listdir(sess_dir)
                      if f.startswith(f"{subj_id}_") and f.endswith(".mat")), None)
        data = loadmat(os.path.join(sess_dir, fname))
        for t in range(24):
            var = next((k for k in data
                        if not k.startswith("__") and k.endswith(f"_eeg{t + 1}")), None)
            trials.append(np.nan_to_num(data[var]).astype(np.float64))
    return trials


# =====================================================================
# 2. 窗口切片 (与主代码 load_raw_with_pairs 逐字同口径) + z-score
# =====================================================================
def window_and_zscore(trials_200hz):
    """trials_200hz: 每被试 72 trial (62, T@200Hz) → 逐被试 z-score → 窗口切片
    与 eeg_v5_coupling_experiment.load_raw_with_pairs 完全一致 (4s/2s hop)"""
    all_X, all_y, all_subj = [], [], []
    for subj_id, trials in enumerate(trials_200hz):
        cat = np.concatenate(trials, axis=1)
        mu = cat.mean(axis=1, keepdims=True)
        sd = cat.std(axis=1, keepdims=True) + 1e-8
        labels = [m.SESSION_LABELS[s][t] for s in [1, 2, 3] for t in range(24)]
        for ds, lab in zip(trials, labels):
            ds = (ds - mu) / sd
            T = ds.shape[1]
            for stp in range(0, T - m.WINDOW_SAMPLES + 1, m.WINDOW_STRIDE):
                all_X.append(torch.from_numpy(ds[:, stp:stp + m.WINDOW_SAMPLES].copy()))
                all_y.append(lab)
                all_subj.append(subj_id)
    return torch.stack(all_X), torch.tensor(all_y, dtype=torch.long), \
        torch.tensor(all_subj, dtype=torch.long)


# =====================================================================
# 3. 数据级 P1-P3 (GPU 分块, 复用模块 PLV 函数, 与 VI-E 同口径)
# =====================================================================
def pfc_cp_plv(X, y, subj, chunk=512):
    """每窗口 PFC–CP PLV × 5频段 → (N, 5)。区域均值 → bandpass → Hilbert → cos → 0.5s 子窗均值"""
    N = X.shape[0]
    out = torch.zeros(N, m.N_BANDS, dtype=torch.float32)
    pfcc = torch.tensor(m.REGION_GROUPS[PFC_IDX], dtype=torch.long)
    cpc = torch.tensor(m.REGION_GROUPS[CP_IDX], dtype=torch.long)
    for i in range(0, N, chunk):
        Xb = X[i:i + chunk].to(DEV)
        rp = Xb[:, pfcc].mean(1)                        # 通道维均值 → (B, 800)
        rc = Xb[:, cpc].mean(1)
        for b, (lo, hi) in enumerate(m.BANDS.values()):
            ph_p = torch.angle(m.hilbert_analytic(m.bandpass_fft(rp, lo, hi)))
            ph_c = torch.angle(m.hilbert_analytic(m.bandpass_fft(rc, lo, hi)))
            d = torch.cos(ph_p - ph_c)                  # (B, 800)
            d = d.reshape(d.shape[0], m.T_COUP, m.SUB_WIN).mean(-1).mean(-1)
            out[i:i + chunk, b] = d.cpu()
        del Xb
        torch.cuda.empty_cache()
    return out.numpy()


def p1_p2_p3(X, y, subj):
    """复现 VI-E: 每被试聚合 → P1(高vs低唤醒×5频段 t(3)), P2(恐惧-中性 α), P3(全局均值)"""
    plv = pfc_cp_plv(X, y, subj)
    n_subj = int(subj.max()) + 1
    per_subj = {s: {c: {b: [] for b in range(m.N_BANDS)} for c in range(4)}
                for s in range(n_subj)}
    for s in range(n_subj):
        msk = subj.numpy() == s
        for c in range(4):
            for b in range(m.N_BANDS):
                per_subj[s][c][b] = plv[msk & (y.numpy() == c), b].mean()
    res = {}
    for b, (lo, hi) in enumerate(m.BANDS.values()):
        high = np.array([np.mean([per_subj[s][2][b], per_subj[s][3][b]]) for s in range(n_subj)])
        low = np.array([np.mean([per_subj[s][0][b], per_subj[s][1][b]]) for s in range(n_subj)])
        d = high - low
        res[f"P1_{m.BAND_NAMES[b]}"] = dict(mean_diff=float(d.mean()),
                                             t=float(st.ttest_rel(high, low)[0]),
                                             p=float(st.ttest_rel(high, low)[1]))
    fear = np.array([per_subj[s][2][m.BAND_NAMES.index("alpha")] for s in range(n_subj)])
    neu = np.array([per_subj[s][0][m.BAND_NAMES.index("alpha")] for s in range(n_subj)])
    res["P2_alpha_fear_neutral"] = dict(mean_diff=float((fear - neu).mean()),
                                         t=float(st.ttest_rel(fear, neu)[0]),
                                         p=float(st.ttest_rel(fear, neu)[1]))
    # P3: 全 66 对全频段 PLV 均值 (分块 plv_coupling)
    tot, cnt = 0.0, 0
    pi, pj = m.make_pairs(m.N_REGIONS)
    for i in range(0, X.shape[0], 256):
        Xb = X[i:i + 256].to(DEV)
        reg = Xb.new_zeros(Xb.shape[0], m.N_REGIONS, Xb.shape[2])
        for r, g in enumerate(m.REGION_GROUPS):           # 脑区通道数不等, 不能 torch.stack
            reg[:, r] = Xb[:, g].mean(1)
        plvb, _ = m.plv_coupling(reg, pi, pj)
        tot += float(plvb.mean().cpu())
        cnt += 1
        del Xb
        torch.cuda.empty_cache()
    res["P3_global_mean_plv"] = tot / cnt
    return res


# =====================================================================
# 4. 主流程
# =====================================================================
def main():
    print("=" * 68)
    print("伪迹剔除快速审计: L1 数据级 P1-P3 + L2 4被试 LOSO 分类")
    print(f"清洗: 1-50Hz带通 + ICA(fastica, 会话1拟合) + ICLabel; 规模: {N_SUBJ}被试")
    print("=" * 68, flush=True)

    # ---- 原始数据 (主代码口径: 降采样200+zscore+窗口) ----
    t0 = time.time()
    X_raw, y, subj, pair_idx, _ = m.load_raw_with_pairs(N_SUBJ)
    print(f"[data] X_raw {tuple(X_raw.shape)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 清洗 (实现=主管线 clean_subject_trials; 结果缓存 results/artifact_clean_cache/) ----
    t0 = time.time()
    cleaned_trials, subjects_report = [], {}
    for s in range(1, N_SUBJ + 1):
        tr = load_trials_800(s)
        cl, info = m.clean_subject_trials(s, tr)
        cleaned_trials.append(cl)                          # 已是 200Hz
        subjects_report[s] = info
        print(f"[clean] subj{s}: fitted={info['comps_fitted']} "
              f"rejected={info['comps_rejected']} var={info['var_rejected']:.1%} "
              f"({info['method']}) ({time.time() - t0:.0f}s)", flush=True)
    X_clean, y_clean, subj_clean = window_and_zscore(cleaned_trials)
    assert X_clean.shape == X_raw.shape, (X_clean.shape, X_raw.shape)
    assert torch.equal(y_clean, y) and torch.equal(subj_clean, subj)
    report["cleaning"]["subjects"] = subjects_report
    print(f"[clean] X_clean {tuple(X_clean.shape)}, 窗口结构与 X_raw 逐一对齐 ✓", flush=True)

    # ---- L1 数据级 ----
    t0 = time.time()
    res_raw = p1_p2_p3(X_raw, y, subj)
    res_clean = p1_p2_p3(X_clean, y, subj)
    report["L1_raw"] = res_raw
    report["L1_clean"] = res_clean
    print(f"[L1] P1-P3 完成 ({time.time() - t0:.0f}s)", flush=True)

    # ---- L2 分类 (GPU) ----
    results = {}
    def save_fn(key, acc):
        results[key] = acc
        with open(os.path.join(AUDIT_DIR, "audit_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    arms = [("C5-NoPred", m.make_variant(use_pred=False), "dame"),
            ("EEGNet", m.MODEL_SPECS["EEGNet"]["fn"], "plain")]
    for cond, X in [("raw", X_raw), ("clean", X_clean)]:
        for aname, fac, kind in arms:
            print(f"\n[L2] {aname} @ {cond} ({N_SUBJ}subj x 1seed x {EPOCHS}ep)", flush=True)
            t1 = time.time()
            m.loso_v5(fac, X, y, subj, pair_idx, SEEDS, epochs=EPOCHS,
                      kind=kind, done_folds={}, save_fn=save_fn,
                      tag=f"audit_{cond}_{aname}")
            print(f"  [{aname}@{cond}] {time.time() - t1:.0f}s", flush=True)

    # ---- 裁决 ----
    print("\n" + "=" * 68)
    print("审计裁决 (预注册判读)")
    print("=" * 68)
    bn = m.BAND_NAMES
    print("\n[L1] P1 β段 (高唤醒−低唤醒): "
          f"raw {res_raw[f'P1_beta']['mean_diff']:+.5f} → "
          f"clean {res_clean[f'P1_beta']['mean_diff']:+.5f} "
          f"({'同号存活' if res_clean['P1_beta']['mean_diff'] * res_raw['P1_beta']['mean_diff'] > 0 else '翻号!'})")
    print(f"[L1] P2 α段 (恐惧−中性): "
          f"raw {res_raw['P2_alpha_fear_neutral']['mean_diff']:+.5f} → "
          f"clean {res_clean['P2_alpha_fear_neutral']['mean_diff']:+.5f} "
          f"({'同号存活' if res_clean['P2_alpha_fear_neutral']['mean_diff'] * res_raw['P2_alpha_fear_neutral']['mean_diff'] > 0 else '翻号!'})")
    dp3 = abs(res_clean["P3_global_mean_plv"] - res_raw["P3_global_mean_plv"])
    print(f"[L1] P3 全局均值: raw {res_raw['P3_global_mean_plv']:.4f} → "
          f"clean {res_clean['P3_global_mean_plv']:.4f} (|Δ|={dp3:.4f}, "
          f"{'守恒' if dp3 < 0.05 else '大幅变动!'})")
    print("\n[L2] 分类对比 (4被试 LOSO, 均值±sd):")
    for aname, _, _ in arms:
        for cond in ["raw", "clean"]:
            accs = [v for k, v in results.items()
                    if k.startswith(f"audit_{cond}_{aname}_dame_s")
                    or k.startswith(f"audit_{cond}_{aname}_plain_s")]
            if accs:
                a = np.array(accs)
                print(f"    {aname:<10} {cond:<6} {a.mean() * 100:.2f}±{a.std(ddof=1) * 100:.2f}% (n={len(a)})")
        rr = [v for k, v in results.items() if f"audit_raw_{aname}" in k]
        cc = [v for k, v in results.items() if f"audit_clean_{aname}" in k]
        if rr and cc:
            d = (np.mean(cc) - np.mean(rr)) * 100
            verdict = ("无实质变化" if abs(d) <= 2 else
                       ("伪迹噪声假设(升>2点)" if d > 2 else "伪迹泄漏嫌疑(跌>2点)"))
            print(f"    Δ(clean−raw) = {d:+.2f} 点 → **{verdict}**")

    report["L2_results"] = results
    with open(os.path.join(AUDIT_DIR, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] {os.path.join(AUDIT_DIR, 'audit_report.json')}")


if __name__ == "__main__":
    main()
