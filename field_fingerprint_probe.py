#!/usr/bin/env python3
"""场域指纹可分离性探针 (2026-08-14)
问题: 路由器的快读数 (每区×频段耦合强度指纹, 60维) 能否区分
   (a) 不同脑袋 (跨被试)  vs  (b) 同一脑袋另一天 (跨会话)?
结果判读:
   若 同脑跨会话距离 ≈ 异脑距离  → 指纹粒度太粗, 路由器在此粒度上被证伪
   若 同脑跨会话距离 << 异脑距离 → 指纹可用, ω≈0.5 是标定/实现问题
与路由验证 (bobnhvvr5) 并发跑, 只用 CPU, 只取前6个被试, 每会话最多150窗
"""
import os
import torch
import numpy as np
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import eeg_v5_coupling_experiment as E

torch.set_num_threads(6)

N_SUBJ = 6
MAX_WIN = 150


def fingerprint(plv, pair_i, pair_j, R):
    """(B,F,P,Tc) → (B, F*R) 每区入射耦合强度, 与 _coup_strength 同式"""
    plv_mean = plv.mean(-1)                                   # (B, F, P)
    B, Fb, _ = plv_mean.shape
    out = plv_mean.new_zeros(B, Fb, R)
    out = out.index_add_(-1, pair_i, plv_mean)
    out = out.index_add_(-1, pair_j, plv_mean)
    return out.reshape(B, -1)


def main():
    X, y, subj, pair_idx, wid = E.load_raw_with_pairs(n_subjects=N_SUBJ)
    print(f"[probe] X={X.shape}  subjects={subj.unique().tolist()}", flush=True)

    front = E.RegionCouplingV2(E.REGION_GROUPS).to('cpu')
    R = front.R
    pi = torch.tensor(front.pair_i, dtype=torch.long)
    pj = torch.tensor(front.pair_j, dtype=torch.long)

    fp = {}   # (subj, session) -> list of fingerprints
    for sid in range(N_SUBJ):
        for sess in [1, 2, 3]:
            m = (subj == sid) & (wid[:, 1] == sess)
            if not m.any():
                continue
            idx = m.nonzero().flatten()[:MAX_WIN]
            fips = []
            for i in range(0, len(idx), 32):
                xb = X[idx[i:i + 32]]
                _, _, plv = front(xb)
                fips.append(fingerprint(plv, pi, pj, R))
            fp[(sid, sess)] = torch.cat(fips)
    print(f"[probe] fingerprints collected: {len(fp)} (subj,session) cells", flush=True)

    # 同脑跨会话距离 vs 异脑距离
    within = []
    between = []
    means = {}
    for k, v in fp.items():
        means[k] = F.normalize(v.mean(0), dim=0)
    for (sid, s1), m1 in means.items():
        for (oid, s2), m2 in means.items():
            d = float(1.0 - F.cosine_similarity(m1, m2, dim=0))
            if oid == sid and s1 != s2:
                within.append(d)
            elif oid != sid:
                between.append(d)
    print(f"[probe] same-brain cross-session d: mean={np.mean(within):.4f} "
          f"range=[{min(within):.4f},{max(within):.4f}] (n={len(within)})")
    print(f"[probe] cross-brain d:            mean={np.mean(between):.4f} "
          f"range=[{min(between):.4f},{max(between):.4f}] (n={len(between)})")

    # 组内窗口噪声 vs 组间差异 (信号噪声比)
    spread_w = np.mean([float(v.std(0).mean()) for v in fp.values()])
    spread_b = np.std([float(v.mean()) for v in means.values()])
    print(f"[probe] mean within-cell std={spread_w:.4f} vs between-cell mean spread={spread_b:.4f}")

    sep = np.mean(between) - np.mean(within)
    print(f"[probe] SEPARATION (cross-brain − same-brain) = {sep:+.4f}")
    if sep > 0.02:
        print("[probe] VERDICT: fingerprint CAN separate fields -> router issue is elsewhere")
    else:
        print("[probe] VERDICT: fingerprint CANNOT separate fields -> router falsified at this granularity (needs edge-level fingerprint)")

    # ---- v2 候选: 边级指纹 (66对 × 5频段) + 时间池化 ----
    print("[probe2] edge-level fingerprint (P*F=%d dims)..." % (len(pi) * plv.shape[1]), flush=True)
    edge_fp = {}
    with torch.no_grad():
        for sid in range(N_SUBJ):
            for sess in [1, 2, 3]:
                m = (subj == sid) & (wid[:, 1] == sess)
                if not m.any():
                    continue
                idx = m.nonzero().flatten()[:MAX_WIN]
                fips = []
                for i in range(0, len(idx), 32):
                    _, _, plv = front(X[idx[i:i + 32]])
                    fips.append(plv.mean(-1).transpose(1, 2).reshape(plv.size(0), -1))
                edge_fp[(sid, sess)] = torch.cat(fips)

    def sep_stats(fp, pool):
        within, between = [], []
        means = {}
        for k, v in fp.items():
            if pool > 1:
                n = v.size(0) // pool * pool
                v = v[:n].view(-1, pool, v.size(1)).mean(1)   # 池化 pool 窗
            means[k] = F.normalize(v.mean(0), dim=0)
        for (sid, s1), m1 in means.items():
            for (oid, s2), m2 in means.items():
                d = float(1.0 - F.cosine_similarity(m1, m2, dim=0))
                if oid == sid and s1 != s2:
                    within.append(d)
                elif oid != sid:
                    between.append(d)
        sw = np.mean([float(v.std(0).mean()) for v in means.values()])
        return np.mean(within), np.mean(between), sw

    for pool in [1, 8, 32]:
        w, b, sp = sep_stats(edge_fp, pool)
        print(f"[probe2] pool={pool:2d}: same-brain={w:.4f} cross-brain={b:.4f} "
              f"separation={b - w:+.4f} between-cell spread={sp:.4f}", flush=True)

    # ---- v3 候选: 功率指纹 (62ch × 5频段 log带功) — 经典个体特异性载体 ----
    print("[probe3] power fingerprint (62ch*5band log power)...", flush=True)
    pow_fp = {}
    with torch.no_grad():
        for sid in range(N_SUBJ):
            for sess in [1, 2, 3]:
                m = (subj == sid) & (wid[:, 1] == sess)
                if not m.any():
                    continue
                idx = m.nonzero().flatten()[:MAX_WIN]
                fips = []
                for i in range(0, len(idx), 32):
                    xb = X[idx[i:i + 32]]                       # (B, 62, T)
                    pw = []
                    for (lo, hi) in E.BANDS.values():
                        bb = E.bandpass_fft(xb, lo, hi)
                        pw.append((bb ** 2).mean(-1))           # (B, 62)
                    fips.append(torch.log1p(torch.stack(pw, 1)).reshape(xb.size(0), -1))
                pow_fp[(sid, sess)] = torch.cat(fips)

    for pool in [1, 8, 32]:
        w, b, sp = sep_stats(pow_fp, pool)
        print(f"[probe3] pool={pool:2d}: same-brain={w:.4f} cross-brain={b:.4f} "
              f"separation={b - w:+.4f} between-cell spread={sp:.4f}", flush=True)

    # ---- v3b: 12脑区功率指纹 (RegionCouplingV2 内部已有的 region 功率, 零额外开销) ----
    print("[probe3b] 12-region power fingerprint...", flush=True)
    regpow_fp = {}
    with torch.no_grad():
        for sid in range(N_SUBJ):
            for sess in [1, 2, 3]:
                m = (subj == sid) & (wid[:, 1] == sess)
                if not m.any():
                    continue
                idx = m.nonzero().flatten()[:MAX_WIN]
                fips = []
                for i in range(0, len(idx), 32):
                    xb = X[idx[i:i + 32]]
                    reg = torch.stack([xb[:, g].mean(1) for g in E.REGION_GROUPS], dim=1)
                    pw = []
                    for (lo, hi) in E.BANDS.values():
                        bb = E.bandpass_fft(reg, lo, hi)
                        pw.append((bb ** 2).mean(-1))
                    fips.append(torch.log1p(torch.stack(pw, 1)).reshape(xb.size(0), -1))
                regpow_fp[(sid, sess)] = torch.cat(fips)

    for pool in [1, 8, 32]:
        w, b, sp = sep_stats(regpow_fp, pool)
        print(f"[probe3b] pool={pool:2d}: same-brain={w:.4f} cross-brain={b:.4f} "
              f"separation={b - w:+.4f} between-cell spread={sp:.4f}", flush=True)

    # ---- v2 路由器全流程离线模拟: 原型记忆库(KMeans) + z标定 ω ----
    print("[probe4] router-v2 simulation (prototype memory + z-calibrated omega)...",
          flush=True)

    def kmeans(Xn, k, iters=25, seed=0):
        rng = np.random.default_rng(seed)
        Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-8)
        idx = [rng.integers(len(Xn))]
        for _ in range(k - 1):
            sim = Xn @ Xn[idx].T
            d2 = np.clip(1.0 - sim.max(1), 0, None)     # 余弦→距离 (夹非负)
            p = d2 / (d2.sum() + 1e-12)
            idx.append(rng.choice(len(Xn), p=p))
        C = Xn[idx].copy()
        for _ in range(iters):
            sim = Xn @ C.T
            assign = sim.argmax(1)
            for j in range(k):
                m = assign == j
                if m.any():
                    C[j] = Xn[m].mean(0)
                    C[j] /= np.linalg.norm(C[j]) + 1e-8
        return C

    def omega_of(fp_dict, train_cells, test_cells, k, a, z_norm=True, pool=1):
        def poolit(v):
            if pool > 1:
                n = v.shape[0] // pool * pool
                v = v[:n].reshape(-1, pool, v.shape[1]).mean(1)
            return v
        fp = {c: poolit(v) for c, v in fp_dict.items()}
        Xtr = np.concatenate([fp[c].numpy() for c in train_cells])
        C = kmeans(Xtr, k)
        def dist(Xt):
            Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)
            return np.clip(1.0 - (Xt @ C.T).max(1), 0, None)
        d_tr = dist(Xtr)
        mu, sd = d_tr.mean(), d_tr.std() + 1e-8
        d0z = np.percentile((d_tr - mu) / sd, 95)
        wms = []
        for tc in test_cells:
            d = dist(fp_dict[tc].numpy())
            dz = (d - mu) / sd if z_norm else d
            d0 = d0z if z_norm else np.percentile(d_tr, 95)
            wms.append(float(np.mean(1.0 / (1.0 + np.exp(-a * (d0 - dz))))))
        return np.mean(wms)

    for a in [3, 5, 10]:
        loso_w, xs_w = [], []
        for h in range(N_SUBJ):
            train_c = [(s, sess) for s in range(N_SUBJ) if s != h
                       for sess in [1, 2, 3] if (s, sess) in regpow_fp]
            test_c = [(h, sess) for sess in [1, 2, 3] if (h, sess) in regpow_fp]
            loso_w.append(omega_of(regpow_fp, train_c, test_c, k=len(train_c) // 3, a=a))
            train_c2 = [(s, sess) for s in range(N_SUBJ) for sess in [1, 2]
                        if (s, sess) in regpow_fp]
            test_c2 = [(s, 3) for s in range(N_SUBJ) if (s, 3) in regpow_fp]
            xs_w.append(omega_of(regpow_fp, train_c2, test_c2, k=len(train_c2) // 2, a=a))
        print(f"[probe4] a={a:2d}: LOSO omega={np.mean(loso_w):.3f} "
              f"xsess omega={np.mean(xs_w):.3f} separation={np.mean(xs_w) - np.mean(loso_w):+.3f}",
              flush=True)

    # ---- probe5: margin 判读 (最近原型 vs 次近原型的差) + 时间池化 ----
    print("[probe5] margin metric (d_2nd − d_1st) + pooled fingerprints...", flush=True)

    def pool_cells(fp_dict, cells, pool):
        out = {}
        for c in cells:
            v = fp_dict[c].numpy()
            n = v.shape[0] // pool * pool
            out[c] = torch.from_numpy(v[:n].reshape(-1, pool, v.shape[1]).mean(1))
        return out

    def margin_omega(fp_dict, train_cells, test_cells, k, a, pool):
        fp = pool_cells(fp_dict, train_cells + test_cells, pool)
        Xtr = np.concatenate([fp[c].numpy() for c in train_cells])
        C = kmeans(Xtr, k)
        def margins(Xt):
            Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)
            d = 1.0 - (Xt @ C.T)                      # (M, K)
            s = np.sort(d, axis=1)
            return s[:, 1] - s[:, 0]                  # margin = d_2nd − d_1st
        m_tr = margins(Xtr)
        m0 = np.percentile(m_tr, 5)                   # 训练 margin 低分位
        wms = []
        for tc in test_cells:
            m = margins(fp[tc].numpy())
            wms.append(float(np.mean(1.0 / (1.0 + np.exp(-a * (m - m0))))))
        return np.mean(wms)

    for a in [3, 5, 10]:
        loso_w, xs_w = [], []
        for h in range(N_SUBJ):
            train_c = [(s, sess) for s in range(N_SUBJ) if s != h
                       for sess in [1, 2, 3] if (s, sess) in regpow_fp]
            test_c = [(h, sess) for sess in [1, 2, 3] if (h, sess) in regpow_fp]
            loso_w.append(margin_omega(regpow_fp, train_c, test_c,
                                       k=len(train_c) // 3, a=a, pool=16))
            train_c2 = [(s, sess) for s in range(N_SUBJ) for sess in [1, 2]
                        if (s, sess) in regpow_fp]
            test_c2 = [(s, 3) for s in range(N_SUBJ) if (s, 3) in regpow_fp]
            xs_w.append(margin_omega(regpow_fp, train_c2, test_c2,
                                     k=len(train_c2) // 2, a=a, pool=16))
        print(f"[probe5] a={a:2d} (pool16): LOSO omega={np.mean(loso_w):.3f} "
              f"xsess omega={np.mean(xs_w):.3f} separation={np.mean(xs_w) - np.mean(loso_w):+.3f}",
              flush=True)

    # ---- probe6: 还活着的两条路 ----
    # A. 长流池化 + 最近原型 + 原始d标定 (不z化, z化被probe4证反转)
    print("[probe6A] nearest-prototype, LONG pooling, raw-d calibration...", flush=True)
    for pool in [32, 64, 128]:
        loso_w, xs_w = [], []
        for h in range(N_SUBJ):
            train_c = [(s, sess) for s in range(N_SUBJ) if s != h
                       for sess in [1, 2, 3] if (s, sess) in regpow_fp]
            test_c = [(h, sess) for sess in [1, 2, 3] if (h, sess) in regpow_fp]
            loso_w.append(omega_of(regpow_fp, train_c, test_c, k=len(train_c) // 3,
                                   a=10, z_norm=False, pool=pool))
            train_c2 = [(s, sess) for s in range(N_SUBJ) for sess in [1, 2]
                        if (s, sess) in regpow_fp]
            test_c2 = [(s, 3) for s in range(N_SUBJ) if (s, 3) in regpow_fp]
            xs_w.append(omega_of(regpow_fp, train_c2, test_c2, k=len(train_c2) // 2,
                                 a=10, z_norm=False, pool=pool))
        print(f"[probe6A] pool={pool:3d}: LOSO omega={np.mean(loso_w):.3f} "
              f"xsess omega={np.mean(xs_w):.3f} separation={np.mean(xs_w) - np.mean(loso_w):+.3f}",
              flush=True)

    # B. 被试原型判读: KNN投票置信度 (训练集被试身份是白给的, 无协议标签)
    print("[probe6B] subject-KNN novelty (votes for plurality subject / k)...", flush=True)

    def knn_conf(fp_dict, train_cells, test_cells, k=5):
        Xtr, ytr = [], []
        for (s, sess) in train_cells:
            v = fp_dict[(s, sess)].numpy()
            Xtr.append(v); ytr.append(np.full(len(v), s))
        Xtr = np.concatenate(Xtr).astype(np.float32)
        ytr = np.concatenate(ytr).astype(np.int64)
        Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)

        def conf(Xt, drop_first):
            Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)
            sim = Xt @ Xtr.T
            nbr = np.argpartition(-sim, k + 1, axis=1)[:, drop_first:k + 1]
            votes = ytr[nbr]                                  # (M, k)
            best = np.array([np.bincount(votes[i], minlength=N_SUBJ).max()
                             for i in range(len(votes))]) / k
            return best

        c_tr = conf(Xtr, drop_first=1)                        # 训练窗: 去掉自匹配
        c0 = np.percentile(c_tr, 10)
        wms = []
        for tc in test_cells:
            c = conf(fp_dict[tc].numpy(), drop_first=0)
            wms.append(float(np.mean(1.0 / (1.0 + np.exp(-5 * (c - c0))))))
        return np.mean(wms)

    loso_conf, xs_conf = [], []
    for h in range(N_SUBJ):
        train_c = [(s, sess) for s in range(N_SUBJ) if s != h
                   for sess in [1, 2, 3] if (s, sess) in regpow_fp]
        test_c = [(h, sess) for sess in [1, 2, 3] if (h, sess) in regpow_fp]
        loso_conf.append(knn_conf(regpow_fp, train_c, test_c))
        train_c2 = [(s, sess) for s in range(N_SUBJ) for sess in [1, 2]
                    if (s, sess) in regpow_fp]
        test_c2 = [(s, 3) for s in range(N_SUBJ) if (s, 3) in regpow_fp]
        xs_conf.append(knn_conf(regpow_fp, train_c2, test_c2))
    print(f"[probe6B] k=5: LOSO omega={np.mean(loso_conf):.3f} "
          f"xsess omega={np.mean(xs_conf):.3f} separation={np.mean(xs_conf) - np.mean(loso_conf):+.3f}",
          flush=True)


if __name__ == "__main__":
    main()
