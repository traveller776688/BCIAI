#!/usr/bin/env python3
"""
EEG V6-P — 性格社会预研 (2026-08-17, 新论文《策略与性格合成》的第一次实验)
================================================================================
只换一个模块: MutualSocietyV3 (脑区锚定社会) → PersonalitySocietyV1 (异构性格社会).
其余一切 (前端/水循环/稳定性头/融合v6/损失/训练协议) 与 DAME-C5 逐字相同;
训练循环直接复用 eeg_v5_coupling_experiment.loso_v5 / session_run_v5 —
协议同一性是本实验的设计前提: Δ = 社会的异构化效应, 不含其他混淆。

异构化三件套 (全部在专家层):
  1. 视图异构 — 12 专家分 3 家族 × 4:
     边族   (f0): 注视本区 11 条入射边 × 5 频段的静态配置   (原设计, 内部对照)
     功率族 (f1): 注视全脑投影功率 (H_pow 池化, 64维) — 能量状态
     慢变族 (f2): 注视入射边模式的窗口内变化量 (plv末步−首步) — 趋势
  2. 动力学异构 — 记忆更新门 η 乘家族尺度: f0×1.0, f1×1.3 (快记), f2×0.7 (慢记)
  3. 门温/投票异构 — 每家族独立门温度曲线 (f0: 0.8→3.5 原样; f1: 1.2→4.0 更锐;
     f2: 0.6→2.5 更钝) + 每家族独立输出投影 (三路投票通道求和 → 仲裁 O)

预注册判读 (负结果照实成文, 与 DAME 论文同款纪律):
  LOSO: |Δ| < 1.0 → 同档 (预期: 异构不跨噪声底, 附录E 900×噪声比);
        Δ ≥ +0.5 → 惊喜 (异构跨过域差, 性格锚可迁移的正面证据);
        Δ ≤ −0.5 → 异构有害 (功率视图泄漏个体身份, probe3 前车之鉴 — 同样重要)
  CS:   Δ ≥ 0   → 异构 ≥ 脑区锚;  Δ ≥ +0.5 → 明显赢 (预期方向: 多样策略吃内容多变)

v1 保留不变: 互助结构 (W_mutual 邻接初始化/社区掩码/社区重分配) — 隔离异构效应.
v2 候选 (本文不启用): 互助倾向异构 (合群/孤僻型), 门控锐度学习化.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import eeg_v5_coupling_experiment as E

DEVICE = E.DEVICE


# =========================================================================
# 1. PersonalitySocietyV1 — 异构性格社会 (唯一的新模块)
# =========================================================================
class PersonalitySocietyV1(nn.Module):
    """12 专家 × 3 家族 × 4: 视图/动力学/门温/投票 四维异构.
    与 MutualSocietyV3 共享: 专家锚(expertise)/三流GRU记忆/W_mutual邻接初始化/
    社区机制/各正则损失 — 便于把 Δ 干净地归因于异构化."""

    FAMILY_VIEWS = {0: "edge", 1: "power", 2: "slow"}

    def __init__(self, N=E.N_REGIONS, D=E.D_MODEL, d_mem=E.D_MEM,
                 n_communities=4, plv_adj=None):
        super().__init__()
        self.N = N
        self.d_mem = d_mem
        self.n_communities = n_communities
        r = max(d_mem // 4, 4)

        # 家族划分: 4 边族 + 4 功率族 + 4 慢变族
        fam = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        self.register_buffer('family_ids', fam)
        self.register_buffer('eta_scale',
                             torch.tensor([1.0, 1.3, 0.7])[fam])   # 快记/慢记
        self.register_buffer('temp_lo', torch.tensor([0.8, 1.2, 0.6])[fam])
        self.register_buffer('temp_hi', torch.tensor([3.5, 4.0, 2.5])[fam])
        self.temp = E.TEMP_INIT

        if n_communities > 0:
            per_comm = N // n_communities
            ids = torch.arange(N) // per_comm
            ids = ids.clamp(max=n_communities - 1)
            self.register_buffer('community_ids', ids)
            cmask = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
            self.register_buffer('community_mask', cmask)
        else:
            self.register_buffer('community_ids', torch.zeros(N, dtype=torch.long))
            self.register_buffer('community_mask', torch.ones(N, N))
        self.register_buffer('_gate_sum', torch.zeros(N, N))
        self.register_buffer('_gate_count', torch.zeros(1))

        e_init = torch.randn(N, d_mem) * 0.1
        try:
            Q, _ = torch.linalg.qr(e_init.T)
            self.expertise = nn.Parameter(Q.T[:N] * 0.1)
        except RuntimeError:
            self.expertise = nn.Parameter(F.normalize(e_init, dim=-1) * 0.1)
        self.gate_bias = nn.Parameter(torch.full((N,), -0.2))

        # 视图投影: 边族(静态配置) / 功率族(能量状态) / 慢变族(趋势)
        self.edge_proj = nn.Linear((N - 1) * E.N_BANDS, d_mem)
        self.pow_proj = nn.Linear(E.D_POW, d_mem)
        self.slow_proj = nn.Linear((N - 1) * E.N_BANDS, d_mem)
        # 耦合状态调制 (环境读数, 各族共享): (B, F*R) → 招募偏置 (B, N)
        self.coup_gate_proj = nn.Linear(E.N_BANDS * N, N)

        scale = 0.005 / math.sqrt(N * r)
        if plv_adj is not None:
            w = torch.zeros(N, N, r, r)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        w[i, j] = (0.01 / math.sqrt(N * r)) * (0.5 + plv_adj[i, j])
            self.W_mutual = nn.Parameter(w + torch.randn(N, N, r, r) * scale * 0.1)
        else:
            self.W_mutual = nn.Parameter(torch.randn(N, N, r, r) * scale)
        self.proj_m_in = nn.Linear(d_mem, r, bias=False)
        self.proj_m_out = nn.Linear(r, d_mem, bias=False)

        self.U = nn.Linear(d_mem, d_mem, bias=False)
        self.V = nn.Linear(d_mem, d_mem, bias=False)
        self.ln_mem = nn.LayerNorm(d_mem)
        self.eta_net = nn.Sequential(nn.Linear(d_mem * 2, d_mem), nn.Sigmoid())

        self.kl_mod_net = nn.Sequential(
            nn.Linear(N, max(N, D // 4)), nn.GELU(),
            nn.Linear(max(N, D // 4), N),
        )

        # 投票异构: 每家族独立输出投影 (三路投票通道)
        self.proj_out = nn.ModuleList([nn.Linear(d_mem, D) for _ in range(3)])
        self.register_buffer('mem', torch.zeros(N, d_mem))
        self.share_ratio = 0.6

    # ---- 结构绑定 (与 V3 逐字一致) ----
    def _bind_pair_indices(self, pair_i, pair_j):
        self.register_buffer('_pair_i', torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer('_pair_j', torch.tensor(pair_j, dtype=torch.long))
        pi = torch.tensor(pair_i, dtype=torch.long)
        pj = torch.tensor(pair_j, dtype=torch.long)
        inc = []
        for r in range(self.N):
            m = (pi == r) | (pj == r)
            nbr = torch.where(pi == r, pj, pi)[m]
            order = torch.argsort(nbr)
            inc.append(torch.nonzero(m, as_tuple=False).flatten()[order])
        self.register_buffer('_inc_idx', torch.stack(inc))          # (N, N-1)

    def _coup_strength(self, plv):
        B = plv.size(0)
        plv_mean = plv.mean(-1)
        out = plv_mean.new_zeros(B, plv_mean.size(1), self.N)
        out = out.index_add_(-1, self._pair_i, plv_mean)
        out = out.index_add_(-1, self._pair_j, plv_mean)
        return out.reshape(B, -1)

    def set_temp(self, progress):
        """每家族独立温度曲线 (progress ∈ [0,1])."""
        self.temp = self.temp_lo + (self.temp_hi - self.temp_lo) * progress

    def calibrate_omega(self, quantile=0.95):
        pass    # 场路由已证伪关闭 (ω≡1), 接口对齐用

    def forward(self, plv, H_pow_pooled):
        """plv: (B,F,P,Tc); H_pow_pooled: (B,64) → O (B,D), gates (B,N)."""
        B = plv.size(0)
        plv_mean = plv.mean(-1)                                   # (B,F,P)
        plv_delta = plv[:, :, :, -1] - plv[:, :, :, 0]            # 窗口内变化
        edges_b = plv_mean.transpose(1, 2)                        # (B,P,F)
        delta_b = plv_delta.transpose(1, 2)

        e_view = edges_b[:, self._inc_idx, :].reshape(B, self.N, -1)
        s_view = delta_b[:, self._inc_idx, :].reshape(B, self.N, -1)
        e_proj = self.edge_proj(e_view)                           # (B,N,32)
        s_proj = self.slow_proj(s_view)
        p_proj = self.pow_proj(H_pow_pooled).unsqueeze(1).expand(B, self.N, -1)

        # 家族掩码选择视图 (专家只看自己家族的那一路)
        fam_mask = F.one_hot(self.family_ids, 3).float().view(1, self.N, 3)
        view = (e_proj * fam_mask[..., 0:1] + p_proj * fam_mask[..., 1:2]
                + s_proj * fam_mask[..., 2:3])

        e_n = F.normalize(self.expertise, dim=-1)
        e_v = F.normalize(view, dim=-1)
        cos_sim = torch.einsum('bnd,nd->bn', e_v, e_n)
        coup = self._coup_strength(plv)
        coup_bias = self.coup_gate_proj(coup)
        omega = coup.new_full((coup.size(0),), 1.0)               # ω≡1
        gates = torch.sigmoid(self.temp * cos_sim + self.gate_bias + coup_bias)

        if self.training:
            with torch.no_grad():
                g = gates.detach()
                self._gate_sum += (g.T @ g) / max(B, 1)
                self._gate_count += 1

        if self.training:
            share_mask = torch.rand(self.N, device=plv.device) < self.share_ratio
            if share_mask.sum() < 2:
                idx = torch.randperm(self.N, device=plv.device)[:2]
                share_mask[idx] = True
        else:
            share_mask = torch.ones(self.N, device=plv.device).bool()

        mem_expanded = self.mem.unsqueeze(0).expand(B, -1, -1)
        mem_r = self.proj_m_in(mem_expanded)
        W_eff = self.W_mutual * self.community_mask.view(self.N, self.N, 1, 1)
        mutual_r = torch.einsum('ijdk,bjk->bid', W_eff, mem_r)
        mutual_r = mutual_r * share_mask.float().view(1, self.N, 1)
        mutual = self.proj_m_out(mutual_r)

        ext = self.U(view)
        slf = self.V(mem_expanded)
        m_tilde = torch.tanh(self.ln_mem(ext + mutual + slf))
        eta_in = torch.cat([view, mem_expanded], dim=-1)
        eta = (self.eta_net(eta_in)
               * self.eta_scale.view(1, self.N, 1)).clamp(0.0, 1.0)   # 家族动力学
        mem_new = (1 - eta) * mem_expanded + eta * m_tilde

        if self.training:
            with torch.no_grad():
                self.mem.data = 0.9 * self.mem + 0.1 * mem_new.mean(0)

        # 投票异构: 家族内门控求和 → 家族独立投影 → 求和仲裁
        O_fam = []
        for f in range(3):
            mf = (self.family_ids == f)
            O_f = (gates[:, mf].unsqueeze(-1) * mem_new[:, mf]).sum(dim=1)
            O_fam.append(self.proj_out[f](O_f))
        O = sum(O_fam)                                            # (B,D)

        kl_mod_per_neuron = F.softplus(self.kl_mod_net(gates.detach()))
        return O, gates, share_mask, kl_mod_per_neuron, omega

    # ---- 正则/社区/指纹 (与 V3 逐字一致) ----
    def gate_entropy_loss(self, gates):
        p = gates.mean(dim=0).clamp(min=1e-6, max=1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        return -entropy

    @torch.no_grad()
    def reassign_communities(self):
        if self.n_communities == 0 or self._gate_count.item() < 20:
            return
        C = self._gate_sum / self._gate_count
        diag = C.diag().clamp(min=1e-6)
        C_norm = C / (diag.unsqueeze(0) * diag.unsqueeze(1) + 1e-8).sqrt()
        n_clusters = self.n_communities
        best_labels = self.community_ids.clone()
        best_score = -float('inf')
        for _ in range(15):
            centroids = C_norm[torch.randperm(self.N, device=C.device)[:n_clusters]]
            for __ in range(50):
                sim = C_norm @ centroids.T
                labels = sim.argmax(dim=1)
                if labels.unique().numel() < n_clusters:
                    break
                for k in range(n_clusters):
                    mask_k = labels == k
                    if mask_k.sum() > 0:
                        centroids[k] = C_norm[mask_k].mean(dim=0)
            score = 0.0
            for k in range(n_clusters):
                mk = labels == k
                if mk.sum() > 1:
                    score += C_norm[mk][:, mk].mean() - C_norm[mk][:, ~mk].mean()
            if score > best_score:
                best_score = score
                best_labels = labels.clone()
        self.community_ids.copy_(best_labels)
        self.community_mask.copy_(
            (best_labels.unsqueeze(0) == best_labels.unsqueeze(1)).float())
        self._gate_sum.zero_()
        self._gate_count.zero_()

    def ortho_loss(self):
        e = F.normalize(self.expertise, dim=-1)
        sim = e @ e.T
        off_mask = ~torch.eye(self.N, dtype=torch.bool, device=e.device)
        return F.relu(sim[off_mask] - 0.05).pow(2).mean()

    def specialization_loss(self, gates):
        B, N = gates.shape
        if B < 4 or N < 2:
            return torch.tensor(0.0, device=gates.device)
        g_centered = gates - gates.mean(dim=0, keepdim=True)
        g_std = g_centered.std(dim=0, keepdim=True).clamp(min=1e-6)
        g_normed = g_centered / g_std
        C = (g_normed.T @ g_normed) / (B - 1)
        off_mask = ~torch.eye(N, dtype=torch.bool, device=C.device)
        return C[off_mask].abs().mean()

    def mutual_loss(self):
        W = self.W_mutual
        Wt = W.transpose(0, 1).transpose(2, 3)
        sym_loss = (W - Wt).pow(2).sum() / max(self.N * (self.N - 1), 1)
        sparse_loss = 0.001 * W.abs().mean()
        return sym_loss + sparse_loss

    @torch.no_grad()
    def strategy_fingerprint(self):
        e = F.normalize(self.expertise, dim=-1)
        sim = e @ e.T
        W_norms = self.W_mutual.view(self.N, self.N, -1).norm(dim=-1).sum(dim=1)
        return {
            "expertise": self.expertise.detach().cpu(),
            "family_ids": self.family_ids.detach().cpu(),
            "community_ids": self.community_ids.detach().cpu(),
            "community_mask": self.community_mask.detach().cpu(),
            "W_outgoing": W_norms.detach().cpu(),
            "expertise_sim": sim.detach().cpu(),
            "gate_bias": self.gate_bias.detach().cpu(),
        }


# =========================================================================
# 2. DAME_Personality — 顶模型 (融合/损失与 E.DAME_Coupling 逐字一致, 仅社会换装)
# =========================================================================
class DAME_Personality(nn.Module):
    def __init__(self, plv_adj=None, groups=E.REGION_GROUPS, D=E.D_MODEL):
        super().__init__()
        self.groups = groups
        self.R = len(groups)
        self.use_coupling = True
        self.use_water = True
        self.use_reflux = True
        self.use_mutual = True
        self.use_pred = True
        self.aux_losses = True

        self.coupling = E.RegionCouplingV2(groups, D, use_coupling=True)
        self.water = E.WaterCycleV2(D, E.K_LATENT, use_reflux=True)
        self.mutual = PersonalitySocietyV1(N=self.R, D=D, d_mem=E.D_MEM,
                                           n_communities=4, plv_adj=plv_adj)
        self.mutual._bind_pair_indices(self.coupling.pair_i, self.coupling.pair_j)
        P = len(self.coupling.pair_i)
        self.pred_head = E.StabilityPredHead(E.K_LATENT, D, d_plv=E.N_BANDS * P)

        self.n_terms = 3
        self.proj_pool = nn.Linear(D, D)
        self.proj_anchor = nn.Linear(D, D)
        self.proj_z = nn.Linear(E.K_LATENT, D)
        self.proj_pow = nn.Linear(E.D_POW, D)
        self.proj_w = nn.Linear(D, self.n_terms)   # 仲裁: 社会输出 → 三路权重
        self.clf = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, E.N_CLASSES))

        self._current_epoch = 0

    def set_epoch(self, epoch, total_epochs=None):
        self._current_epoch = epoch
        progress = min(1.0, epoch / max(E.TEMP_ANNEAL_EPOCHS, 1))
        self.mutual.set_temp(progress)
        if epoch >= 3:
            self.mutual.reassign_communities()

    def forward(self, X):
        """X: (B, 62, 800) raw window. 融合段与 E.DAME_Coupling 逐字一致."""
        H_coup, H_pow, plv = self.coupling(X)
        H_pow_pooled = H_pow.mean(dim=1)                     # (B, 64)
        H_pool = H_coup.mean(dim=1)                          # (B, D)

        Z, A, kl, kl_w, convergence, Z_init, reflux_mag = self.water(H_coup)

        O, gates, share_mask, kl_mod, omega = self.mutual(plv, H_pow_pooled)
        w_soc = F.softmax(self.proj_w(O), dim=-1)            # (B, 3)

        terms = [self.proj_pool(H_pool), self.proj_pow(H_pow_pooled)]
        terms.append(self.proj_anchor(A) + self.proj_z(Z))
        T = torch.stack(terms, dim=1)                        # (B, 3, D)
        w = T.new_full((T.size(0), T.size(1)), 1.0 / T.size(1))
        w = omega.unsqueeze(-1) * w_soc + (1 - omega.unsqueeze(-1)) * w
        f = (w.unsqueeze(-1) * T).sum(dim=1)

        logits = self.clf(f)
        return {
            "logits": logits, "kl_loss": kl, "kl_w": kl_w, "kl_mod": kl_mod,
            "Z_star": Z, "Z_init": Z_init, "O": O, "A": A, "gates": gates,
            "convergence": convergence, "reflux_mag": reflux_mag,
            "plv": plv, "H_pow": H_pow_pooled, "omega": omega,
        }

    def compute_loss(self, out, labels, y_next=None, plv_next=None):
        loss = F.cross_entropy(out["logits"], labels, label_smoothing=0.1)

        warmup = min(1.0, (self._current_epoch + 1) / max(E.KL_WARMUP_EPOCHS, 1))
        loss = loss + E.KL_W * warmup * out["kl_loss"]

        gates = out["gates"]
        loss = loss + E.ORTHO_W * self.mutual.ortho_loss()
        loss = loss + E.MUTUAL_W * self.mutual.mutual_loss()
        loss = loss + E.SPEC_W * self.mutual.specialization_loss(gates)
        loss = loss + E.GATE_ENTROPY_W * self.mutual.gate_entropy_loss(gates)

        pred = self.pred_head(out["Z_init"], out["A"])
        loss = loss + E.PRED_W * F.mse_loss(
            pred["c_self"], self.pred_head.embed(out["plv"]))
        if y_next is not None:
            stab_warm = min(1.0, max(0.0, (self._current_epoch - 3)) / 5.0)
            stab_target = (labels == y_next).long()
            loss = loss + E.STAB_W * stab_warm * F.cross_entropy(
                pred["stab_logits"], stab_target)

        loss = loss + E.REFLUX_W * F.relu(0.01 - out["reflux_mag"])
        return loss


# =========================================================================
# 3. Main — 双臂对照 (P6-Soc vs DAME-C5), 协议逐字复用
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=int, default=8)
    ap.add_argument("--subjects", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=E.EPOCHS)
    ap.add_argument("--session-split", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 2被试 2epoch LOSO")
    args = ap.parse_args()
    if args.smoke:
        args.fast = 2
        args.subjects = 2
        args.seeds = 1
        args.epochs = 2
        print("[SMOKE] 2 subjects, 2 epochs", flush=True)
    if args.fast:
        args.subjects = args.fast
        print(f"[FAST] {args.fast} subjects", flush=True)

    seeds = E.ALL_SEEDS[args.seed_start: args.seed_start + args.seeds]
    out_path = os.path.join(E.RESULTS_DIR,
                            "eeg_v6p_results_smoke.json" if args.smoke
                            else "eeg_v6p_results.json")
    done = {}
    if os.path.exists(out_path):
        done = json.load(open(out_path))

    st = se = None
    if args.session_split:
        st, se = [1, 2], [3]
        print("[PROTOCOL] cross-session (train s1+s2 → test s3)", flush=True)
    else:
        print("[PROTOCOL] LOSO", flush=True)

    X, y, subj, pair_idx, wid = E.load_raw_with_pairs(
        n_subjects=args.subjects, norm_sessions=tuple(st) if st else None)

    def save_fn(key, acc):
        done[key] = acc
        json.dump(done, open(out_path, "w"), indent=2)

    arms = {
        "P6-Soc": dict(fn=lambda plv_adj=None: DAME_Personality(plv_adj=plv_adj)),
        "DAME-C5": dict(fn=E.MODEL_SPECS["DAME-C5"]["fn"]),   # 基线: 原社会
    }
    for mname, spec in arms.items():
        print(f"\n{'=' * 60}\nMODEL: {mname}\n{'=' * 60}", flush=True)
        if args.session_split:
            E.session_run_v5(spec["fn"], X, y, subj, wid, pair_idx, seeds,
                             epochs=args.epochs, kind="dame",
                             done_folds=done, save_fn=save_fn, tag=mname,
                             sessions_train=st, sessions_test=se)
        else:
            E.loso_v5(spec["fn"], X, y, subj, pair_idx, seeds,
                      epochs=args.epochs, kind="dame",
                      done_folds=done, save_fn=save_fn, tag=mname)

    # ---- 汇总 + 预注册判读 ----
    # 注: 同文件缓存含 LOSO (_dame_sN_) 与 CS (_dame_sessN_) 两种 key,
    #     必须按协议过滤, 否则会混出假均值。
    stats = {}
    for mname in arms:
        fold_accs = [v for k, v in done.items()
                     if k.startswith(f"{mname}_") and isinstance(v, float)
                     and (("_sess" in k) if args.session_split
                          else ("_sess" not in k))]
        if not fold_accs:
            continue
        stats[mname] = {
            "mean": float(np.mean(fold_accs)),
            "std": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
            "n_folds": len(fold_accs),
        }
    print(f"\n{'=' * 70}\nV6-P SUMMARY ({args.subjects}subj x {args.seeds}seed "
          f"x {args.epochs}ep, {'CS' if args.session_split else 'LOSO'})\n{'=' * 70}")
    for mname in arms:
        if mname in stats:
            s = stats[mname]
            print(f"{mname:<10} {s['mean'] * 100:>6.2f}±{s['std'] * 100:.2f} "
                  f"(n={s['n_folds']})")
    if "P6-Soc" in stats and "DAME-C5" in stats:
        d = stats["P6-Soc"]["mean"] - stats["DAME-C5"]["mean"]
        d_pt = d * 100   # 判读单位=百分点 (文件头预注册门槛均为"点")
        print(f"\nΔ = P6-Soc − DAME-C5 = {d:+.4f} ({d_pt:+.2f} 点)")
        if args.session_split:
            verdict = ("明显赢 (异构≥脑区锚确认)" if d_pt >= 0.5 else
                       ("异构≥脑区锚 (名义)" if d_pt >= 0
                        else "异构差于脑区锚 — 如实成文"))
        else:
            # 头注 |Δ|<1.0 与 Δ≥+0.5 区间重叠: 伞条款优先 (1.0 内=同档),
            # ±0.5 作为名义方向标注; 跨 1.0 才判惊喜/有害。
            if d_pt >= 1.0:
                verdict = "惊喜: 异构跨过域差"
            elif d_pt <= -1.0:
                verdict = "异构有害: 功率视图疑似泄漏身份 — 如实成文"
            elif d_pt >= 0.5:
                verdict = "同档 (名义正面倾向, 未跨 1.0)"
            elif d_pt <= -0.5:
                verdict = "同档 (名义负面倾向, 未跨 1.0)"
            else:
                verdict = "同档 (预期: 异构不跨噪声底)"
        print(f"[verdict] {verdict}")
    print(f"[results] {out_path}")


if __name__ == "__main__":
    main()
