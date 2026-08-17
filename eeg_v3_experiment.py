#!/usr/bin/env python3
"""
EEG V3 — DAME Architecture for SEED-IV Emotion Recognition
==========================================================
Ports NLP's proven DAME-Lite (WaterCycleV2 + MutualSocietyV2) to EEG.
Adds Fast/Slow pathways with ablation study.
8-model comparison matching NLP experiment pattern.

Dataset: SEED-IV — 15 subjects × 3 sessions × 24 trials, 4 emotions
Features: DE (Differential Entropy) with LDS smoothing, 62ch × 5 bands

Architecture:
  DE Features (B,C=62,T,F=5)
    → InputProj: Conv1d(C*F→D=256)
    → [FastPathway ON/OFF]: 2D Conv spatial-spectral gate + adversarial
    → SlowPathway: Bottleneck TCN (k=7, dil=1,2,4)
    → WaterCycleV2: VIB(K=32)→CrossAttn→Banach reflux
    → MutualSocietyV2: N=32, C=4 communities, cosine gate + 3-stream GRU
    → Classifier: 4-class

Evaluation: LOSO (15-fold), 3 seeds, 8 models
"""
import math, json, random, time, os, sys, warnings
from collections import Counter

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEED_DIR = os.path.join(BASE_DIR, "SEED_IV", "SEED_IV")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"SEED dir: {SEED_DIR}")

# =========================================================================
# CONFIG — tuned for SEED-IV 4-class emotion recognition
# =========================================================================
D_MODEL = 256        # Unified hidden dimension
K_LATENT = 32        # VIB compression: D→K (NLP-proven: stronger bottleneck → better distillation)
N_NEURONS = 32       # Mutual society size
N_COMMUNITIES = 4    # Regularized communities
D_MEM = 32           # GRU memory dimension
WINDOW_SIZE = 16     # Time steps per segment (~4s @ 4Hz after LDS smoothing)
WINDOW_STRIDE = 8    # 50% overlap

# Loss weights (rebalanced: weaker regularization, stronger classification)
KL_W = 0.008         # NLP-proven: stronger KL forces discriminative bottleneck (K=32 → 8:1)
ORTHO_W = 0.0005     # 10× reduction (was 0.005): prevent loss competition
MUTUAL_W = 0.0001    # 10× reduction (was 0.001)
SPEC_W = 0.0002      # 10× reduction (was 0.002)
REFLUX_W = 0.0       # Disabled: Banach guarantee sufficient; magnitude reg adds noise
GATE_ENTROPY_W = 0.0001  # 10× reduction (was 0.001)
PRED_W = 0.0         # Disabled: within-window fallback adds noise on small EEG data

# Adversarial (on encoder output, lightweight DANN-style regularizer)

# Temperature annealing
TEMP_INIT = 0.8
TEMP_FINAL = 2.5
TEMP_ANNEAL_EPOCHS = 8

# KL warmup
KL_WARMUP_EPOCHS = 8  # Extended from 3 epochs (VIB needs time to learn good latents)

# Training
LR = 2e-4
EPOCHS = 15
BATCH_SIZE = 64
N_SEEDS = 5
ALL_SEEDS = [42, 123, 789, 456, 999]

# Quick test mode
QUICK_TEST = False
QUICK_EPOCHS = 3
QUICK_SUBJECTS = 3
LEVEL2_EPOCHS = 8
LEVEL2_SUBJECTS = 5

# SEED-IV session labels (from ReadMe)
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}
# 0=neutral, 1=sad, 2=fear, 3=happy
N_CLASSES = 4

# =========================================================================
# 1. SEED-IV Data Loader
# =========================================================================

def load_seed_iv(data_dir=SEED_DIR, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Load SEED-IV DE features, segment into windows, normalize per-subject.

    Each .mat file: de_LDS1..de_LDS24, shape (62, T_var, 5)
    Returns: X (N, 62, window_size, 5), y (N,), subj (N,), session (N,)
    """
    feature_dir = os.path.join(data_dir, "eeg_feature_smooth")
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Feature dir not found: {feature_dir}")

    all_X, all_y, all_subj, all_sess = [], [], [], []
    n_subjects = 0

    for session in [1, 2, 3]:
        sess_dir = os.path.join(feature_dir, str(session))
        if not os.path.exists(sess_dir):
            print(f"  [WARN] Session dir missing: {sess_dir}")
            continue

        labels = SESSION_LABELS[session]
        mat_files = sorted([f for f in os.listdir(sess_dir) if f.endswith('.mat')])

        # Each .mat file = one subject in this session
        # Filename: {subject_id}_{date}.mat, subject_id from 1-15
        for fname in mat_files:
            try:
                subj_id = int(fname.split('_')[0])
            except ValueError:
                continue
            n_subjects = max(n_subjects, subj_id)

            mat_path = os.path.join(sess_dir, fname)
            try:
                data = loadmat(mat_path)
            except Exception as e:
                print(f"  [WARN] Failed to load {fname}: {e}")
                continue

            # Extract 24 trials × de_LDS features
            for trial_idx in range(24):
                var_name = f"de_LDS{trial_idx + 1}"
                if var_name not in data:
                    continue

                trial_data = data[var_name]  # (62, T_var, 5)
                if trial_data.ndim != 3:
                    continue

                T_var = trial_data.shape[1]
                label = labels[trial_idx]

                # Segment into windows
                for start in range(0, T_var - window_size + 1, stride):
                    window = trial_data[:, start:start + window_size, :]  # (62, W, 5)
                    all_X.append(torch.tensor(window, dtype=torch.float32))
                    all_y.append(label)
                    all_subj.append(subj_id - 1)  # 0-indexed
                    all_sess.append(session - 1)   # 0-indexed

    X = torch.stack(all_X)  # (N, 62, W, 5)
    y = torch.tensor(all_y, dtype=torch.long)
    subj = torch.tensor(all_subj, dtype=torch.long)
    sess = torch.tensor(all_sess, dtype=torch.long)

    # Per-subject z-score normalization
    # Also assign trial_window_id for valid pre-migration pairing
    for s in range(n_subjects):
        mask = subj == s
        if mask.sum() > 0:
            mean = X[mask].mean(dim=(0, 1, 3), keepdim=True)
            std = X[mask].std(dim=(0, 1, 3), keepdim=True) + 1e-8
            X[mask] = (X[mask] - mean) / std

    print(f"[Data] SEED-IV: {X.shape[0]} windows | {n_subjects} subjects | "
          f"{X.shape[2]} time steps | {X.shape[3]} bands | "
          f"classes={dict(Counter(all_y))}")
    return X, y, subj, sess


class SEEDDataset(Dataset):
    """Simple dataset wrapper for SEED-IV windows."""
    def __init__(self, X, y, subj):
        self.X = X
        self.y = y
        self.subj = subj if subj is not None else torch.zeros(len(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subj[idx]


# =========================================================================
# 2. DAME Core Components (ported from NLP dame_full_experiment.py)
# =========================================================================

class GradReverse(torch.autograd.Function):
    """Gradient reversal layer for adversarial domain confusion"""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None


class InputProjection(nn.Module):
    """Project SEED DE features (62ch × 5bands) → unified D_MODEL

    EEG version of NLP's LightEncoder — no pretrained weights, from scratch.
    Uses depthwise-separable conv for parameter efficiency.
    """
    def __init__(self, C=62, n_bands=5, D=D_MODEL):
        super().__init__()
        in_ch = C * n_bands  # 310
        self.C = C
        self.n_bands = n_bands
        # Depthwise: each channel-band pair convolved separately
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, D, kernel_size=1)
        self.bn = nn.BatchNorm1d(D)
        self.proj_out = nn.Linear(D, D)

    def forward(self, X):
        # X: (B, C, T, F) → (B, C*F, T) → (B, D, T) → (B, T, D)
        B, C, T, n_bands = X.shape
        x = X.permute(0, 1, 3, 2).reshape(B, C * n_bands, T)  # (B, 310, T)
        x = F.gelu(self.depthwise(x))
        x = F.gelu(self.bn(self.pointwise(x)))           # (B, D, T)
        x = x.transpose(1, 2)                             # (B, T, D)
        return self.proj_out(x)                           # (B, T, D)


class FastPathwayEEG(nn.Module):
    """Fast pathway — spatial-spectral gate for subject-noise removal + adversarial

    Philosophy (from NLP FastPathwayNLP):
      Gate each (channel, time, frequency) point → mask out subject-specific noise
      Adversarial: removed portion should NOT predict subject identity

    EEG adaptation:
      NLP gates tokens → EEG gates (ch, time) points via 2D Conv
      Gate learns: which spatial-spectral patterns are emotion-relevant vs subject-idiosyncratic
    """
    def __init__(self, C=62, T=16, D=D_MODEL, n_subjects=15, use_adv=True):
        super().__init__()
        self.T_sharpen = 3.0
        self.smooth_k = 5
        self.use_adv = use_adv

        # 2D Conv gate over (C, T) — input has F=5 bands collapsed into channels
        self.gate_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16), nn.GELU(), nn.Dropout(0.15),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), nn.GELU(), nn.Dropout(0.15),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        # Adversary: tries to predict subject from REMOVED portion.
        # NLP lesson: DANN −1.3% on small domain count (3 domains). EEG has 15
        # subjects — adversarial confusion may similarly destroy within-subject
        # emotion signal. Gated behind use_adv for ablation.
        if use_adv:
            self.adversary = nn.Sequential(
                nn.Linear(D, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, n_subjects)
            )
        else:
            self.adversary = None

    def forward(self, X_raw, H_proj):
        """X_raw: (B, C, T, F) — raw DE input for spatial gate
        H_proj: (B, T, D) — projected sequence to gate
        Returns: M (B, T, 1), X_tilde (B, T, D)
        """
        B, C, T, _ = X_raw.shape

        # Spatial gate from raw DE (collapsed frequency bands)
        X_for_gate = X_raw.mean(dim=-1, keepdim=True)  # (B, C, T, 1)
        X_for_gate = X_for_gate.permute(0, 3, 1, 2)    # (B, 1, C, T)
        M_spatial = self.gate_net(X_for_gate)           # (B, 1, C, T)
        M_spatial = M_spatial.squeeze(1).mean(dim=1, keepdim=True)  # (B, 1, T)
        M_spatial = M_spatial.transpose(1, 2)           # (B, T, 1)

        # Temporal smoothing
        M = F.avg_pool1d(
            M_spatial.squeeze(-1).unsqueeze(1),
            kernel_size=self.smooth_k, stride=1,
            padding=self.smooth_k // 2
        ).squeeze(1).unsqueeze(-1)                      # (B, T, 1)
        M = torch.sigmoid(M * self.T_sharpen)

        # Gated output: keep emotion signal, suppress subject noise
        X_tilde = M * H_proj + (1 - M) * H_proj.mean(dim=1, keepdim=True)
        return M, X_tilde

    def sparsity(self, M):
        return M.mean()

    def compute_fast_loss(self, H_proj, M, subject_ids):
        """L_sparse: prevent M=1 everywhere. L_adv: removed portion → can't predict subject."""
        L_sparse = M.mean()
        if self.adversary is None:
            return L_sparse, torch.tensor(0.0, device=M.device)

        removed = (1 - M) * H_proj  # (B, T, D)
        removed_pool = removed.mean(dim=1)  # (B, D)
        rev = GradReverse.apply(removed_pool, 0.01)  # Gradient reversal (unused in v3)
        adv_logits = self.adversary(rev)

        L_adv = F.cross_entropy(adv_logits, subject_ids)
        return L_sparse, L_adv


class SlowPathwayEEG(nn.Module):
    """Slow pathway — bottleneck TCN for slow cortical rhythm extraction

    Ported from NLP SlowPathwayNLP (lines 480-516).
    EEG adaptation: NLP's "semantic rhythm over paragraphs" → EEG's "slow cortical
    oscillations over time" (α/θ band analogues captured by large-kernel TCN).

    Architecture: D→bottleneck(96)→dilated TCN(k=7,dil=1,2,4)→D, with residual
    Receptive field: 1+6+12+24=43 time steps (covers several seconds of EEG)
    """
    def __init__(self, D=D_MODEL, kernel=7, dilations=(1, 2, 4), bottleneck=96):
        super().__init__()
        self.proj_down = nn.Conv1d(D, bottleneck, 1)
        self.bn_down = nn.BatchNorm1d(bottleneck)

        layers = []
        for d in dilations:
            layers.extend([
                nn.Conv1d(bottleneck, bottleneck, kernel_size=kernel, dilation=d,
                         padding=d * (kernel - 1) // 2),
                nn.BatchNorm1d(bottleneck),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        self.tcn = nn.Sequential(*layers)

        self.proj_up = nn.Conv1d(bottleneck, D, 1)
        self.bn_up = nn.BatchNorm1d(D)
        self.proj_out = nn.Linear(D, D)
        self.receptive_field = 1 + sum((kernel - 1) * d for d in dilations)

    def forward(self, X_tilde):
        """X_tilde: (B, T, D) — gated sequence from fast pathway"""
        H = X_tilde.transpose(1, 2)                          # (B, D, T)
        H = F.gelu(self.bn_down(self.proj_down(H)))          # (B, bottleneck, T)
        H = self.tcn(H)                                       # (B, bottleneck, T)
        H = F.gelu(self.bn_up(self.proj_up(H)))               # (B, D, T)
        H = H.transpose(1, 2)                                 # (B, T, D)
        return self.proj_out(H) + X_tilde                     # Residual


class WaterCycleV2(nn.Module):
    """Water Cycle V2 — VIB → CrossAttn → Banach Fixed-Point Reflux

    Direct port from NLP dame_full_experiment.py lines 774-872.

    Philosophy:
      P2: Evaporate (VIB, D→K compression) → Precipitate (CrossAttn, Z queries time steps)
          → Reflux (self-map T: K→D→K, spectral_norm guarantees contraction)
      P3: Banach theorem → ∃!Z*, no adversarial shortcut possible

    Key changes for EEG:
      - CrossAttn over TIME steps (like NLP's CrossAttn over tokens)
      - R_pooled = mean pool over time (global EEG state)
      - Spectral norm on reflux_net for Lipschitz guarantee
    """
    def __init__(self, D=D_MODEL, K=K_LATENT, max_iter=5, converge_thresh=0.98,
                 use_reflux=True):
        super().__init__()
        self.use_reflux = use_reflux
        # VIB: D→K compression
        self.mu_proj = nn.Linear(D, K)
        self.logvar_proj = nn.Linear(D, K)

        # CrossAttn: Z(B,K) queries over time steps H(B,T,D)
        self.W_Q = nn.Linear(K, K)
        self.W_K = nn.Linear(D, K)
        self.W_V = nn.Linear(D, D)

        # Reflux: g_φ(K→D) with spectral norm guarantee
        self.reflux_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(K, D // 2)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(D // 2, D)),
        )
        self.reflux_scale = nn.Parameter(torch.tensor(0.05))  # Learnable, starts small

        # Single learnable KL weight (P1: strategy is fixed, not per-sample)
        self.log_kl_w = nn.Parameter(torch.tensor(-4.83))  # ln(0.008) — NLP-proven VIB

        self.max_iter = max_iter
        self.converge_thresh = converge_thresh

    def evaporate(self, R_global):
        """P2: VIB compression D→K — distill pure emotional intrinsic Z"""
        mu = self.mu_proj(R_global)
        logvar = torch.clamp(self.logvar_proj(R_global), -10, 10)
        std = torch.exp(0.5 * logvar)
        Z = mu + (std * torch.randn_like(std) if self.training else 0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return Z, mu, logvar, kl

    def precipitate(self, Z, H_seq):
        """P2: CrossAttn — Z queries each time step for emotion contribution

        Z: (B, K) — current intrinsic state
        H_seq: (B, T, D) — full time sequence
        Returns: A (B, D) — attention-weighted anchor
        """
        Q = self.W_Q(Z)                                     # (B, K)
        K = self.W_K(H_seq)                                 # (B, T, K)
        V = self.W_V(H_seq)                                 # (B, T, D)
        scale = math.sqrt(K.size(-1))
        attn = F.softmax(Q.unsqueeze(1) @ K.transpose(-2, -1) / scale, dim=-1)  # (B, 1, T)
        A = (attn @ V).squeeze(1)                           # (B, D)
        return A, attn

    def reflux_step(self, Z, R_global):
        """P2+P3: Single self-map step T(Z) = evaporate(R + g_φ(Z)·scale)

        g_φ: K→D with spectral norm (Lipschitz ≤ 1)
        evaporate: D→K (VIB compression)
        Composite T: K→D→K — Banach theorem → ∃!Z*
        """
        g_Z = self.reflux_net(Z) * self.reflux_scale        # (B, K)→(B, D)
        combined = R_global + g_Z                            # Reflux correction
        Z_new, _, _, _ = self.evaporate(combined)
        return Z_new

    def forward(self, H_seq):
        """Full water cycle: evaporate → precipitate → reflux iteration

        H_seq: (B, T, D) — time sequence from slow pathway
        """
        R_pooled = H_seq.mean(dim=1)                         # (B, D) — global state

        # Phase 1: Evaporate
        Z_init, _, _, kl = self.evaporate(R_pooled)

        # Phase 2: Precipitate — Z queries time steps
        A, attn = self.precipitate(Z_init, H_seq)

        # Phase 3: Reflux — Banach fixed-point iteration (optional)
        Z_current = Z_init
        convergence = []
        if self.use_reflux:
            min_iters = min(2, self.max_iter - 1)
            for t in range(self.max_iter - 1):
                Z_new = self.reflux_step(Z_current, R_pooled)
                cos_sim = F.cosine_similarity(
                    Z_current.flatten(1), Z_new.flatten(1), dim=-1).mean()
                convergence.append(cos_sim.item())
                if t >= min_iters and cos_sim > self.converge_thresh:
                    break
                Z_current = Z_new

        reflux_mag = (Z_current - Z_init).norm() / (Z_init.norm() + 1e-8)
        kl_weight = torch.exp(self.log_kl_w)

        return Z_current, A, kl, kl_weight, convergence, Z_init, reflux_mag


class MutualSocietyV2(nn.Module):
    """Mutual Neuron Society V2 — Cooperative Expert Ensemble

    Direct port from NLP dame_full_experiment.py lines 875-1138.

    P1 (Fixed Strategy): Expertise vectors e_i and mutual matrix W_{j→i} are fixed
    after training. Gating sees raw features H_raw (complete feature space).

    Core mechanisms:
      1. Cosine gating: g_i = σ(α·cos(ĥ, ê_i) + b_i) with temperature annealing
      2. 3-stream GRU memory: external(U·A) + mutual(ΣW_{j→i}·m_j) + self(V·m_i)
      3. Regularized communities: co-activation clustering → community mask on W
      4. Per-neuron KL modulation: each neuron independently assesses domain difficulty

    EEG adaptation: N=32 (vs NLP's N=24) — EEG has more inter-subject variation
    """
    def __init__(self, N=N_NEURONS, D=D_MODEL, d_mem=D_MEM,
                 share_ratio=0.6, n_communities=N_COMMUNITIES):
        super().__init__()
        self.N = N
        self.d_mem = d_mem
        self.share_ratio = share_ratio
        self.n_communities = n_communities
        r = max(d_mem // 4, 4)

        # Community mask initialization
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

        # P1: Expertise vectors — orthogonal initialization
        e_init = torch.randn(N, d_mem) * 0.1
        try:
            Q, _ = torch.linalg.qr(e_init.T)
            self.expertise = nn.Parameter(Q.T[:N] * 0.1)
        except RuntimeError:
            self.expertise = nn.Parameter(F.normalize(e_init, dim=-1) * 0.1)
        self.gate_bias = nn.Parameter(torch.zeros(N))
        self.temp = TEMP_INIT
        self.raw_proj = nn.Linear(D, d_mem)

        # Anchor projection (for memory)
        self.anchor_proj = nn.Linear(D, d_mem)

        # Mutual matrix W_{j→i} (community-masked)
        scale = 0.005 / math.sqrt(N * r)
        self.W_mutual = nn.Parameter(torch.randn(N, N, r, r) * scale)
        self.proj_m_in = nn.Linear(d_mem, r, bias=False)
        self.proj_m_out = nn.Linear(r, d_mem, bias=False)

        # 3-stream memory system
        self.U = nn.Linear(d_mem, d_mem, bias=False)  # External
        self.V = nn.Linear(d_mem, d_mem, bias=False)  # Self-feedback
        self.ln_mem = nn.LayerNorm(d_mem)
        self.eta_net = nn.Sequential(
            nn.Linear(d_mem * 2, d_mem), nn.Sigmoid()
        )

        # Per-neuron KL modulation
        self.kl_mod_net = nn.Sequential(
            nn.Linear(N, max(N, D // 4)), nn.GELU(),
            nn.Linear(max(N, D // 4), N),
        )

        self.proj_out = nn.Linear(d_mem, D)
        self.register_buffer('mem', torch.zeros(N, d_mem))

    def forward(self, A, H_raw=None):
        B = A.size(0)
        if H_raw is None:
            H_raw = A

        # 1. P1: Cosine gating on raw features (society "sees" full feature space)
        H_proj = self.raw_proj(H_raw)                    # (B, d_mem)
        e_n = F.normalize(self.expertise, dim=-1)         # (N, d_mem)
        h_n = F.normalize(H_proj, dim=-1)                 # (B, d_mem)
        cos_sim = h_n @ e_n.T                             # (B, N)
        gates = torch.sigmoid(self.temp * cos_sim + self.gate_bias)

        if self.training:
            with torch.no_grad():
                g = gates.detach()
                self._gate_sum += (g.T @ g) / max(B, 1)
                self._gate_count += 1

        # 2. Memory update from anchor A
        A_proj = self.anchor_proj(A)                      # (B, d_mem)

        # Stochastic subgroup sharing
        if self.training:
            share_mask = torch.rand(self.N, device=A.device) < self.share_ratio
            if share_mask.sum() < 2:
                idx = torch.randperm(self.N, device=A.device)[:2]
                share_mask[idx] = True
        else:
            share_mask = torch.ones(self.N, device=A.device).bool()

        # Mutual information (community-masked)
        mem_expanded = self.mem.unsqueeze(0).expand(B, -1, -1)
        mem_r = self.proj_m_in(mem_expanded)
        W_eff = self.W_mutual * self.community_mask.view(self.N, self.N, 1, 1)
        mutual_r = torch.einsum('ijdk,bjk->bid', W_eff, mem_r)
        mutual_r = mutual_r * share_mask.float().view(1, self.N, 1)
        mutual = self.proj_m_out(mutual_r)

        # 3-stream GRU update
        ext = self.U(A_proj.unsqueeze(1).expand(-1, self.N, -1))
        slf = self.V(mem_expanded)
        m_tilde = self.ln_mem(ext + mutual + slf)
        m_tilde = torch.tanh(m_tilde)
        eta_in = torch.cat([A_proj.unsqueeze(1).expand(-1, self.N, -1),
                           mem_expanded], dim=-1)
        eta = self.eta_net(eta_in)
        mem_new = (1 - eta) * mem_expanded + eta * m_tilde

        if self.training:
            with torch.no_grad():
                self.mem.data = 0.9 * self.mem + 0.1 * mem_new.mean(0)

        # 4. Weighted fusion
        O = (gates.unsqueeze(-1) * mem_new).sum(dim=1)
        O = self.proj_out(O)

        # 5. Per-neuron KL modulation (diagnostic)
        kl_mod_per_neuron = F.softplus(self.kl_mod_net(gates.detach()))

        return O, gates, share_mask, kl_mod_per_neuron

    def set_temp(self, temp):
        self.temp = temp

    def gate_entropy_loss(self, gates):
        """Reward diverse neuron activation patterns"""
        p = gates.mean(dim=0).clamp(min=1e-6, max=1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        return -entropy

    @torch.no_grad()
    def reassign_communities(self):
        """K-means on gate co-activation matrix → reassign community masks"""
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
            (best_labels.unsqueeze(0) == best_labels.unsqueeze(1)).float()
        )
        self._gate_sum.zero_()
        self._gate_count.zero_()

    def ortho_loss(self):
        """Expertise diversity — prevent neuron convergence"""
        e = F.normalize(self.expertise, dim=-1)
        sim = e @ e.T
        off_mask = ~torch.eye(self.N, dtype=torch.bool, device=e.device)
        return F.relu(sim[off_mask] - 0.05).pow(2).mean()

    def specialization_loss(self, gates):
        """Batch-level specialization: penalize correlated neuron activations"""
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
        """Mutual reciprocity: symmetry + sparsity (vectorized)

        ||W[i,j] - W[j,i]^T||² summed over all (i,j) ≈ 2× the i<j pair sum;
        divide by N(N-1) to keep the same scale as the per-pair mean.
        """
        W = self.W_mutual  # (N, N, r, r)
        Wt = W.transpose(0, 1).transpose(2, 3)  # Wt[i,j] = W[j,i].T
        sym_loss = (W - Wt).pow(2).sum() / max(self.N * (self.N - 1), 1)
        sparse_loss = 0.001 * W.abs().mean()
        return sym_loss + sparse_loss

    @torch.no_grad()
    def louvain_communities(self):
        """Community detection from co-activation"""
        if self.n_communities > 0:
            ids = self.community_ids.cpu().tolist()
            comms = {}
            for i, cid in enumerate(ids):
                comms.setdefault(cid, []).append(i)
            return comms
        # Natural emergence: connected components from W_mutual weights
        W = self.W_mutual
        edge_weight = W.view(self.N, self.N, -1).norm(dim=-1)
        threshold = edge_weight.flatten().quantile(0.75)
        adj = (edge_weight > threshold).float().cpu().numpy()
        visited = set()
        comms = {}
        cid = 0
        for i in range(self.N):
            if i not in visited:
                stack = [i]
                comm = []
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        comm.append(node)
                        for j in range(self.N):
                            if adj[node, j] > 0 and j not in visited:
                                stack.append(j)
                comms[cid] = comm
                cid += 1
        return comms

    @torch.no_grad()
    def strategy_fingerprint(self):
        """Personality/strategy profile of each neuron.

        Returns dict with:
          - expertise: (N, d_mem) — what each neuron specializes in
          - community: (N,) — which community each belongs to
          - W_norms: (N,) — how much each neuron engages in mutual aid
          - expertise_sim: (N, N) — expertise similarity matrix
        """
        e = F.normalize(self.expertise, dim=-1)
        sim = e @ e.T
        W_norms = self.W_mutual.view(self.N, self.N, -1).norm(dim=-1).sum(dim=1)
        return {
            "expertise": self.expertise.detach().cpu(),
            "community_ids": self.community_ids.detach().cpu(),
            "community_mask": self.community_mask.detach().cpu(),
            "W_outgoing": W_norms.detach().cpu(),       # Total mutual aid given
            "expertise_sim": sim.detach().cpu(),         # Who is similar to whom
            "gate_bias": self.gate_bias.detach().cpu(),  # Baseline activation tendency
            "temperature": self.temp,
        }


class PredictionHeadV2(nn.Module):
    """Pre-migration + Error-Correction Look-Ahead (P4 enhanced)

    NOT simple MSE fitting. Three-stage predictive coding:
      1. Predict one step:  Ẑ_{t+1} = f(Z_t, O_t)
      2. Compute error:      ε = Z_{t+1} - Ẑ_{t+1}
      3. Correct + predict:  Z'_t = Z_t + g(ε, Z_t)  →  Ẑ_{t+2} = f(Z'_t, O_t)

    Dual loss: MSE(Ẑ_{t+1}, Z_{t+1}) + λ·MSE(Ẑ_{t+2}, Z_{t+2})
    Two-step prevents myopic extrapolation — forces understanding of error patterns.
    Corrector g(ε, Z) learns: "given this prediction error pattern, how should I
    adjust my internal state to see further ahead?"
    """
    def __init__(self, K=K_LATENT, D=D_MODEL, hidden=128):
        super().__init__()
        # Predictor f: (Z_t, O_t) → Ẑ_next
        self.predictor = nn.Sequential(
            nn.Linear(K + D, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, K),
        )
        # Corrector g: (ε, Z_t) → ΔZ — learns systematic error patterns
        self.corrector = nn.Sequential(
            nn.Linear(K + K, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, K),
            nn.Tanh(),  # Bounded correction → stable
        )
        self.correction_scale = nn.Parameter(torch.tensor(0.1))  # Start conservative

    def forward(self, Z_t, O_t, Z_next=None, Z_next2=None):
        """Returns dict with predictions and correction info.

        Z_t:   (B, K) current intrinsic
        O_t:   (B, D) mutual society context
        Z_next:  (B, K) actual next intrinsic (optional, for training)
        Z_next2: (B, K) actual two-step intrinsic (optional, for training)
        """
        feat = torch.cat([Z_t, O_t], dim=-1)                     # (B, K+D)
        Z_pred_t1 = self.predictor(feat)                          # Ẑ_{t+1}

        out = {"Z_pred_t1": Z_pred_t1}

        if Z_next is not None:
            # Step 2: Error signal
            epsilon = Z_next - Z_pred_t1                          # (B, K)

            # Step 3: Error-guided correction
            corr_in = torch.cat([epsilon, Z_t], dim=-1)           # (B, 2K)
            delta_Z = self.corrector(corr_in) * self.correction_scale
            Z_corrected = Z_t + delta_Z                            # Z'_t

            # Step 4: Two-step prediction from corrected state
            feat2 = torch.cat([Z_corrected, O_t], dim=-1)
            Z_pred_t2 = self.predictor(feat2)                      # Ẑ_{t+2}

            out.update({
                "Z_pred_t2": Z_pred_t2,
                "epsilon": epsilon,
                "delta_Z": delta_Z,
                "Z_corrected": Z_corrected,
            })

        return out


# =========================================================================
# 3. Full Model Variants
# =========================================================================

class DAME_EEG(nn.Module):
    """DAME-EEG: InputProj -> [WaterCycleV2] -> [MutualSocietyV2] -> Classifier

    Unified class with component flags for clean ablation. No Fast/Slow pathways
    (DE features are pre-extracted and LDS-smoothed). No adversarial domain
    confusion (NLP lesson: small domain gaps make adversarial destructive).

    Five Philosophies (all preserved):
      P1 Dual Representation — H_seq (temporal) + Z (global intrinsic)
      P2 Water Cycle — evaporate(VIB)->precipitate(CrossAttn)->reflux(Banach)
      P3 Mutual Society — cooperative ensemble, community emergence
      P4 Pre-Migration — predict intrinsic evolution, learn invariant structure
      P5 Nonlinear IB — VIB KL with dynamic modulation

    Flags:
      use_water  — WaterCycleV2 (VIB->CrossAttn->[Reflux])
      use_reflux — Banach fixed-point iteration (within WaterCycle; ignored if !water)
      use_mutual — MutualSocietyV2 cooperative ensemble

    Ablation matrix:
      DAME-EEG:        water=1 reflux=1 mutual=1  (full)
      DAME-NoReflux:   water=1 reflux=0 mutual=1  (single-pass VIB)
      DAME-NoMutual:   water=1 reflux=1 mutual=0  (WaterCycle only)
      DAME-NoWater:    water=0 reflux=0 mutual=1  (MutualSociety raw)
      DAME-Base:       water=0 reflux=0 mutual=0  (encoder + classifier)
    """
    def __init__(self, C=62, F=5, T=WINDOW_SIZE, nc=N_CLASSES,
                 use_water=True, use_reflux=True, use_mutual=True):
        super().__init__()
        self.use_water = use_water
        self.use_reflux = use_reflux and use_water
        self.use_mutual = use_mutual

        # Input projection: (B, C*F) -> (B, D) per time step
        self.input_proj = InputProjection(C, F, D_MODEL)

        # Water cycle (P2 + P5)
        if use_water:
            self.water = WaterCycleV2(D_MODEL, K_LATENT, use_reflux=self.use_reflux)
        else:
            self.water = None

        # Mutual society (P3)
        if use_mutual:
            self.mutual = MutualSocietyV2(N_NEURONS, D_MODEL, D_MEM,
                                           n_communities=N_COMMUNITIES)
        else:
            self.mutual = None

        # Prediction head (P4: only when reflux is active)
        if self.use_reflux:
            self.pred_head = PredictionHeadV2(K_LATENT, D_MODEL)
        else:
            self.pred_head = None

        # Classifier
        self.clf = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, nc)
        )

        self._current_epoch = 0
        self._total_epochs = EPOCHS

    def set_epoch(self, epoch, total_epochs=None):
        """Temperature annealing + KL warmup + community reassignment"""
        self._current_epoch = epoch
        if total_epochs is not None:
            self._total_epochs = total_epochs

        if self.use_mutual:
            progress = min(1.0, epoch / max(TEMP_ANNEAL_EPOCHS, 1))
            temp = TEMP_INIT + (TEMP_FINAL - TEMP_INIT) * progress
            self.mutual.set_temp(temp)
            if epoch >= 3:
                self.mutual.reassign_communities()

    def forward(self, X, subject_ids=None):
        """X: (B, C, T, F) — DE features"""
        # P1: Input projection -> H_seq (temporal representation)
        H_seq = self.input_proj(X)
        H_pooled = H_seq.mean(dim=1)  # (B, D) — global state

        if self.use_water:
            # P2+P5: Water cycle — VIB -> CrossAttn -> [Reflux]
            Z, A, kl, kl_w, convergence, Z_init, reflux_mag = self.water(H_seq)
        else:
            Z = H_pooled
            A = H_pooled
            kl = torch.tensor(0.0, device=X.device)
            kl_w = torch.tensor(0.0, device=X.device)
            Z_init = H_pooled
            convergence = [0.0]
            reflux_mag = torch.tensor(0.0, device=X.device)

        if self.use_mutual:
            # P3: Mutual society on anchor A + raw features H_pooled
            O, gates, share_mask, kl_mod = self.mutual(A, H_pooled)
        else:
            O = A
            gates = torch.zeros(X.size(0), 1, device=X.device)
            kl_mod = torch.zeros(X.size(0), 1, device=X.device)

        logits = self.clf(O)

        return {
            "logits": logits,
            "kl_loss": kl,
            "kl_w": kl_w,
            "kl_mod": kl_mod,
            "Z_star": Z,
            "Z_init": Z_init,
            "O": O,
            "gates": gates,
            "convergence": convergence,
            "reflux_mag": reflux_mag if isinstance(reflux_mag, torch.Tensor) else torch.tensor(reflux_mag, device=X.device),
        }

    def compute_loss(self, out, labels, subject_ids=None, Z_next=None, Z_next2=None):
        """DAME loss with component-gated terms.

        Always: cross-entropy (label smoothing 0.1)
        If use_water: KL warmup loss (P5)
        If use_mutual: ortho + mutual + specialization + gate_entropy (P3)
        If use_reflux: pre-migration (P4) + reflux effectiveness (P2)
        """
        loss = F.cross_entropy(out["logits"], labels, label_smoothing=0.1)

        # P5: KL warmup (only with WaterCycle)
        if self.use_water:
            warmup = min(1.0, (self._current_epoch + 1) / max(KL_WARMUP_EPOCHS, 1))
            kl_val = out["kl_loss"]
            if isinstance(kl_val, torch.Tensor):
                loss = loss + KL_W * warmup * kl_val

        # P3: Mutual constraints
        if self.use_mutual:
            gates = out["gates"]
            loss = loss + ORTHO_W * self.mutual.ortho_loss()
            loss = loss + MUTUAL_W * self.mutual.mutual_loss()
            loss = loss + SPEC_W * self.mutual.specialization_loss(gates)
            loss = loss + GATE_ENTROPY_W * self.mutual.gate_entropy_loss(gates)

        # P4: Pre-migration (only when reflux is active)
        if self.use_reflux and self.pred_head is not None:
            Z_t = out["Z_init"]
            O_t = out["O"]
            pred_out = self.pred_head(Z_t, O_t, Z_next, Z_next2)

            if Z_next is not None:
                min_b1 = min(pred_out["Z_pred_t1"].size(0), Z_next.size(0))
                pred_loss = F.mse_loss(
                    pred_out["Z_pred_t1"][:min_b1],
                    Z_next[:min_b1].detach()
                )
                if Z_next2 is not None and "Z_pred_t2" in pred_out:
                    min_b2 = min(pred_out["Z_pred_t2"].size(0), Z_next2.size(0))
                    if min_b2 > 0:
                        pred_loss = pred_loss + 0.5 * F.mse_loss(
                            pred_out["Z_pred_t2"][:min_b2],
                            Z_next2[:min_b2].detach()
                        )
                if "delta_Z" in pred_out:
                    delta = pred_out["delta_Z"][:min_b1]
                    pred_loss = pred_loss + 0.001 * delta.norm() / max(min_b1, 1)
            else:
                pred_loss = F.mse_loss(pred_out["Z_pred_t1"], out["Z_star"].detach())

            loss = loss + PRED_W * pred_loss

            # P2: Encourage meaningful reflux displacement
            reflux_mag = out["reflux_mag"]
            if isinstance(reflux_mag, torch.Tensor) and reflux_mag.numel() == 1:
                loss = loss + REFLUX_W * F.relu(0.01 - reflux_mag)

        # Diagnostics
        gates = out["gates"]
        kl_mod = out["kl_mod"]
        gate_weighted_kl = (gates * kl_mod).sum(-1) / gates.sum(-1).clamp(min=1)
        n_active = gates.gt(0.5).float().sum(-1).mean().item()

        kl_v = out["kl_loss"].item() if isinstance(out["kl_loss"], torch.Tensor) else 0.0
        reflux_v = out["reflux_mag"].item() if isinstance(out["reflux_mag"], torch.Tensor) else 0.0

        stats = {
            "total": loss.item(),
            "kl": kl_v,
            "kl_m": gate_weighted_kl.mean().item(),
            "n_active": n_active,
            "reflux": reflux_v,
        }
        return loss, stats




# =================================================================================
# 4. Baseline Models (BCI Transfer Learning)
# =================================================================================

class DAME_Base(nn.Module):
    """Parameter-aligned baseline: InputProjection + Classifier only.

    No WaterCycle, no MutualSociety, no pre-migration.
    Pure encoder+classifier — tests whether DAME mechanisms add value
    beyond a simple deep classifier.
    """
    def __init__(self, C=62, n_bands=5, T=WINDOW_SIZE, nc=N_CLASSES):
        super().__init__()
        self.input_proj = InputProjection(C, n_bands, D_MODEL)

        self.clf = nn.Sequential(
            nn.Linear(D_MODEL, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, nc)
        )

    def forward(self, X):
        H = self.input_proj(X)          # (B, T, D)
        H = H.mean(dim=1)              # (B, D)
        return self.clf(H)




class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018) — standard compact BCI baseline"""
    def __init__(self, C=62, T=WINDOW_SIZE, nc=N_CLASSES, F1=16, D=2, F2=32):
        super().__init__()
        # Adaptive pooling to handle variable T
        pool1 = min(4, max(1, T // 4))
        pool2 = min(8, max(1, T // pool1 // 2))
        # Block 1: temporal → spatial depthwise
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, min(15, T//2+1)), padding=(0, min(7, T//4))),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, pool1)), nn.Dropout(0.25)
        )
        # Block 2: separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, min(15, T//pool1//2+1)),
                     padding=(0, min(7, T//pool1//4)), groups=F1 * D),
            nn.Conv2d(F1 * D, F2, (1, 1)), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, pool2)), nn.Dropout(0.25)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            self.fd = self.block2(self.block1(dummy)).numel()
        self.clf = nn.Linear(self.fd, nc)

    def forward(self, X):
        # X: (B, C, T, F) — take mean over frequency bands for EEGNet
        x = X.mean(dim=-1).unsqueeze(1)  # (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        return self.clf(x.flatten(1))


class TSception(nn.Module):
    """TSception (Ding et al., 2022): Multi-scale temporal + spatial attention"""
    def __init__(self, C=62, T=WINDOW_SIZE, nc=N_CLASSES):
        super().__init__()
        # Multi-scale temporal convs
        self.tconv1 = nn.Conv2d(1, 16, (1, 11), padding=(0, 5))
        self.tconv2 = nn.Conv2d(1, 16, (1, 7), padding=(0, 3))
        self.tconv3 = nn.Conv2d(1, 16, (1, 5), padding=(0, 2))
        # Spatial conv (electrode-wise)
        self.sconv = nn.Conv2d(48, 48, (C, 1))
        self.bn = nn.BatchNorm2d(48)
        # Asymmetric attention
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(48, 12, 1),
            nn.ReLU(), nn.Conv2d(12, 48, 1), nn.Sigmoid()
        )
        self.clf = nn.Linear(48, nc)

    def forward(self, X):
        x = X.mean(dim=-1).unsqueeze(1)  # (B, 1, C, T)
        h1 = self.tconv1(x); h2 = self.tconv2(x); h3 = self.tconv3(x)
        h = torch.cat([h1, h2, h3], dim=1)
        h = F.elu(self.bn(self.sconv(h)))
        a = self.attn(h)
        h = (h * a).mean(dim=[2, 3])
        return self.clf(h)


class DGCNN(nn.Module):
    """DGCNN (Song et al., 2018): Dynamic Graph CNN — learns adjacency from data"""
    def __init__(self, C=62, T=WINDOW_SIZE, nc=N_CLASSES):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 11), padding=(0, 5)),
            nn.BatchNorm2d(32), nn.ELU()
        )
        self.gconv1 = nn.Sequential(
            nn.Conv1d(C * 32, 64, 1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.gconv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(128, nc)

    def forward(self, X):
        x = X.mean(dim=-1).unsqueeze(1)  # (B, 1, C, T)
        x = self.tconv(x)                 # (B, 32, C, T_out)
        B, F_ch, C_out, T_out = x.shape
        x = x.reshape(B, F_ch * C_out, T_out)
        x = self.gconv1(x); x = self.gconv2(x)
        x = self.pool(x).squeeze(-1)
        return self.clf(x)


class EEGConformer(nn.Module):
    """EEGConformer (Song et al., 2022): CNN + Transformer for EEG.

    Simplified port of the key idea: multi-scale CNN → Transformer encoder →
    mean pool → classify. Represents the SOTA CNN-Transformer hybrid paradigm
    that dominates recent BCI benchmarks.
    """
    def __init__(self, C=62, T=WINDOW_SIZE, nc=N_CLASSES, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        # Multi-scale temporal conv
        self.conv1 = nn.Conv2d(1, d_model // 2, (1, 15), padding=(0, 7))
        self.conv2 = nn.Conv2d(1, d_model // 2, (1, 5), padding=(0, 2))
        self.bn = nn.BatchNorm2d(d_model)
        # Spatial compression
        self.spatial_conv = nn.Conv2d(d_model, d_model, (C, 1))
        self.bn2 = nn.BatchNorm2d(d_model)
        # Positional encoding + Transformer
        self.pos_embed = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.clf = nn.Linear(d_model, nc)

    def forward(self, X):
        x = X.mean(dim=-1).unsqueeze(1)  # (B, 1, C, T)
        h1 = self.conv1(x); h2 = self.conv2(x)
        h = torch.cat([h1, h2], dim=1)  # (B, d_model, C, T')
        h = F.gelu(self.bn(h))
        h = self.bn2(self.spatial_conv(h)).squeeze(2)  # (B, d_model, T')
        h = h.transpose(1, 2)  # (B, T', d_model)
        T_out = h.size(1)
        h = h + self.pos_embed[:, :T_out, :]
        h = self.transformer(h)  # (B, T', d_model)
        h = h.mean(dim=1)  # (B, d_model)
        return self.clf(h)


class DANN_EEG(nn.Module):
    """Domain-Adversarial Neural Network — subject as domain"""
    def __init__(self, C=62, T=WINDOW_SIZE, nc=N_CLASSES, n_subjects=15):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (1, 15), padding=(0, 7)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, (1, 15), padding=(0, 7)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8))
        )
        with torch.no_grad():
            self.fd = self.feature(torch.zeros(1, 1, C, T)).numel()
        self.task_clf = nn.Linear(self.fd, nc)
        self.domain_clf = nn.Sequential(
            nn.Linear(self.fd, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_subjects)
        )

    def forward(self, X, subject_ids=None, lam=1.0):
        x = X.mean(dim=-1).unsqueeze(1)
        f = self.feature(x).flatten(1)
        logits = self.task_clf(f)
        out = {"logits": logits}
        if subject_ids is not None:
            rev = GradReverse.apply(f, lam)
            out["domain_logits"] = self.domain_clf(rev)
            out["domain_labels"] = subject_ids
        return out


class DeepCORAL_EEG(nn.Module):
    """Deep CORAL — second-order statistics alignment across subjects"""
    def __init__(self, C=62, T=WINDOW_SIZE, nc=N_CLASSES):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (1, 15), padding=(0, 7)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, (1, 15), padding=(0, 7)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8))
        )
        with torch.no_grad():
            self.fd = self.feature(torch.zeros(1, 1, C, T)).numel()
        self.clf = nn.Linear(self.fd, nc)

    def forward(self, X):
        x = X.mean(dim=-1).unsqueeze(1)
        f = self.feature(x).flatten(1)
        return {"logits": self.clf(f), "features": f}

    @staticmethod
    def coral_loss(xs, xt):
        d = xs.size(1)
        cs = (xs.T @ xs) / (xs.size(0) - 1)
        ct = (xt.T @ xt) / (xt.size(0) - 1)
        return (cs - ct).pow(2).sum() / (4 * d * d)


# =========================================================================
# 5. Training & Evaluation
# =========================================================================

def train_epoch_dame(model, dl, opt):
    """Train one epoch with pre-migration (P4).

    Within-window prediction: Z_init → Z_star captures reflux convergence dynamics.
    Cross-window (Z_t→Z_{t+1}) requires trial-aware sampler — shuffled batches mix
    different subjects/trials, making "next window" semantically meaningless.
    The fallback in compute_loss handles this correctly.
    """
    model.train()
    total_loss = 0
    stats_sum = {}

    for Xb, yb, sb in dl:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        sb = sb.to(DEVICE)
        opt.zero_grad()

        # Forward pass on current window
        out = model(Xb, sb)

        # P4: within-window Z_init→Z_star prediction (reflux dynamics).
        # Cross-window prediction needs trial-contiguous sampling — TODO for v4.
        loss, stats = model.compute_loss(out, yb, sb, Z_next=None, Z_next2=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        for k, v in stats.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + float(v)

    n = max(len(dl), 1)
    return total_loss / n, {k: v / n for k, v in stats_sum.items()}


def train_epoch_baseline(model, dl, opt, is_dann=False, dann_lam=1.0):
    """Train one epoch for baseline models"""
    model.train()
    total_loss = 0

    for Xb, yb, sb in dl:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        sb = sb.to(DEVICE)
        opt.zero_grad()

        if is_dann:
            out = model(Xb, sb, lam=dann_lam)
            loss = F.cross_entropy(out["logits"], yb, label_smoothing=0.1)
            if "domain_logits" in out:
                loss = loss + 0.1 * F.cross_entropy(out["domain_logits"], out["domain_labels"])
        elif isinstance(model, DeepCORAL_EEG):
            out = model(Xb)
            loss = F.cross_entropy(out["logits"], yb, label_smoothing=0.1)
            # CORAL loss between random subject splits within batch
            unique_s = sb.unique()
            if len(unique_s) >= 2:
                s0, s1 = unique_s[0], unique_s[1]
                f0 = out["features"][sb == s0]
                f1 = out["features"][sb == s1]
                if f0.size(0) > 1 and f1.size(0) > 1:
                    loss = loss + 0.1 * DeepCORAL_EEG.coral_loss(f0, f1)
        else:
            if isinstance(model, (EEGNet, TSception, DGCNN, DAME_Base, EEGConformer)):
                logits = model(Xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            else:
                out = model(Xb)
                loss = F.cross_entropy(out["logits"], yb, label_smoothing=0.1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

    return total_loss / max(len(dl), 1)


@torch.no_grad()
def evaluate(model, dl, is_dame=False):
    """Evaluate on test set"""
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for Xb, yb, sb in dl:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        sb = sb.to(DEVICE)

        if is_dame:
            out = model(Xb)
            preds = out["logits"].argmax(-1)
        elif isinstance(model, (EEGNet, TSception, DGCNN, DAME_Base, EEGConformer)):
            preds = model(Xb).argmax(-1)
        elif isinstance(model, (DANN_EEG, DeepCORAL_EEG)):
            preds = model(Xb)["logits"].argmax(-1)
        else:
            preds = model(Xb)["logits"].argmax(-1)

        correct += (preds == yb).sum().item()
        total += yb.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Per-class accuracy
    per_class = {}
    for c in range(N_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[c] = (all_preds[mask] == c).float().mean().item()

    return correct / max(total, 1), per_class, all_preds, all_labels


# =========================================================================
# 6. LOSO Cross-Validation
# =========================================================================

def loso_eval(model_factory, X, y, subj, epochs=EPOCHS, bs=BATCH_SIZE, lr=LR,
              is_dame=False, is_dann=False, is_coral=False, verbose=True):
    """Leave-One-Subject-Out evaluation

    Returns: (mean_acc, std_acc, min_acc, f1_macro, kappa)
    """
    n_subjects = int(subj.max().item()) + 1
    accs, f1s, kappas = [], [], []

    for s in range(n_subjects):
        test_mask = subj == s
        train_mask = ~test_mask

        Xt, yt = X[train_mask], y[train_mask]
        Xe, ye = X[test_mask], y[test_mask]
        st = subj[train_mask]

        # Build model
        model = model_factory().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=2)

        tr_ds = SEEDDataset(Xt, yt, st)
        tr_dl = DataLoader(tr_ds, bs, shuffle=True)
        te_ds = SEEDDataset(Xe, ye, torch.full_like(ye, s))
        te_dl = DataLoader(te_ds, bs * 2, shuffle=False)

        for ep in range(epochs):
            if is_dame:
                if hasattr(model, 'set_epoch'):
                    model.set_epoch(ep, epochs)
                train_epoch_dame(model, tr_dl, opt)
            else:
                dann_lam = min(1.0, 2.0 * ep / max(1, epochs - 3)) if is_dann else 0
                train_epoch_baseline(model, tr_dl, opt, is_dann=is_dann, dann_lam=dann_lam)
            sched.step()

        acc, per_class, preds, labels = evaluate(model, te_dl, is_dame=is_dame)
        accs.append(acc)

        try:
            from sklearn.metrics import f1_score, cohen_kappa_score
            f1s.append(f1_score(labels, preds, average='macro'))
            kappas.append(cohen_kappa_score(labels, preds))
        except ImportError:
            f1s.append(0)
            kappas.append(0)

        if verbose:
            print(f"  [{s+1}/{n_subjects}] acc={acc:.4f} "
                  f"per_class={ {k: f'{v:.3f}' for k, v in per_class.items()} }")

    try:
        from sklearn.metrics import f1_score, cohen_kappa_score
    except ImportError:
        pass

    return (np.mean(accs), np.std(accs), np.min(accs),
            np.mean(f1s) if f1s else 0, np.mean(kappas) if kappas else 0)


# =========================================================================
# 7. Main Experiment
# =========================================================================

def compute_statistical_tests(all_seed_results, model_names, seeds):
    """Compute paired t-tests, Cohen's d, and bootstrap CIs.

    Returns dict: model_name → {mean, std, ci95, cohens_d_vs_best, p_vs_best}
    """
    stats = {}
    best_model = max(model_names, key=lambda m: np.mean(
        [all_seed_results[s][m]["mean"] for s in seeds]))

    for mname in model_names:
        accs = np.array([all_seed_results[s][mname]["mean"] for s in seeds])
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0

        # Bootstrap 95% CI
        if len(accs) >= 3:
            np.random.seed(42)
            boots = [np.mean(np.random.choice(accs, size=len(accs), replace=True))
                     for _ in range(10000)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        else:
            ci_lo, ci_hi = mean_acc - 2*std_acc, mean_acc + 2*std_acc

        # Paired t-test vs best model
        best_accs = np.array([all_seed_results[s][best_model]["mean"] for s in seeds])
        if len(accs) >= 3 and mname != best_model:
            try:
                from scipy.stats import ttest_rel
                _, p_val = ttest_rel(best_accs, accs)
            except ImportError:
                p_val = 1.0
        else:
            p_val = 1.0

        # Cohen's d vs best
        diff = best_accs - accs
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8) if len(diff) > 1 else 0

        # Per-seed F1 and kappa
        f1s = np.array([all_seed_results[s][mname].get("f1", 0) for s in seeds])
        kps = np.array([all_seed_results[s][mname].get("kappa", 0) for s in seeds])

        stats[mname] = {
            "mean": float(mean_acc), "std": float(std_acc),
            "ci95": [float(ci_lo), float(ci_hi)],
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0,
            "kappa_mean": float(np.mean(kps)), "kappa_std": float(np.std(kps, ddof=1)) if len(kps) > 1 else 0,
            "cohens_d": float(d), "p_vs_best": float(p_val),
            "seeds": [float(a) for a in accs],
        }
    return stats


def compute_ablation_delta(all_seed_results, model_a, model_b, seeds, n_bootstrap=10000):
    """Bootstrap CI for ablation delta: model_a − model_b."""
    a = np.array([all_seed_results[s][model_a]["mean"] for s in seeds])
    b = np.array([all_seed_results[s][model_b]["mean"] for s in seeds])
    delta = np.mean(a - b)
    np.random.seed(42)
    boots = [np.mean(np.random.choice(a - b, size=len(seeds), replace=True))
             for _ in range(n_bootstrap)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    # One-sided: is delta significantly > 0?
    p_one_sided = np.mean(np.array(boots) <= 0)
    return {"delta": float(delta), "ci95": [float(ci_lo), float(ci_hi)],
            "p_one_sided": float(p_one_sided), "significant": bool(p_one_sided < 0.05)}


def run_experiments(X, y, subj, quick=False, level2=False):
    """Run all models with multi-seed LOSO evaluation

    Matches NLP experiment pattern: 3 seeds, 8 models, full statistics
    """
    if level2:
        epochs = LEVEL2_EPOCHS
        n_subjects_use = LEVEL2_SUBJECTS
    elif quick:
        epochs = QUICK_EPOCHS
        n_subjects_use = QUICK_SUBJECTS
    else:
        epochs = EPOCHS
        n_subjects_use = int(subj.max().item()) + 1

    seeds = ALL_SEEDS[:1] if (quick or level2) else ALL_SEEDS[:N_SEEDS]

    if quick or level2:
        # Use subset of subjects and data
        mask = subj < n_subjects_use
        X, y, subj = X[mask], y[mask], subj[mask]
        tag = "QUICK" if quick else "LEVEL-2"
        print(f"[{tag}] {n_subjects_use} subjects, {epochs} epochs, 1 seed")

    C, T, F_bands = X.shape[1], X.shape[2], X.shape[3]
    n_subjects_total = int(subj.max().item()) + 1
    print(f"Data: {X.shape[0]} windows, {C}ch × {T}tp × {F_bands}bands, "
          f"{n_subjects_total} subjects, {N_CLASSES} classes")

    # Model registry
    model_specs = [
        ("DAME-EEG",        lambda: DAME_EEG(C, F_bands, T, N_CLASSES, True, True, True),     "dame"),
        ("DAME-NoReflux",   lambda: DAME_EEG(C, F_bands, T, N_CLASSES, True, False, True),    "dame"),
        ("DAME-NoMutual",   lambda: DAME_EEG(C, F_bands, T, N_CLASSES, True, True, False),    "dame"),
        ("DAME-NoWater",    lambda: DAME_EEG(C, F_bands, T, N_CLASSES, False, False, True),   "dame"),
        ("DAME-Base",       lambda: DAME_Base(C, F_bands, T, N_CLASSES),                       "base"),
        ("EEGConformer",    lambda: EEGConformer(C, T, N_CLASSES),                                 "base"),
        ("EEGNet",          lambda: EEGNet(C, T, N_CLASSES),                                       "base"),
        ("TSception",       lambda: TSception(C, T, N_CLASSES),                                    "base"),
        ("DGCNN",           lambda: DGCNN(C, T, N_CLASSES),                                        "base"),
        ("DANN",            lambda: DANN_EEG(C, T, N_CLASSES, n_subjects_total),                   "dann"),
        ("DeepCORAL",       lambda: DeepCORAL_EEG(C, T, N_CLASSES),                                "coral"),
    ]

    # Level 2: focused ablation — DAME variants + key baselines only
    if level2:
        l2_keep = {"DAME-EEG", "DAME-NoReflux", "DAME-NoMutual", "DAME-NoWater",
                   "DAME-Base", "EEGConformer", "EEGNet", "DeepCORAL"}
        model_specs = [(n, f, t) for n, f, t in model_specs if n in l2_keep]

    model_names = [m[0] for m in model_specs]

    all_seed_results = {}

    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        tag = f"[Seed {seed} {seed_idx+1}/{len(seeds)}]"
        print(f"\n{'='*60}\n  {tag}\n{'='*60}")

        seed_results = {}
        for mname, factory, mtype in model_specs:
            t0 = time.time()
            is_dame = (mtype == "dame")
            is_dann = (mtype == "dann")
            is_coral = (mtype == "coral")

            print(f"\n  [{mname}]")
            mu, std, mi, f1, kp = loso_eval(
                factory, X, y, subj, epochs=epochs, bs=BATCH_SIZE, lr=LR,
                is_dame=is_dame, is_dann=is_dann, is_coral=is_coral,
                verbose=(not (quick or level2) or mname.startswith("DAME"))
            )
            elapsed = time.time() - t0

            print(f"  {mname:<16s}: {mu*100:5.1f}±{std*100:.1f}% "
                  f"| F1={f1:.3f} | κ={kp:.3f} | min={mi*100:.1f}% "
                  f"| {elapsed:.0f}s")

            seed_results[mname] = {
                "mean": float(mu), "std": float(std), "min": float(mi),
                "f1": float(f1), "kappa": float(kp), "time": elapsed
            }

        all_seed_results[seed] = seed_results

        # Save after each seed
        save_path = os.path.join(RESULTS_DIR, f"eeg_v3_seed{seed}.json")
        json.dump({str(s): {k: v["mean"] for k, v in r.items()}
                   for s, r in all_seed_results.items()},
                  open(save_path, "w"), indent=2)

    # --- Compute full statistics with significance tests ---
    stat_results = compute_statistical_tests(all_seed_results, model_names, seeds)

    # --- Final cross-seed statistics ---
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS — {len(seeds)} seeds × {n_subjects_total} subjects × {len(model_specs)} models")
    print(f"{'='*80}")
    header = f"  {'Model':<18s} {'Acc±Std':<14s} {'95% CI':<18s} {'F1':<8s} {'κ':<8s} {'p(vs best)':<10s} {'Cohen d':<8s}"
    print(header)
    print(f"  {'-'*78}")

    best_model = max(model_names, key=lambda m: stat_results[m]["mean"])
    base_mean = stat_results["EEGNet"]["mean"]

    for mname in model_names:
        s = stat_results[mname]
        marker = " ★" if mname == best_model else ""
        ci_str = f"[{s['ci95'][0]*100:.1f}, {s['ci95'][1]*100:.1f}]"
        p_str = f"{s['p_vs_best']:.4f}" if s['p_vs_best'] < 1.0 else "—"
        d_str = f"{s['cohens_d']:.2f}" if mname != best_model else "—"
        print(f"  {mname:<18s} {s['mean']*100:5.1f}±{s['std']*100:.2f}%  {ci_str:<18s} "
              f"{s['f1_mean']:.3f}   {s['kappa_mean']:.3f}   {p_str:<10s} {d_str:<8s}{marker}")

    # --- Ablation Analysis with Bootstrap CIs ---
    print(f"\n  {'Ablation Analysis (bootstrap 95% CI)':-^60}")
    dame_key = "DAME-EEG"

    ablation_pairs = [
        ("Reflux",        "DAME-NoReflux"),
        ("MutualSociety", "DAME-NoMutual"),
        ("WaterCycle",    "DAME-NoWater"),
        ("vs DAME-Base",  "DAME-Base"),
        ("vs EEGConformer","EEGConformer"),
        ("vs DANN",       "DANN"),
        ("vs DeepCORAL",  "DeepCORAL"),
    ]

    ablation_results = {}
    for label, baseline in ablation_pairs:
        if baseline in stat_results:
            delta = compute_ablation_delta(all_seed_results, dame_key, baseline, seeds)
            ablation_results[label] = delta
            sig = " *" if delta["significant"] else ""
            print(f"  {label:<20s}: {delta['delta']*100:+5.2f}%  "
                  f"95% CI [{delta['ci95'][0]*100:+.2f}, {delta['ci95'][1]*100:+.2f}]%  "
                  f"p={delta['p_one_sided']:.3f}{sig}")

    # --- Best baseline comparison ---
    baseline_models = ["EEGConformer", "EEGNet", "TSception", "DGCNN", "DAME-Base"]
    baseline_best = max((stat_results[m]["mean"] for m in baseline_models if m in stat_results), default=0)
    dame_mean = stat_results[dame_key]["mean"]
    print(f"\n  vs Best non-DAME baseline: +{(dame_mean - baseline_best)*100:.2f}%")

    # --- Save comprehensive results ---
    save_path = os.path.join(RESULTS_DIR, "eeg_v3_final.json")
    json.dump({
        "config": {
            "dataset": "SEED-IV", "n_subjects": n_subjects_total,
            "n_seeds": len(seeds), "epochs": epochs,
            "D": D_MODEL, "K": K_LATENT, "N": N_NEURONS,
            "C_communities": N_COMMUNITIES, "d_mem": D_MEM,
        },
        "per_seed": {str(s): {k: v["mean"] for k, v in r.items()}
                     for s, r in all_seed_results.items()},
        "statistics": stat_results,
        "ablation": ablation_results,
    }, open(save_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nResults saved: {save_path}")

    return stat_results


# =========================================================================
# 8. Main Entry Point
# =========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEG V3: DAME for SEED-IV")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test: 1 seed, 3 subjects, 3 epochs")
    parser.add_argument("--level2", action="store_true",
                       help="Level 2 focused ablation: 1 seed, 5 subjects, 8 epochs")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--seed", type=int, default=None,
                       help="Single seed override")
    parser.add_argument("--model", type=str, default=None,
                       help="Run single model only (e.g., DAME-EEG)")
    args = parser.parse_args()

    if args.quick:
        QUICK_TEST = True
    if args.level2:
        QUICK_TEST = False  # level2 overrides quick

    print("=" * 70)
    print("  EEG V3 — DAME Architecture for SEED-IV Emotion Recognition")
    print(f"  Fast+Slow Pathways + WaterCycleV2 + MutualSocietyV2")
    print(f"  D={D_MODEL} K={K_LATENT} N={N_NEURONS} C={N_COMMUNITIES} dm={D_MEM}")
    if args.level2:
        print(f"  10 models × LOSO × 1-seed (Level 2 focused ablation)")
    else:
        print(f"  11 models × LOSO × {N_SEEDS}-seed (ICML experimental standard)")
    print("=" * 70)

    # Load data
    print("\n[1/2] Loading SEED-IV dataset...")
    X, y, subj, sess = load_seed_iv()

    # Run experiments
    print("\n[2/2] Running experiments...")
    results = run_experiments(X, y, subj, quick=args.quick, level2=args.level2)

    print("\nDONE. Results in:", RESULTS_DIR)
