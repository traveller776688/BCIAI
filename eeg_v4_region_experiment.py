#!/usr/bin/env python3
"""
EEG V4 — Region-Coupling DAME for Cross-Subject Emotion Transfer
=================================================================
核心科学主张（用户主线）：情绪的神经表征是**脑区间的联动耦合**（大尺度网络、
默认模式网络DMN级别），而非逐个电极的节点活动。跨被试迁移 = 学习"忽略被试
差异的情绪特征"——脑区耦合动力学的演化规律跨被试不变，单电极信号细节因被试
而异。故：表示放在脑区耦合层面 → 天然获得被试不变性（以不变应万变）。

架构（纯DAME，无对抗，五哲学不变）:
  RAW 62ch (B,62,800@200Hz)
    → 12脑区分组 (PFC=DMN前枢纽/FL/FR/FC/C/TL/TR/CP=PCC后枢纽/P/PO/O/CB)
    → 5频段FFT带通 (δθαβγ) → Hilbert相位
    → 脑区间PLV耦合序列 (0.5s子窗, 8步) + 区域功率   [P1双重表示]
    → WaterCycleV2: VIB蒸馏→CrossAttn→Banach回流      [P2/P5]
    → MutualSocietyV2: 神经元=功能组件, W_mutual=组件间耦合, 社区涌现 [P3]
    → PredictionHeadV2: 预测4s后耦合本征演化           [P4预迁移]
    → Classifier: 4类情绪

协议: 跨被试LOSO (15折, 3-5种子), 单数据集, 原始波形
消融: 脑区耦合vs单节点 / 耦合vs无耦合 / 预测vs无预测 / 机制消融
基线: EEGNet, TSception, DGCNN, EEGConformer, LMDA-Net, DANN, DeepCORAL
机制权重: 全激活 (REFLUX_W=0.01, PRED_W=0.01, ORTHO/MUTUAL/SPEC=1e-3)
"""
import math, json, random, time, os, sys, warnings
from collections import Counter, defaultdict

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.signal import decimate

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEED_DIR = os.path.join(BASE_DIR, "SEED_IV", "SEED_IV")
RAW_DIR = os.path.join(BASE_DIR, "SEED_IV", "eeg_raw_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# Reuse DAME core components (WaterCycleV2 / MutualSocietyV2 / PredictionHeadV2)
sys.path.insert(0, BASE_DIR)
from eeg_v3_experiment import (WaterCycleV2, MutualSocietyV2, PredictionHeadV2,
                               SESSION_LABELS, D_MODEL, K_LATENT, N_NEURONS,
                               N_COMMUNITIES, D_MEM, N_CLASSES,
                               TEMP_INIT, TEMP_FINAL, TEMP_ANNEAL_EPOCHS)

# =========================================================================
# CONFIG — mechanisms FULLY ACTIVE (v3 lesson: inert weights = circular ablations)
# =========================================================================
FS_TARGET = 200           # decimated rate
WINDOW_SAMPLES = 4 * FS_TARGET   # 800 = 4s
WINDOW_STRIDE = 2 * FS_TARGET    # 400 = 2s hop
SUB_WIN = FS_TARGET // 2         # 100 = 0.5s coupling resolution
T_COUP = WINDOW_SAMPLES // SUB_WIN   # 8 coupling steps per window
PRED_GAP = 2              # prediction target = window k+2 (4s later, zero overlap)

ORTHO_W = 1e-3            # was 0.0005 → restored
MUTUAL_W = 1e-3           # was 0.0001 → restored
SPEC_W = 1e-3             # was 0.0002 → restored
GATE_ENTROPY_W = 1e-3     # was 0.0001 → restored
REFLUX_W = 0.01           # was 0.0 → restored (encourage meaningful fixed-point displacement)
PRED_W = 0.01             # was 0.0 → restored (pre-migration core mechanism)
KL_W = 0.008              # learnable log_kl_w inside WaterCycleV2, init ln(0.008)

LR = 2e-4
EPOCHS = 15
BATCH_SIZE = 64
KL_WARMUP_EPOCHS = 8
ALL_SEEDS = [42, 123, 789]

# =========================================================================
# 1. 脑区定义 — 62ch extended 10-20 → 12 cortical regions
#    PFC = DMN前枢纽(mPFC), CP = DMN后枢纽(PCC/楔前叶), O = 视觉网络
# =========================================================================
CH_ORDER = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
            'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
            'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
            'PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']

REGIONS = {
    'PFC': ['FP1','FPZ','FP2','AF3','AF4'],                          # 前额皮层 — DMN前枢纽(mPFC)
    'FL':  ['F7','F5','F3','F1','FZ'],                               # 额叶左
    'FR':  ['F2','F4','F6','F8'],                                    # 额叶右
    'FC':  ['FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8'],  # 额中/前扣带
    'C':   ['C5','C3','C1','CZ','C2','C4','C6'],                     # 中央感觉运动
    'TL':  ['T7','TP7'],                                             # 颞叶左
    'TR':  ['T8','TP8'],                                             # 颞叶右
    'CP':  ['CP5','CP3','CP1','CPZ','CP2','CP4','CP6'],              # 顶中 — DMN后枢纽(PCC)
    'P':   ['P7','P5','P3','P1','PZ','P2','P4','P6','P8'],           # 顶叶/楔前叶
    'PO':  ['PO7','PO5','PO3','POZ','PO4','PO6','PO8'],              # 顶枕
    'O':   ['O1','OZ','O2'],                                         # 枕叶 — 视觉网络
    'CB':  ['CB1','CB2'],                                            # 小脑
}
REGION_NAMES = list(REGIONS.keys())
REGION_GROUPS = [[CH_ORDER.index(ch) for ch in REGIONS[name]] for name in REGION_NAMES]
N_REGIONS = len(REGION_GROUPS)
assert sum(len(g) for g in REGION_GROUPS) == 62, "region grouping must cover all 62ch"

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 45)}
BAND_NAMES = list(BANDS.keys())
N_BANDS = len(BANDS)

def make_pairs(R):
    """Upper-triangle region pairs (undirected coupling)."""
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    return [p[0] for p in pairs], [p[1] for p in pairs]

# =========================================================================
# 2. 信号处理 — FFT带通 + Hilbert相位 + 脑区PLV耦合 (全可微)
# =========================================================================
def bandpass_fft(x, lo, hi, fs=FS_TARGET):
    """Zero-phase band-pass via FFT masking. x: (..., T) real."""
    f = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / fs).to(x.device)
    mask = ((freqs >= lo) & (freqs < hi)).float()
    return torch.fft.irfft(f * mask.view(*([1] * (x.dim() - 1)), -1), n=x.size(-1), dim=-1)

def hilbert_analytic(x):
    """Analytic signal via FFT. x: (..., T) real → (..., T) complex."""
    f = torch.fft.rfft(x, dim=-1)
    mult = torch.ones(f.size(-1), device=x.device)
    mult[1:-1] = 2.0
    return torch.fft.irfft(f * mult, n=x.size(-1), dim=-1)

def plv_coupling(reg, pair_i, pair_j, chunk=64):
    """Region-level phase-locking value over 0.5s sub-windows.

    reg: (B, R, T) region signals (200Hz)
    Returns: plv (B, N_BANDS, N_PAIRS, T_coup), power (B, N_BANDS, R, T_coup)
    """
    B, R, T = reg.shape
    P = len(pair_i)
    T_c = T // SUB_WIN
    plv_all, pow_all = [], []
    for (lo, hi) in BANDS.values():
        xb = bandpass_fft(reg, lo, hi)                       # (B, R, T)
        # Band power per sub-window
        pw = (xb ** 2).reshape(B, R, T_c, SUB_WIN).mean(-1)  # (B, R, T_c)
        pow_all.append(pw)
        ph = torch.angle(hilbert_analytic(xb))               # (B, R, T) instant phase
        # PLV per pair, chunked to bound memory (node variant has 1891 pairs)
        plv_pairs = []
        for c in range(0, P, chunk):
            i1 = pair_i[c:c + chunk]; i2 = pair_j[c:c + chunk]
            d = ph[:, i1, :] - ph[:, i2, :]                  # (B, ch, T)
            cosv = torch.cos(d).reshape(B, -1, T_c, SUB_WIN).mean(-1)
            sinv = torch.sin(d).reshape(B, -1, T_c, SUB_WIN).mean(-1)
            plv_pairs.append(torch.sqrt(cosv ** 2 + sinv ** 2))  # (B, ch, T_c)
        plv_all.append(torch.cat(plv_pairs, dim=1))          # (B, P, T_c)
    return torch.stack(plv_all, dim=1), torch.stack(pow_all, dim=1)  # (B,F,P,T_c), (B,F,R,T_c)

class RegionCoupling(nn.Module):
    """62ch raw → region coupling tokens (P1: 耦合 + 活动双重表示)."""

    def __init__(self, groups, D=D_MODEL, use_coupling=True):
        super().__init__()
        self.groups = groups
        self.R = len(groups)
        self.use_coupling = use_coupling
        self.pair_i, self.pair_j = make_pairs(self.R)
        P = len(self.pair_i)
        d_in = N_BANDS * P + N_BANDS * self.R if use_coupling else N_BANDS * self.R
        self.proj = nn.Sequential(
            nn.Linear(d_in, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D))
        self.pos_embed = nn.Parameter(torch.randn(1, T_COUP, D) * 0.02)

    def forward(self, x):
        """x: (B, 62, T) raw → tokens (B, T_c, D) + feats for diagnostics."""
        B, C, T = x.shape
        # Region signals: mean over group channels
        reg = torch.stack([x[:, g].mean(1) for g in self.groups], dim=1)  # (B, R, T)
        plv, power = plv_coupling(reg, self.pair_i, self.pair_j)
        # plv (B,F,P,Tc); power (B,F,R,Tc) — per-band region activity
        if self.use_coupling:
            feats = torch.cat([
                plv.permute(0, 3, 1, 2).reshape(B, T_COUP, -1),   # (B,Tc,F*P)
                power.permute(0, 3, 1, 2).reshape(B, T_COUP, -1), # (B,Tc,F*R)
            ], dim=-1)
        else:
            feats = power.permute(0, 3, 1, 2).reshape(B, T_COUP, -1)  # (B,Tc,F*R)
        tok = self.proj(feats) + self.pos_embed
        return tok, feats, plv  # plv (B,F,P,Tc) for DMN diagnostics

# =========================================================================
# 3. DAME-Region — 脑区耦合 + 水循环 + 互助社会 + 预迁移
# =========================================================================
class DAME_Region(nn.Module):
    """Region-coupling DAME. Same five philosophies, region-level representation.

    Flags:
      use_coupling — PLV inter-region coupling (vs region activity only)
      use_water / use_reflux / use_mutual / use_pred — mechanism ablations
    """

    def __init__(self, groups=REGION_GROUPS, D=D_MODEL, use_coupling=True,
                 use_water=True, use_reflux=True, use_mutual=True, use_pred=True):
        super().__init__()
        self.use_coupling = use_coupling
        self.use_water = use_water
        self.use_reflux = use_reflux and use_water
        self.use_mutual = use_mutual
        self.use_pred = use_pred and self.use_reflux

        self.coupling = RegionCoupling(groups, D, use_coupling=use_coupling)
        self.water = (WaterCycleV2(D, K_LATENT, use_reflux=self.use_reflux)
                      if use_water else None)
        self.mutual = (MutualSocietyV2(N_NEURONS, D, D_MEM, n_communities=N_COMMUNITIES)
                       if use_mutual else None)
        self.pred_head = (PredictionHeadV2(K_LATENT, D) if self.use_pred else None)
        self.clf = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, N_CLASSES))

        self._current_epoch = 0

    def set_epoch(self, epoch, total_epochs=None):
        self._current_epoch = epoch
        if self.use_mutual:
            progress = min(1.0, epoch / max(TEMP_ANNEAL_EPOCHS, 1))
            self.mutual.set_temp(TEMP_INIT + (TEMP_FINAL - TEMP_INIT) * progress)
            if epoch >= 3:
                self.mutual.reassign_communities()

    def forward(self, X, subject_ids=None):
        """X: (B, 62, 800) raw window."""
        H_seq, feats, plv = self.coupling(X)                # (B, Tc, D)

        if self.use_water:
            Z, A, kl, kl_w, convergence, Z_init, reflux_mag = self.water(H_seq)
        else:
            H_pooled = H_seq.mean(dim=1)
            Z = A = H_pooled
            kl = kl_w = torch.tensor(0.0, device=X.device)
            Z_init = H_pooled
            convergence = [0.0]
            reflux_mag = torch.tensor(0.0, device=X.device)

        H_pooled = H_seq.mean(dim=1)
        if self.use_mutual:
            O, gates, share_mask, kl_mod = self.mutual(A, H_pooled)
        else:
            O = A
            gates = torch.zeros(X.size(0), 1, device=X.device)
            kl_mod = torch.zeros(X.size(0), 1, device=X.device)

        logits = self.clf(O)
        return {
            "logits": logits, "kl_loss": kl, "kl_w": kl_w, "kl_mod": kl_mod,
            "Z_star": Z, "Z_init": Z_init, "O": O, "gates": gates,
            "convergence": convergence, "reflux_mag": reflux_mag,
            "feats": feats, "plv": plv,
        }

    def compute_loss(self, out, labels, Z_next=None, Z_next2=None):
        loss = F.cross_entropy(out["logits"], labels, label_smoothing=0.1)

        if self.use_water:
            warmup = min(1.0, (self._current_epoch + 1) / max(KL_WARMUP_EPOCHS, 1))
            loss = loss + KL_W * warmup * out["kl_loss"]

        if self.use_mutual:
            gates = out["gates"]
            loss = loss + ORTHO_W * self.mutual.ortho_loss()
            loss = loss + MUTUAL_W * self.mutual.mutual_loss()
            loss = loss + SPEC_W * self.mutual.specialization_loss(gates)
            loss = loss + GATE_ENTROPY_W * self.mutual.gate_entropy_loss(gates)

        if self.use_pred and self.pred_head is not None:
            Z_t, O_t = out["Z_init"], out["O"]
            pred_out = self.pred_head(Z_t, O_t, Z_next, Z_next2)
            if Z_next is not None:
                pred_loss = F.mse_loss(pred_out["Z_pred_t1"][:Z_next.size(0)],
                                       Z_next.detach())
                if Z_next2 is not None and "Z_pred_t2" in pred_out:
                    pred_loss = pred_loss + 0.5 * F.mse_loss(
                        pred_out["Z_pred_t2"][:Z_next2.size(0)], Z_next2.detach())
                if "delta_Z" in pred_out:
                    pred_loss = pred_loss + 0.001 * pred_out["delta_Z"].norm() \
                                / max(Z_next.size(0), 1)
            else:
                pred_loss = F.mse_loss(pred_out["Z_pred_t1"], out["Z_star"].detach())
            loss = loss + PRED_W * pred_loss

        if self.use_reflux and isinstance(out["reflux_mag"], torch.Tensor):
            if out["reflux_mag"].numel() == 1:
                loss = loss + REFLUX_W * F.relu(0.01 - out["reflux_mag"])
        return loss

# =========================================================================
# 4. Baselines — raw input (B, 62, 800), 5-year SOTA
# =========================================================================
class EEGNet_Raw(nn.Module):
    """EEGNet (Lawhern 2018) — classic BCI baseline, raw variant."""
    def __init__(self, C=62, T=WINDOW_SAMPLES, nc=N_CLASSES, F1=16, D=2, F2=32):
        super().__init__()
        pool1 = 8; pool2 = 16
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 129), padding=(0, 64)), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, pool1)), nn.Dropout(0.25))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 33), padding=(0, 16), groups=F1 * D),
            nn.Conv2d(F1 * D, F2, (1, 1)), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, pool2)), nn.Dropout(0.25))
        with torch.no_grad():
            self.fd = self.block2(self.block1(torch.zeros(1, 1, C, T))).numel()
        self.clf = nn.Linear(self.fd, nc)

    def forward(self, X):
        x = X.unsqueeze(1)                                   # (B,1,62,800)
        return self.clf(self.block2(self.block1(x)).flatten(1))

class TSception_Raw(nn.Module):
    """TSception (Ding 2021, IEEE TAFFC) — multi-scale temporal + spatial attn."""
    def __init__(self, C=62, nc=N_CLASSES):
        super().__init__()
        self.tconv1 = nn.Conv2d(1, 16, (1, 129), padding=(0, 64))
        self.tconv2 = nn.Conv2d(1, 16, (1, 65), padding=(0, 32))
        self.tconv3 = nn.Conv2d(1, 16, (1, 33), padding=(0, 16))
        self.pool = nn.AvgPool2d((1, 4))          # 800→200, 4× speedup
        self.sconv = nn.Conv2d(48, 48, (C, 1))
        self.bn = nn.BatchNorm2d(48)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(48, 12, 1),
            nn.ReLU(), nn.Conv2d(12, 48, 1), nn.Sigmoid())
        self.clf = nn.Linear(48, nc)

    def forward(self, X):
        x = X.unsqueeze(1)
        h = torch.cat([self.tconv1(x), self.tconv2(x), self.tconv3(x)], dim=1)
        h = self.pool(h)
        h = F.elu(self.bn(self.sconv(h)))
        return self.clf((h * self.attn(h)).mean(dim=[2, 3]))

class DGCNN_Raw(nn.Module):
    """DGCNN (Song 2018) — dynamic graph CNN, node-level graph baseline."""
    def __init__(self, C=62, nc=N_CLASSES):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 33), padding=(0, 16)), nn.BatchNorm2d(32), nn.ELU())
        self.gconv1 = nn.Sequential(nn.Conv1d(C * 32, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.gconv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(128, nc)

    def forward(self, X):
        x = self.tconv(X.unsqueeze(1))                      # (B,32,C,T')
        B, F, C_out, T_out = x.shape
        x = self.gconv2(self.gconv1(x.reshape(B, F * C_out, T_out)))
        return self.clf(self.pool(x).squeeze(-1))

class EEGConformer_Raw(nn.Module):
    """EEGConformer (Song 2022, IEEE TNSRE) — CNN+Transformer SOTA hybrid."""
    def __init__(self, C=62, nc=N_CLASSES, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model // 2, (1, 129), padding=(0, 64))
        self.conv2 = nn.Conv2d(1, d_model // 2, (1, 33), padding=(0, 16))
        self.bn = nn.BatchNorm2d(d_model)
        self.spatial_conv = nn.Conv2d(d_model, d_model, (C, 1))
        self.bn2 = nn.BatchNorm2d(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.clf = nn.Linear(d_model, nc)

    def forward(self, X):
        x = X.unsqueeze(1)
        h = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        h = self.bn2(self.spatial_conv(F.gelu(self.bn(h)))).squeeze(2)  # (B,d,T')
        h = F.adaptive_avg_pool1d(h, 100)                              # 800→100 tokens
        h = h.transpose(1, 2)
        T_out = h.size(1)
        # Sinusoidal positional encoding (T-robust)
        pos = torch.arange(T_out, device=X.device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, h.size(-1), 2, device=X.device).float()
                        * (-math.log(10000.0) / h.size(-1)))
        pe = torch.zeros(1, T_out, h.size(-1), device=X.device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        h = self.transformer(h + pe)
        return self.clf(h.mean(dim=1))

class LMDA_Net(nn.Module):
    """LMDA-Net (Miao 2022, NeurIPS) — lightweight EEG SOTA."""
    def __init__(self, C=62, nc=N_CLASSES, D=24, k=24):
        super().__init__()
        self.conv1 = nn.Conv2d(1, D, (1, 51), padding=(0, 25))
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, (C, 1), groups=D)       # spatial depthwise
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, k, (1, 1))                 # channel attention
        self.bn3 = nn.BatchNorm2d(k)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.clf = nn.Linear(k * 8, nc)

    def forward(self, X):
        x = self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(X.unsqueeze(1))))))
        x = F.gelu(self.bn3(self.conv3(x)))
        return self.clf(self.pool(x).flatten(1))

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

class DANN_Raw(nn.Module):
    """DANN (Ganin 2016) — subject domain adversarial (transfer baseline)."""
    def __init__(self, C=62, nc=N_CLASSES, n_subjects=15):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (1, 33), padding=(0, 16)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, (1, 33), padding=(0, 16)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8)))
        with torch.no_grad():
            self.fd = self.feature(torch.zeros(1, 1, C, WINDOW_SAMPLES)).numel()
        self.task_clf = nn.Linear(self.fd, nc)
        self.domain_clf = nn.Sequential(
            nn.Linear(self.fd, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_subjects))

    def forward(self, X, subject_ids=None, lam=1.0):
        f = self.feature(X.unsqueeze(1)).flatten(1)
        out = {"logits": self.task_clf(f)}
        if subject_ids is not None:
            out["domain_logits"] = self.domain_clf(GradReverse.apply(f, lam))
            out["domain_labels"] = subject_ids
        return out

class DeepCORAL_Raw(nn.Module):
    """DeepCORAL (Sun 2016) — second-order alignment (transfer baseline)."""
    def __init__(self, C=62, nc=N_CLASSES):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (1, 33), padding=(0, 16)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, (1, 33), padding=(0, 16)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8)))
        with torch.no_grad():
            self.fd = self.feature(torch.zeros(1, 1, C, WINDOW_SAMPLES)).numel()
        self.clf = nn.Linear(self.fd, nc)

    def forward(self, X):
        f = self.feature(X.unsqueeze(1)).flatten(1)
        return {"logits": self.clf(f), "features": f}

    @staticmethod
    def coral_loss(xs, xt):
        d = xs.size(1)
        cs = (xs.T @ xs) / (xs.size(0) - 1)
        ct = (xt.T @ xt) / (xt.size(0) - 1)
        return (cs - ct).pow(2).sum() / (4 * d * d)

# =========================================================================
# 5. Data loader — raw windows + prediction pairs (w_k ↔ w_{k+2}, 4s apart)
# =========================================================================
def load_raw_with_pairs(n_subjects=None):
    """Load raw SEED-IV: 800→200Hz, 4s windows (2s hop), per-subject z-score.
    Tracks (subj, session, trial, window_idx) so prediction pairs w_k↔w_{k+2}
    (4s apart, zero overlap) stay within one trial. Returns X, y, subj, pair_idx.
    """
    subjects = sorted({int(f.split('_')[0])
                       for s in [1, 2, 3]
                       for f in os.listdir(os.path.join(RAW_DIR, str(s)))
                       if f.endswith('.mat')})
    if n_subjects:
        subjects = subjects[:n_subjects]

    all_X, all_y, all_subj, all_wid = [], [], [], []
    t0 = time.time()

    for subj_id in subjects:
        trials, labels, wids = [], [], []
        for session in [1, 2, 3]:
            sess_dir = os.path.join(RAW_DIR, str(session))
            fname = next((f for f in os.listdir(sess_dir)
                          if f.startswith(f"{subj_id}_") and f.endswith('.mat')), None)
            if fname is None:
                continue
            data = loadmat(os.path.join(sess_dir, fname))
            sess_labels = SESSION_LABELS[session]
            for t in range(24):
                var = next((k for k in data
                            if not k.startswith('__') and k.endswith(f'_eeg{t + 1}')), None)
                if var is None:
                    continue
                raw = np.nan_to_num(data[var]).astype(np.float32)
                ds = decimate(raw, 4, axis=1)
                trials.append(ds); labels.append(sess_labels[t])
                wids.append(t)  # trial id as window key

        if not trials:
            continue
        cat = np.concatenate(trials, axis=1)
        mu = cat.mean(axis=1, keepdims=True)
        sd = cat.std(axis=1, keepdims=True) + 1e-8
        del cat

        for ds, lab, wid in zip(trials, labels, wids):
            ds = (ds - mu) / sd
            T = ds.shape[1]
            k = 0
            for st in range(0, T - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
                all_X.append(torch.from_numpy(ds[:, st:st + WINDOW_SAMPLES].copy()))
                all_y.append(lab); all_subj.append(subj_id - 1)
                all_wid.append((subj_id - 1, wid, k))
                k += 1
        print(f"  [Data] subject {subj_id}: {len(all_X)} windows ({time.time() - t0:.0f}s)",
              flush=True)

    X = torch.stack(all_X)
    y = torch.tensor(all_y, dtype=torch.long)
    subj = torch.tensor(all_subj, dtype=torch.long)

    # Prediction pairs: (subj, trial, k) → (subj, trial, k+2) within trial
    pos = {}
    for i, key in enumerate(all_wid):
        pos.setdefault(key[:2], {})[key[2]] = i
    pair_a, pair_b = [], []
    for (s, tr), win_map in pos.items():
        for k, idx in sorted(win_map.items()):
            if k + PRED_GAP in win_map:
                pair_a.append(idx); pair_b.append(win_map[k + PRED_GAP])
    pair_idx = torch.stack([torch.tensor(pair_a), torch.tensor(pair_b)], dim=1)

    print(f"[Data] RAW SEED-IV: {X.shape[0]} windows | {len(subjects)} subjects | "
          f"{WINDOW_SAMPLES}samples({WINDOW_SAMPLES // FS_TARGET}s) | "
          f"classes={dict(Counter(all_y))} | {len(pair_idx)} prediction pairs | "
          f"prep {time.time() - t0:.0f}s", flush=True)
    return X, y, subj, pair_idx

# =========================================================================
# 6. Training loops
# =========================================================================
def train_epoch_dame(model, X_train, y_train, pair_map, opt, bs=BATCH_SIZE):
    """DAME training with activated pre-migration, TWO passes (no data confound):
    Pass 1: ALL windows → CE + KL + mutual losses (pred head runs self-consistency
            fallback: predict own converged Z_star from Z_init).
    Pass 2: paired windows only → genuine temporal prediction Z(t)→Ẑ(t+4s).
    pair_map: local window index → local index of its +4s pair (train set only)."""
    model.train()
    total_loss, n_b = 0.0, 0

    # Pass 1: all windows (same data as non-pred variants)
    n = X_train.size(0)
    idx = torch.randperm(n)
    for i in range(0, n, bs):
        bidx = idx[i:i + bs]
        xb = X_train[bidx].to(DEVICE)
        yb = y_train[bidx].to(DEVICE)
        out = model(xb)
        loss = model.compute_loss(out, yb, None, None)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n_b += 1

    # Pass 2: prediction pairs (temporal evolution of coupling dynamics)
    if model.use_pred and pair_map:
        a_list = list(pair_map.keys())
        perm = torch.randperm(len(a_list))
        for i in range(0, len(a_list), bs):
            ba = [a_list[j] for j in perm[i:i + bs].tolist()]
            xb = X_train[ba].to(DEVICE)
            yb = y_train[ba].to(DEVICE)
            out = model(xb)
            bb = [pair_map[a] for a in ba]
            with torch.no_grad():
                out_b = model(X_train[bb].to(DEVICE))
            Z_next = out_b["Z_star"].detach()      # (bs, K) — aligned with batch
            loss = model.compute_loss(out, yb, Z_next, Z_next)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_b += 1
    return total_loss / max(n_b, 1)

def train_epoch_plain(model, X_train, y_train, opt, bs=BATCH_SIZE):
    model.train()
    n = X_train.size(0)
    idx = torch.randperm(n)
    total_loss, n_b = 0.0, 0
    for i in range(0, n, bs):
        bidx = idx[i:i + bs]
        xb = X_train[bidx].to(DEVICE)
        yb = y_train[bidx].to(DEVICE)
        loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n_b += 1
    return total_loss / max(n_b, 1)

def train_epoch_dann(model, X_train, y_train, subj_train, opt, bs=BATCH_SIZE):
    model.train()
    n = X_train.size(0)
    idx = torch.randperm(n)
    total_loss, n_b = 0.0, 0
    for i in range(0, n, bs):
        bidx = idx[i:i + bs]
        xb = X_train[bidx].to(DEVICE)
        yb = y_train[bidx].to(DEVICE)
        db = subj_train[bidx].to(DEVICE)
        lam = min(1.0, 2.0 * model._lam_epoch / max(EPOCHS - 1, 1))
        out = model(xb, subject_ids=db, lam=lam)
        loss = F.cross_entropy(out["logits"], yb, label_smoothing=0.1) \
            + 0.1 * F.cross_entropy(out["domain_logits"], out["domain_labels"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n_b += 1
    return total_loss / max(n_b, 1)

def train_epoch_coral(model, X_train, y_train, X_target, opt, bs=BATCH_SIZE):
    """DeepCORAL: aligns training subjects with UNLABELED target subject (UDA)."""
    model.train()
    n = X_train.size(0)
    idx = torch.randperm(n)
    total_loss, n_b = 0.0, 0
    for i in range(0, n, bs):
        bidx = idx[i:i + bs]
        xb = X_train[bidx].to(DEVICE)
        yb = y_train[bidx].to(DEVICE)
        ti = torch.randint(0, X_target.size(0), (bs,))
        xt = X_target[ti].to(DEVICE)
        out_s = model(xb)
        out_t = model(xt)
        loss = F.cross_entropy(out_s["logits"], yb, label_smoothing=0.1) \
            + 0.5 * DeepCORAL_Raw.coral_loss(out_s["features"], out_t["features"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n_b += 1
    return total_loss / max(n_b, 1)

@torch.no_grad()
def evaluate(model, X_test, y_test, bs=256, is_dame=False, is_dann=False):
    model.eval()
    preds = []
    for i in range(0, X_test.size(0), bs):
        xb = X_test[i:i + bs].to(DEVICE)
        if is_dame:
            out = model(xb)["logits"]
        elif is_dann:
            out = model(xb)["logits"]
        else:
            out = model(xb)
        preds.append(out.argmax(1).cpu())
    pred = torch.cat(preds)
    acc = (pred == y_test).float().mean().item()
    return acc, pred

@torch.no_grad()
def collect_diag(model, X_test, y_test, pair_map_test, bs=128):
    """Mechanism-level diagnostics on the HELD-OUT subject (机制级证据):
    1. 每类情绪的平均PLV耦合矩阵 (脑区联动耦合随情绪重组 — DMN可解释性)
    2. 策略指纹: 平均门控向量 + 社区分配 (专家涌现/策略不动如山)
    3. 预测功能: 自洽预测 + 时间演化预测 (vs 平凡基线 MSE(Z_init, Z_star))
       + 两步纠错是否优于一步
    Returns dict of numpy arrays/scalars.
    """
    model.eval()
    P = len(model.coupling.pair_i)
    plv_sums = torch.zeros(N_CLASSES, N_BANDS, P, T_COUP, device=DEVICE)
    plv_cnt = torch.zeros(N_CLASSES, device=DEVICE)
    gates_sum = torch.zeros(N_NEURONS, device=DEVICE)
    self_mse_list = []
    gate_corr_diag = []

    for i in range(0, X_test.size(0), bs):
        xb = X_test[i:i + bs].to(DEVICE)
        out = model(xb)
        yb = y_test[i:i + bs]
        for c in range(N_CLASSES):
            m = yb == c
            if m.any():
                plv_sums[c] += out["plv"][m].sum(0)
                plv_cnt[c] += m.sum()
        if model.use_mutual:
            gates_sum += out["gates"].sum(0)
        if model.use_pred and model.pred_head is not None:
            po = model.pred_head(out["Z_init"], out["O"])
            self_mse_list.append(
                F.mse_loss(po["Z_pred_t1"], out["Z_star"], reduction='none')
                .mean(-1).cpu())

    plv_per_class = (plv_sums / plv_cnt.clamp(min=1).view(-1, 1, 1, 1)).cpu().numpy()
    gates_mean = (gates_sum / max(X_test.size(0), 1)).cpu().numpy()
    self_mse = float(torch.cat(self_mse_list).mean()) if self_mse_list else float('nan')

    diag = {
        "plv_per_class": plv_per_class,        # (4, F, P, Tc)
        "gates_mean": gates_mean,              # (N,)
        "self_pred_mse": self_mse,
        "community_ids": model.mutual.community_ids.cpu().numpy() if model.use_mutual else None,
        "W_mutual_norm": model.mutual.W_mutual.data.norm().item() if model.use_mutual else None,
    }

    # 预测功能 — 时间演化 (跨被试: 在被试外样本上预测4s后的耦合本征)
    if model.use_pred and model.pred_head is not None and pair_map_test:
        mse1, mse2, cos1, trivial = [], [], [], []
        a_list = list(pair_map_test.keys())
        for i in range(0, len(a_list), bs):
            ba = a_list[i:i + bs]
            out_a = model(X_test[ba].to(DEVICE))
            bb = [pair_map_test[a] for a in ba]
            out_b = model(X_test[bb].to(DEVICE))
            Zb = out_b["Z_star"]
            po = model.pred_head(out_a["Z_init"], out_a["O"], Zb, Zb)
            mse1.append(F.mse_loss(po["Z_pred_t1"], Zb, reduction='none').mean(-1).cpu())
            mse2.append(F.mse_loss(po["Z_pred_t2"], Zb, reduction='none').mean(-1).cpu())
            cos1.append(F.cosine_similarity(po["Z_pred_t1"], Zb, dim=-1).cpu())
            trivial.append(F.mse_loss(out_a["Z_init"], Zb, reduction='none').mean(-1).cpu())
        diag.update({
            "pred_temporal_mse1": float(torch.cat(mse1).mean()),
            "pred_temporal_mse2": float(torch.cat(mse2).mean()),
            "pred_temporal_cos": float(torch.cat(cos1).mean()),
            "trivial_mse": float(torch.cat(trivial).mean()),  # 不预测的基线误差
        })
    return diag

# =========================================================================
# 7. LOSO cross-subject protocol with per-fold checkpoints
# =========================================================================
def loso_v4(model_factory, X, y, subj, pair_idx, seeds, epochs=EPOCHS,
            kind="dame", done_folds=None, save_fn=None, tag=""):
    """Strict leave-one-subject-out. kind: dame | plain | dann | coral."""
    Ns = int(subj.max().item()) + 1
    per_seed = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        for s in range(Ns):
            key = f"{tag}_{kind}_s{s}_seed{seed}"
            if done_folds and key in done_folds:
                print(f"  [skip] {key} (done)")
                continue
            tm = subj == s; trm = ~tm
            X_train, y_train = X[trm], y[trm]
            X_test, y_test = X[tm], y[tm]
            subj_train = subj[trm]

            # Global → train-local index map for prediction pairs (within train only)
            train_glob = torch.nonzero(trm).flatten().tolist()
            g2l = {g: li for li, g in enumerate(train_glob)}
            pair_map = {g2l[a]: g2l[b] for a, b in pair_idx.tolist()
                        if a in g2l and b in g2l}

            model = model_factory().to(DEVICE)
            if kind == "dann":
                model._lam_epoch = 0
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

            t0 = time.time()
            for ep in range(epochs):
                if kind == "dame":
                    model.set_epoch(ep, epochs)
                    loss = train_epoch_dame(model, X_train, y_train, pair_map, opt)
                elif kind == "plain":
                    loss = train_epoch_plain(model, X_train, y_train, opt)
                elif kind == "dann":
                    model._lam_epoch = ep
                    loss = train_epoch_dann(model, X_train, y_train, subj_train, opt)
                else:  # coral
                    loss = train_epoch_coral(model, X_train, y_train, X_test, opt)
                sched.step()
                if ep % 5 == 4 or ep == epochs - 1:
                    acc, _ = evaluate(model, X_test, y_test, is_dame=(kind != "plain"),
                                      is_dann=(kind == "dann"))
                    print(f"    [{s + 1}/{Ns}] ep{ep + 1}: loss={loss:.3f} "
                          f"val_acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)

            acc, pred = evaluate(model, X_test, y_test, is_dame=(kind != "plain"),
                                 is_dann=(kind == "dann"))
            per_seed[key] = acc
            per_class = {c: (pred == c).float().mean().item() for c in range(N_CLASSES)}
            print(f"  [{s + 1}/{Ns}] seed{seed} acc={acc:.4f} "
                  f"per_class={per_class} ({time.time() - t0:.0f}s)", flush=True)

            # 机制级诊断 (仅DAME家族): PLV耦合模式/策略指纹/预测质量
            if kind == "dame" and isinstance(model, DAME_Region):
                test_glob = torch.nonzero(tm).flatten().tolist()
                g2l_t = {g: li for li, g in enumerate(test_glob)}
                pair_map_test = {g2l_t[a]: g2l_t[b] for a, b in pair_idx.tolist()
                                 if a in g2l_t and b in g2l_t}
                diag = collect_diag(model, X_test, y_test, pair_map_test)
                ddir = os.path.join(RESULTS_DIR, "diag")
                os.makedirs(ddir, exist_ok=True)
                np.savez(os.path.join(ddir, f"{key}.npz"), **diag)
                active = int((diag["gates_mean"] > 0.5).sum()) if model.use_mutual else 0
                pstr = (f"selfMSE={diag['self_pred_mse']:.3f} "
                        f"predMSE={diag.get('pred_temporal_mse1', float('nan')):.3f} "
                        f"trivMSE={diag.get('trivial_mse', float('nan')):.3f} "
                        f"predCos={diag.get('pred_temporal_cos', float('nan')):.3f} "
                        f"active={active}/{N_NEURONS}")
                print(f"    [diag] {pstr}", flush=True)
            if save_fn:
                save_fn(key, acc)
            del model, opt
            torch.cuda.empty_cache()
    return per_seed

# =========================================================================
# 8. Model registry
# =========================================================================
def make_variant(**kw):
    def _f():
        return DAME_Region(**kw)
    return _f

MODEL_SPECS = {
    # 脑区耦合DAME家族（全激活）— OURS
    "DAME-Region":    dict(fn=make_variant(), kind="dame"),                       # 全架构
    "DAME-Node":      dict(fn=make_variant(groups=[[i] for i in range(62)]),
                           kind="dame"),                                          # A1: 单节点
    "DAME-NoCoupling": dict(fn=make_variant(use_coupling=False), kind="dame"),   # A2: 无联动耦合
    "DAME-NoPred":    dict(fn=make_variant(use_pred=False), kind="dame"),        # A3: 无预测
    "DAME-NoReflux":  dict(fn=make_variant(use_reflux=False), kind="dame"),      # 机制消融
    "DAME-NoMutual":  dict(fn=make_variant(use_mutual=False), kind="dame"),
    "DAME-NoWater":   dict(fn=make_variant(use_water=False), kind="dame"),
    "DAME-Base":      dict(fn=make_variant(use_water=False, use_mutual=False,
                                           use_pred=False, use_reflux=False),
                           kind="dame"),
    # 近5年SOTA基线
    "EEGNet":         dict(fn=lambda: EEGNet_Raw(), kind="plain"),
    "TSception":      dict(fn=lambda: TSception_Raw(), kind="plain"),
    "DGCNN":          dict(fn=lambda: DGCNN_Raw(), kind="plain"),
    "EEGConformer":   dict(fn=lambda: EEGConformer_Raw(), kind="plain"),
    "LMDA-Net":       dict(fn=lambda: LMDA_Net(), kind="plain"),
    # 迁移学习基线
    "DANN":           dict(fn=lambda: DANN_Raw(n_subjects=15), kind="dann"),
    "DeepCORAL":      dict(fn=lambda: DeepCORAL_Raw(), kind="coral"),
}

# =========================================================================
# 9. Main — resume-capable multi-seed LOSO
# =========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--subjects", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model names")
    ap.add_argument("--tag", type=str, default="v4")
    args = ap.parse_args()

    if args.quick:
        args.subjects, args.seeds, args.epochs = 3, 1, 3
        print("[QUICK MODE] 3 subjects, 1 seed, 3 epochs")

    seeds = ALL_SEEDS[:args.seeds]
    models = args.models.split(",") if args.models else list(MODEL_SPECS.keys())
    out_path = os.path.join(RESULTS_DIR, f"eeg_{args.tag}_results.json")
    done = {}
    if os.path.exists(out_path):
        done = json.load(open(out_path))

    X, y, subj, pair_idx = load_raw_with_pairs(n_subjects=args.subjects)

    def save_fn(key, acc):
        done[key] = acc
        json.dump(done, open(out_path, "w"), indent=2)

    all_results = {}
    for mname in models:
        spec = MODEL_SPECS[mname]
        print(f"\n{'=' * 60}\nMODEL: {mname} ({spec['kind']})\n{'=' * 60}", flush=True)
        per_seed = loso_v4(spec["fn"], X, y, subj, pair_idx, seeds,
                           epochs=args.epochs, kind=spec["kind"],
                           done_folds=done, save_fn=save_fn, tag=mname)
        all_results[mname] = per_seed
        save_fn(f"SUMMARY_{mname}", per_seed)

    # Final report
    print(f"\n{'=' * 70}\nV4 FINAL — 跨被试LOSO (原始波形)\n{'=' * 70}")
    print(f"{'Model':<18} {'Acc±std':<14} {'min':<7} {'seeds'}")
    print("-" * 70)
    for mname in models:
        fold_accs = [v for k, v in done.items()
                     if k.startswith(f"{mname}_dame_s") or k.startswith(f"{mname}_plain_s")
                     or k.startswith(f"{mname}_dann_s") or k.startswith(f"{mname}_coral_s")]
        if not fold_accs:
            continue
        mu, sd = np.mean(fold_accs), np.std(fold_accs, ddof=1) if len(fold_accs) > 1 else 0.0
        print(f"{mname:<18} {mu * 100:>6.2f}±{sd * 100:.2f}  {min(fold_accs) * 100:>6.1f}  "
              f"{len(fold_accs)}")

if __name__ == "__main__":
    main()
