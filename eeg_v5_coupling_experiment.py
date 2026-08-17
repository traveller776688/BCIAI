#!/usr/bin/env python3
"""
EEG V5 — DAME-Coupling: 机制深度绑定脑区联动耦合 (全局更新, v4试错全部落地)
==================================================================================
v4 教训（本文件逐一修复）:
  L1. 机制浮在特征下游 → 消融中性 → 顶刊线差 +2~4 点。
      → 修复: 水循环直接蒸馏耦合token; 互助社会N=12脑区专家 + W_mutual用测得的
        PLV邻接初始化 + 门控由耦合状态调制; 预测头预测耦合本征演化(不是Z向量)。
  L2. 预测器与平凡基线打平(predMSE≈trivMSE) → 预测功能不可证伪。
      → 修复: 目标=固定随机正交投影的PLV耦合本征(投影器frozen, 无作弊路径),
        诊断直接对比 predMSE vs trivMSE(不预测的基线) + selfMSE(Z是否编码耦合)。
        (iter4 更进一步: 预测目标换成跨trial情绪稳定性二分类, 机会基线20.3%;
         frozen投影 W_proj 保留为 Pass1 耦合自洽约束的目标)
  L3. 机制必须激活而非调静音。→ 全激活权重保留(PRED_W/REFLUX_W=0.01)。
  L4. 预测对数据量混淆。→ 两遍训练保留: Pass1全窗口(自洽目标), Pass2配对窗口(真实4s后)。
  L5. Base(耦合+线性分类器)超过完整DAME → 机制必须做加法而非拖累。
      → 分类器 = [O(D) ⊕ 功率残差(64)] 双重表示; 每机制消融单独可测。
  L6. 迁移=被试之间, 单数据集LOSO, 原始波形(相位信息是耦合的前提)。→ 协议不变。

五哲学与耦合前提的对应(论文论证结构):
  P1 双重表示 = 耦合token(时序) ⊕ 本征Z(全局) ⊕ 功率残差(活动)
  P2 水循环    = 从脑区耦合序列中蒸馏情绪本征(蒸发VIB→降雨CrossAttn→回流Banach)
  P3 互助社会  = 12脑区专家, 社区=功能网络涌现, 互助强度=测得PLV邻接
  P4 预迁移    = 预测跨trial情绪稳定性(保持vs翻转, 可证伪目标; iter4替换旧耦合数值预测)
  P5 非线性信息瓶颈 = K=32瓶颈必须保留耦合结构(由自洽预测强制)

协议: 跨被试LOSO (15折×3种子), 单数据集, 原始波形62ch
消融: NoCoupling / NoPred / NoMutual / NoWater / NoReflux / Base
基线: EEGNet, TSception, DGCNN, EEGConformer, LMDA-Net, DANN, DeepCORAL
迭代模式: --fast N (前N被试) 小规模快速消融, 每轮落盘 results/v5_iterations.jsonl
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

SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
D_MODEL = 256
K_LATENT = 32
D_MEM = 32
N_CLASSES = 4
TEMP_INIT = 0.8
TEMP_FINAL = 3.5        # iter2: 2.5时门控温吞(active=0/12), 加温逼出真专家
TEMP_ANNEAL_EPOCHS = 8

# =========================================================================
# CONFIG — mechanisms FULLY ACTIVE
# =========================================================================
FS_TARGET = 200
WINDOW_SAMPLES = 4 * FS_TARGET        # 800 = 4s
WINDOW_STRIDE = 2 * FS_TARGET         # 400 = 2s hop
SUB_WIN = FS_TARGET // 2              # 100 = 0.5s coupling resolution
T_COUP = WINDOW_SAMPLES // SUB_WIN    # 8 coupling steps per window
PRED_GAP = 2                          # predict w_{k+2} (4s later, zero overlap)

ORTHO_W = 1e-3
MUTUAL_W = 1e-3
SPEC_W = 1e-3
GATE_ENTROPY_W = 1e-3
REFLUX_W = 0.01
PRED_W = 0.005          # Pass1 耦合自洽约束 (Z保留耦合结构, P5)
STAB_W = 0.05           # Pass2 跨trial情绪稳定性预测 (iter4: 替换耦合数值预测)
KL_W = 0.008

LR = 2e-4
EPOCHS = 15
BATCH_SIZE = 64
KL_WARMUP_EPOCHS = 8
ALL_SEEDS = [42, 123, 789]
D_POW = 64                            # 功率残差维度 (双重表示: 活动支路)
D_COUP_ESS = 64                       # 耦合本征维度 (固定投影目标)

# =========================================================================
# 1. 脑区定义 — 62ch extended 10-20 → 12 cortical regions
# =========================================================================
CH_ORDER = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
            'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

REGIONS = {
    'PFC': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4'],                          # DMN前枢纽(mPFC)
    'FL':  ['F7', 'F5', 'F3', 'F1', 'FZ'],
    'FR':  ['F2', 'F4', 'F6', 'F8'],
    'FC':  ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    'C':   ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'],
    'TL':  ['T7', 'TP7'],
    'TR':  ['T8', 'TP8'],
    'CP':  ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],            # DMN后枢纽(PCC)
    'P':   ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    'PO':  ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'],
    'O':   ['O1', 'OZ', 'O2'],
    'CB':  ['CB1', 'CB2'],
}
REGION_NAMES = list(REGIONS.keys())
REGION_GROUPS = [[CH_ORDER.index(ch) for ch in REGIONS[name]] for name in REGION_NAMES]
N_REGIONS = len(REGION_GROUPS)
assert sum(len(g) for g in REGION_GROUPS) == 62

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 45)}
BAND_NAMES = list(BANDS.keys())
N_BANDS = len(BANDS)

def make_pairs(R):
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    return [p[0] for p in pairs], [p[1] for p in pairs]

# =========================================================================
# 2. 信号处理 — FFT带通 + Hilbert相位 + 脑区PLV耦合 (全可微)
# =========================================================================
def bandpass_fft(x, lo, hi, fs=FS_TARGET):
    f = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / fs).to(x.device)
    mask = ((freqs >= lo) & (freqs < hi)).float()
    return torch.fft.irfft(f * mask.view(*([1] * (x.dim() - 1)), -1), n=x.size(-1), dim=-1)

def hilbert_analytic(x):
    f = torch.fft.rfft(x, dim=-1)
    mult = torch.ones(f.size(-1), device=x.device)
    mult[1:-1] = 2.0
    return torch.fft.irfft(f * mult, n=x.size(-1), dim=-1)

def plv_coupling(reg, pair_i, pair_j, chunk=64):
    """Region-level PLV over 0.5s sub-windows.
    reg: (B, R, T) → plv (B, F, P, Tc), power (B, F, R, Tc)"""
    B, R, T = reg.shape
    P = len(pair_i)
    T_c = T // SUB_WIN
    plv_all, pow_all = [], []
    for (lo, hi) in BANDS.values():
        xb = bandpass_fft(reg, lo, hi)
        pw = (xb ** 2).reshape(B, R, T_c, SUB_WIN).mean(-1)
        pow_all.append(pw)
        ph = torch.angle(hilbert_analytic(xb))
        plv_pairs = []
        for c in range(0, P, chunk):
            i1 = pair_i[c:c + chunk]; i2 = pair_j[c:c + chunk]
            d = ph[:, i1, :] - ph[:, i2, :]
            cosv = torch.cos(d).reshape(B, -1, T_c, SUB_WIN).mean(-1)
            sinv = torch.sin(d).reshape(B, -1, T_c, SUB_WIN).mean(-1)
            plv_pairs.append(torch.sqrt(cosv ** 2 + sinv ** 2))
        plv_all.append(torch.cat(plv_pairs, dim=1))
    return torch.stack(plv_all, dim=1), torch.stack(pow_all, dim=1)

class RegionCouplingV2(nn.Module):
    """62ch raw → 双支路表示:
      耦合支路 H_coup (B,Tc,D): PLV耦合token — 水循环的蒸馏对象 (P1/P2)
      活动支路 H_pow  (B,Tc,64): 区域功率 — 互助社会门控原始特征 + 分类残差 (P1)
    plv 始终返回 (诊断/预测目标/消融可比性)。"""

    def __init__(self, groups, D=D_MODEL, use_coupling=True):
        super().__init__()
        self.groups = groups
        self.R = len(groups)
        self.use_coupling = use_coupling
        self.pair_i, self.pair_j = make_pairs(self.R)
        P = len(self.pair_i)
        d_coup = N_BANDS * P if use_coupling else N_BANDS * self.R
        self.coup_proj = nn.Sequential(
            nn.Linear(d_coup, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D))
        self.pow_proj = nn.Sequential(
            nn.Linear(N_BANDS * self.R, D_POW), nn.LayerNorm(D_POW), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(1, T_COUP, D) * 0.02)

    def forward(self, x):
        """x: (B, 62, T) raw → H_coup (B,Tc,D), H_pow (B,Tc,64), plv (B,F,P,Tc)"""
        B, C, T = x.shape
        reg = torch.stack([x[:, g].mean(1) for g in self.groups], dim=1)  # (B,R,T)
        plv, power = plv_coupling(reg, self.pair_i, self.pair_j)
        coup_feat = plv.permute(0, 3, 1, 2).reshape(B, T_COUP, -1)        # (B,Tc,F*P)
        pow_feat = power.permute(0, 3, 1, 2).reshape(B, T_COUP, -1)       # (B,Tc,F*R)
        if self.use_coupling:
            H_coup = self.coup_proj(coup_feat) + self.pos_embed
        else:
            H_coup = self.coup_proj(pow_feat) + self.pos_embed
        H_pow = self.pow_proj(pow_feat)
        return H_coup, H_pow, plv

@torch.no_grad()
def compute_plv_adjacency(groups, X_sub, bs=128, max_win=256):
    """(R,R) 平均PLV邻接 — 互助社会 W_mutual 的结构初始化 (测得的耦合, 非随机)。"""
    front = RegionCouplingV2(groups).to(DEVICE)
    R = front.R
    adj = torch.zeros(R, R, device=DEVICE)
    pi, pj = front.pair_i, front.pair_j
    n = min(X_sub.size(0), max_win)
    for i in range(0, n, bs):
        xb = X_sub[i:i + bs].to(DEVICE)
        _, _, plv = front(xb)
        plv_mean = plv.mean(dim=(1, 3))                     # (B, P)
        for k in range(len(pi)):
            adj[pi[k], pj[k]] += plv_mean[:, k].sum()
    adj = adj + adj.T
    adj /= adj.max().clamp(min=1e-8)
    del front
    torch.cuda.empty_cache()
    return adj.cpu()

# =========================================================================
# 3. WaterCycleV2 — 从耦合token蒸馏情绪本征 (v3移植, 语义不变)
# =========================================================================
class WaterCycleV2(nn.Module):
    """VIB蒸发 → CrossAttn降雨 → Banach回流. H_seq=(B,T,D) 耦合token序列."""

    def __init__(self, D=D_MODEL, K=K_LATENT, max_iter=5, converge_thresh=0.98,
                 use_reflux=True):
        super().__init__()
        self.use_reflux = use_reflux
        self.mu_proj = nn.Linear(D, K)
        self.logvar_proj = nn.Linear(D, K)
        self.W_Q = nn.Linear(K, K)
        self.W_K = nn.Linear(D, K)
        self.W_V = nn.Linear(D, D)
        self.reflux_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(K, D // 2)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(D // 2, D)),
        )
        self.reflux_scale = nn.Parameter(torch.tensor(0.05))
        self.log_kl_w = nn.Parameter(torch.tensor(-4.83))   # ln(0.008)
        self.max_iter = max_iter
        self.converge_thresh = converge_thresh

    def evaporate(self, R_global):
        mu = self.mu_proj(R_global)
        logvar = torch.clamp(self.logvar_proj(R_global), -10, 10)
        std = torch.exp(0.5 * logvar)
        Z = mu + (std * torch.randn_like(std) if self.training else 0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return Z, mu, logvar, kl

    def precipitate(self, Z, H_seq):
        Q = self.W_Q(Z)
        K = self.W_K(H_seq)
        V = self.W_V(H_seq)
        scale = math.sqrt(K.size(-1))
        attn = F.softmax(Q.unsqueeze(1) @ K.transpose(-2, -1) / scale, dim=-1)
        A = (attn @ V).squeeze(1)
        return A, attn

    def reflux_step(self, Z, R_global):
        g_Z = self.reflux_net(Z) * self.reflux_scale
        combined = R_global + g_Z
        Z_new, _, _, _ = self.evaporate(combined)
        return Z_new

    def forward(self, H_seq):
        R_pooled = H_seq.mean(dim=1)
        Z_init, _, _, kl = self.evaporate(R_pooled)
        A, attn = self.precipitate(Z_init, H_seq)
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

# =========================================================================
# 4. MutualSocietyV3 — 脑区专家社会 (N=R个专家, 一个专家一个脑区)
#    绑定耦合的三处: ①W_mutual用测得PLV邻接初始化 ②门控由耦合状态调制
#    ③社区涌现=功能网络涌现 (DMN等大尺度网络的可解释性出口)
# =========================================================================
FIELD_ROUTER = False   # 场域路由器 (快慢通路失配度 ω) — v5_router 三判决门全部证伪,
                       # 已关闭: ω≡1 (社会全权重, 恢复 v6 融合行为).
                       # 证伪证据: 指纹分离度 区级PLV +0.0034 / 边级PLV +0.0088 /
                       # 功率最近原型 −0.075 / 功率margin +0.02, 全部远低于可用阈值.
                       # 探针: field_fingerprint_probe.py (2026-08-14)

class MutualSocietyV3(nn.Module):
    """v2移植+耦合绑定:
      - N = 脑区数 (专家=脑区代表/策略, 身份不变)
      - W_mutual[i,j] ∝ 测得的PLV邻接 (互助强度=实际耦合强度)
      - 边锚定 (v5_modedge): 专家注视的对象 = 其11条入射耦合边的模式 (PLV×F频段),
        如如不动的锚 = 学习到的 expertise 策略向量 vs 实时边模式的相似度;
        专家状态由入射边更新, GRU记忆跟踪边的跨窗口动态 — 与"核心=联动耦合"基座对齐
      - 门控: σ(α·cos(边模式, ê_i) + b_i + β·coup_i)
      - 3流GRU记忆/社区正则/KL调制/策略指纹: 与v2一致"""

    def __init__(self, N=N_REGIONS, D=D_MODEL, d_mem=D_MEM,
                 share_ratio=0.6, n_communities=4, plv_adj=None,
                 d_pow=D_POW, n_bands=N_BANDS):
        super().__init__()
        self.N = N
        self.d_mem = d_mem
        self.share_ratio = share_ratio
        self.n_communities = n_communities
        r = max(d_mem // 4, 4)

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
        self.gate_bias = nn.Parameter(torch.full((N,), -0.2))  # iter2: 起步少招募, 由学习+温度招募
        self.temp = TEMP_INIT
        # 边锚定输入: 每专家注视其 (N-1)*F 入射边模式 → d_mem 策略空间
        self.edge_proj = nn.Linear((N - 1) * n_bands, d_mem)
        # 耦合状态调制: 每区与全脑的耦合强度 (B, F*R) → 专家招募偏置 (B, N)
        self.coup_gate_proj = nn.Linear(n_bands * N, N)

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

        self.proj_out = nn.Linear(d_mem, D)
        self.register_buffer('mem', torch.zeros(N, d_mem))

        # 场域感知 (快慢通路): 慢参考=训练集耦合强度指纹 EMA, 快读数=当前样本指纹
        self.register_buffer('coupling_ref', torch.zeros(n_bands * N))
        self.register_buffer('ref_ready', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('_ref_n', torch.zeros(1))
        self._dist_log = []                      # 训练期失配度日志 (标定 ω 阈值, 无泄漏)
        self.omega_d0 = 0.5                      # 标定前默认
        self.omega_a = 10.0                      # 软路由锐度
        self.field_router = FIELD_ROUTER         # 证伪后默认关闭 (ω≡1)

    def _coup_strength(self, plv):
        """(B,F,P,Tc) → 每区与全脑耦合强度 (B, F*R). 只依赖结构索引, 无参数."""
        B = plv.size(0)
        plv_mean = plv.mean(-1)                                  # (B, F, P)
        out = plv_mean.new_zeros(B, plv_mean.size(1), self.N)
        out = out.index_add_(-1, self._pair_i, plv_mean)
        out = out.index_add_(-1, self._pair_j, plv_mean)
        return out.reshape(B, -1)                                # (B, F*R)

    def _bind_pair_indices(self, pair_i, pair_j):
        self.register_buffer('_pair_i', torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer('_pair_j', torch.tensor(pair_j, dtype=torch.long))
        # 边锚定: 每区的入射边索引 (N, N-1), 按邻居编号排序 — 专家注视的对象=耦合边
        pi = torch.tensor(pair_i, dtype=torch.long)
        pj = torch.tensor(pair_j, dtype=torch.long)
        inc = []
        for r in range(self.N):
            m = (pi == r) | (pj == r)
            nbr = torch.where(pi == r, pj, pi)[m]
            order = torch.argsort(nbr)
            inc.append(torch.nonzero(m, as_tuple=False).flatten()[order])
        self.register_buffer('_inc_idx', torch.stack(inc))          # (N, N-1)

    def forward(self, plv):
        """plv: (B, F, P, Tc) → O (B, D), gates (B, N)"""
        B = plv.size(0)
        # 边锚定: 每专家注视其 11 条入射耦合边 (F 频段模式) — 如如不动的对象=耦合边
        plv_mean = plv.mean(-1)                                   # (B, F, P)
        edges_b = plv_mean.transpose(1, 2)                        # (B, P, F)
        E_view = edges_b[:, self._inc_idx, :].reshape(B, self.N, -1)  # (B, N, (N-1)*F)
        E_proj = self.edge_proj(E_view)                           # (B, N, d_mem)

        e_n = F.normalize(self.expertise, dim=-1)
        e_v = F.normalize(E_proj, dim=-1)
        cos_sim = torch.einsum('bnd,nd->bn', e_v, e_n)            # (B, N)
        coup = self._coup_strength(plv)                    # (B, F*R) 快读数
        coup_bias = self.coup_gate_proj(coup)
        # 场域亲和度 ω = σ(a·(d0 − d)): d = 快读数 vs 慢参考的余弦失配
        # 陌生场域(跨被试) d 大 → ω→0 社会沉默; 熟悉场域(跨会话) d 小 → ω→1
        # [v5_router 证伪后默认关闭] FIELD_ROUTER=False 时 ω≡1, 社会全权重
        if self.field_router:
            if self.training:
                with torch.no_grad():
                    self.coupling_ref.mul_(0.99).add_(coup.mean(0) * 0.01)
                    self._ref_n += 1
                    if self._ref_n.item() >= 20:
                        self.ref_ready.fill_(True)
                        d_tr = 1.0 - F.cosine_similarity(
                            coup.detach(), self.coupling_ref.detach().expand_as(coup), dim=-1)
                        self._dist_log.append(float(d_tr.mean()))
                        if len(self._dist_log) > 4000:
                            self._dist_log.pop(0)
            if self.ref_ready.item():
                d = 1.0 - F.cosine_similarity(coup, self.coupling_ref.detach().expand_as(coup), dim=-1)
                omega = torch.sigmoid(self.omega_a * (self.omega_d0 - d))     # (B,)
            else:
                omega = coup.new_full((coup.size(0),), 0.5)
        else:
            omega = coup.new_full((coup.size(0),), 1.0)
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

        ext = self.U(E_proj)                                       # 专家状态由入射边更新
        slf = self.V(mem_expanded)
        m_tilde = self.ln_mem(ext + mutual + slf)
        m_tilde = torch.tanh(m_tilde)
        eta_in = torch.cat([E_proj, mem_expanded], dim=-1)
        eta = self.eta_net(eta_in)
        mem_new = (1 - eta) * mem_expanded + eta * m_tilde

        if self.training:
            with torch.no_grad():
                self.mem.data = 0.9 * self.mem + 0.1 * mem_new.mean(0)

        O = (gates.unsqueeze(-1) * mem_new).sum(dim=1)
        O = self.proj_out(O)

        kl_mod_per_neuron = F.softplus(self.kl_mod_net(gates.detach()))
        return O, gates, share_mask, kl_mod_per_neuron, omega

    def set_temp(self, temp):
        self.temp = temp

    @torch.no_grad()
    def calibrate_omega(self, quantile=0.95):
        """训练结束标定: d0 = 训练期快慢失配度的 quantile 分位.
        全部统计量来自训练分布 (无测试泄漏) — 熟悉场域 ω≥0.5, 陌生场域 ω→0."""
        if not self.field_router or not self._dist_log:
            return
        self.omega_d0 = float(np.percentile(self._dist_log, quantile * 100))

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
    def louvain_communities(self):
        if self.n_communities > 0:
            ids = self.community_ids.cpu().tolist()
            comms = {}
            for i, cid in enumerate(ids):
                comms.setdefault(cid, []).append(i)
            return comms
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
        e = F.normalize(self.expertise, dim=-1)
        sim = e @ e.T
        W_norms = self.W_mutual.view(self.N, self.N, -1).norm(dim=-1).sum(dim=1)
        return {
            "expertise": self.expertise.detach().cpu(),
            "community_ids": self.community_ids.detach().cpu(),
            "community_mask": self.community_mask.detach().cpu(),
            "W_outgoing": W_norms.detach().cpu(),
            "expertise_sim": sim.detach().cpu(),
            "gate_bias": self.gate_bias.detach().cpu(),
            "temperature": self.temp,
        }

# =========================================================================
# 5. StabilityPredHead — 预迁移 = 预测情绪状态的时序持续性 (跨trial, 可证伪)
# =========================================================================
class StabilityPredHead(nn.Module):
    """P4(iter4): 输入当前窗口的 (Z_init 情绪本征, A 耦合锚点) →
    预测下一 trial 的情绪标签是否保持 (二分类 logits)。
    基线 = 训练数据的真实保持率 (非50%, 非100%) — 显著超过才有意义。
    与分类器彻底解耦: 输出不进融合层, 梯度只回流到共享表示 (预迁移压力)。
    附带耦合自洽约束 (Pass1): embed 诊断 Z 是否保留耦合结构 (P5)。"""

    def __init__(self, K=K_LATENT, D=D_MODEL, Dc=D_COUP_ESS, d_plv=None, seed=0):
        super().__init__()
        assert d_plv is not None and d_plv > Dc
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d_plv, Dc, generator=g)
        W = W / W.norm(dim=0, keepdim=True)
        self.register_buffer('W_proj', W / math.sqrt(d_plv))
        self.predictor = nn.Sequential(
            nn.Linear(K + D, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))
        # 耦合自洽读出: Z → ĉ_self (Pass1约束, 与分类/稳定性头解耦)
        self.coup_readout = nn.Sequential(
            nn.Linear(K, 64), nn.GELU(), nn.Linear(64, Dc))

    @torch.no_grad()
    def embed(self, plv):
        """plv (B,F,P,Tc) → 耦合本征 (B, Dc) — 仅诊断用"""
        flat = plv.mean(-1).reshape(plv.size(0), -1)
        return flat @ self.W_proj

    def forward(self, Z_init, A):
        feat = torch.cat([Z_init, A], dim=-1)
        return {"stab_logits": self.predictor(feat),
                "c_self": self.coup_readout(Z_init)}

# =========================================================================
# 6. DAME_Coupling — 顶模型
# =========================================================================
class DAME_Coupling(nn.Module):
    """脑区耦合DAME v5. 机制与耦合深度绑定 (见文件头 L1-L6).

    Flags: use_coupling / use_water / use_reflux / use_mutual / use_pred"""

    def __init__(self, groups=REGION_GROUPS, D=D_MODEL, use_coupling=True,
                 use_water=True, use_reflux=True, use_mutual=True, use_pred=True,
                 plv_adj=None, aux_losses=True):
        super().__init__()
        self.groups = groups
        self.R = len(groups)
        self.use_coupling = use_coupling
        self.use_water = use_water
        self.use_reflux = use_reflux and use_water
        self.use_mutual = use_mutual
        # 公平性消融: aux_losses=False → 仅CE训练 (完整架构, 无辅助损失;
        # 回答"精度来自架构还是多损失正则"的审稿质疑, 2026-08-16)
        self.aux_losses = aux_losses
        # 预测与回流解耦: NoReflux消融只删回流, 预测头独立可测 (v4教训: 双重消融=混淆)
        self.use_pred = use_pred

        self.coupling = RegionCouplingV2(groups, D, use_coupling=use_coupling)
        self.water = (WaterCycleV2(D, K_LATENT, use_reflux=self.use_reflux)
                      if use_water else None)
        # 无水循环时把A投影到K维本征 (预测头输入维度统一)
        self.z_to_k = nn.Linear(D, K_LATENT) if (not use_water and use_pred) else None
        if use_mutual:
            self.mutual = MutualSocietyV3(N=self.R, D=D, d_mem=D_MEM,
                                          n_communities=4, plv_adj=plv_adj)
            self.mutual._bind_pair_indices(self.coupling.pair_i, self.coupling.pair_j)
        else:
            self.mutual = None
        if self.use_pred:
            P = len(self.coupling.pair_i)
            self.pred_head = StabilityPredHead(K_LATENT, D, d_plv=N_BANDS * P)
        else:
            self.pred_head = None
        # 融合 v6: 权重调制 (策略调制器) — 社会不再是可加项, 而是调制各机制项的权重:
        #   f = Σ_i w_i·term_i,  term = [proj_pool(池化耦合), proj_pow(功率)] (+ [proj_anchor+proj_z] 水循环)
        #   use_mutual: w = softmax(proj_w(O)) ← 社会输出策略化调制各机制
        #   NoMutual:   w = 均匀 1/n_terms   ← 对照臂结构相同, 只删社会调制
        # iter5教训(L1): 可加项浮在下游可加槽位会被 LayerNorm 中和; 调制走非线性权重路径
        self.n_terms = 3 if use_water else 2
        self.proj_pool = nn.Linear(D, D)
        self.proj_anchor = nn.Linear(D, D) if use_water else None
        self.proj_z = nn.Linear(K_LATENT, D) if use_water else None
        self.proj_pow = nn.Linear(D_POW, D)
        self.proj_w = nn.Linear(D, self.n_terms) if use_mutual else None  # 策略调制器: 社会输出→机制权重
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
        H_coup, H_pow, plv = self.coupling(X)
        H_pow_pooled = H_pow.mean(dim=1)                     # (B, 64)
        H_pool = H_coup.mean(dim=1)                          # (B, D) Base路径

        if self.use_water:
            Z, A, kl, kl_w, convergence, Z_init, reflux_mag = self.water(H_coup)
        else:
            A = H_pool                                        # (B, D)
            Z = A
            kl = kl_w = torch.tensor(0.0, device=X.device)
            Z_init = self.z_to_k(A) if self.z_to_k is not None else A
            convergence = [0.0]
            reflux_mag = torch.tensor(0.0, device=X.device)

        if self.use_mutual:
            O, gates, share_mask, kl_mod, omega = self.mutual(plv)
            w_soc = F.softmax(self.proj_w(O), dim=-1)      # (B, n_terms) 社会策略调制
        else:
            O = torch.zeros_like(A)
            w_soc = None
            omega = None
            gates = torch.zeros(X.size(0), 1, device=X.device)
            kl_mod = torch.zeros(X.size(0), 1, device=X.device)

        # 融合 v6: 权重调制 f = Σ w_i·term_i (NoMutual=均匀权重, 结构相同)
        # v6.1 场域路由 ω 已证伪关闭 (FIELD_ROUTER=False → ω≡1, 社会全权重)
        # iter4: ĉ 不进入融合 (预测头与分类器彻底解耦, 恢复忠实预测职责)
        terms = [self.proj_pool(H_pool), self.proj_pow(H_pow_pooled)]
        if self.use_water:
            terms.append(self.proj_anchor(A) + self.proj_z(Z))
        T = torch.stack(terms, dim=1)                      # (B, n_terms, D)
        w = T.new_full((T.size(0), T.size(1)), 1.0 / T.size(1))  # 均匀权重 (基准)
        if w_soc is not None:
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
        if not self.aux_losses:
            return F.cross_entropy(out["logits"], labels, label_smoothing=0.1)
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
            pred = self.pred_head(out["Z_init"], out["A"])
            # Pass1: 耦合自洽 — Z 必须保留耦合结构 (P5 信息瓶颈保耦合)
            loss = loss + PRED_W * F.mse_loss(
                pred["c_self"], self.pred_head.embed(out["plv"]))
            # Pass2: 跨trial情绪稳定性预测 (P4 预迁移, iter4)
            if y_next is not None:
                stab_warm = min(1.0, max(0.0, (self._current_epoch - 3)) / 5.0)
                stab_target = (labels == y_next).long()
                loss = loss + STAB_W * stab_warm * F.cross_entropy(
                    pred["stab_logits"], stab_target)

        if self.use_reflux and isinstance(out["reflux_mag"], torch.Tensor):
            if out["reflux_mag"].numel() == 1:
                loss = loss + REFLUX_W * F.relu(0.01 - out["reflux_mag"])
        return loss

# =========================================================================
# 7. Baselines — raw input (B, 62, 800), 5-year SOTA (v4移植)
# =========================================================================
class EEGNet_Raw(nn.Module):
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
        x = X.unsqueeze(1)
        return self.clf(self.block2(self.block1(x)).flatten(1))

class ChannelMatmulConv(nn.Module):
    """等价 Conv2d(C_in, C_out, (C_spatial, 1)) 的矩阵乘法实现:
    原生 conv2d (62,1) 内核在 Blackwell nightly 上间歇挂起(卡在 _conv_forward,
    2026-08-13 全部挂点), 改用 einsum 走标准 gemm 规避。输出形状与 Conv2d 一致。"""
    def __init__(self, C_in, C_out, C_spatial):
        super().__init__()
        self.W = nn.Parameter(torch.empty(C_out, C_in, C_spatial))
        self.b = nn.Parameter(torch.empty(C_out))
        bound = 1.0 / math.sqrt(C_in * C_spatial)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.zeros_(self.b)

    def forward(self, h):  # (B, C_in, C_spatial, T) -> (B, C_out, 1, T)
        return (torch.einsum('oic,bict->bot', self.W, h) + self.b[:, None]
                ).unsqueeze(2)

class TSception_Raw(nn.Module):
    def __init__(self, C=62, nc=N_CLASSES):
        super().__init__()
        self.tconv1 = nn.Conv2d(1, 16, (1, 129), padding=(0, 64))
        self.tconv2 = nn.Conv2d(1, 16, (1, 65), padding=(0, 32))
        self.tconv3 = nn.Conv2d(1, 16, (1, 33), padding=(0, 16))
        self.pool = nn.AvgPool2d((1, 4))
        self.sconv = ChannelMatmulConv(48, 48, C)
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
    def __init__(self, C=62, nc=N_CLASSES):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 33), padding=(0, 16)), nn.BatchNorm2d(32), nn.ELU())
        self.gconv1 = nn.Sequential(nn.Conv1d(C * 32, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.gconv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(128, nc)

    def forward(self, X):
        x = self.tconv(X.unsqueeze(1))
        B, F, C_out, T_out = x.shape
        x = self.gconv2(self.gconv1(x.reshape(B, F * C_out, T_out)))
        return self.clf(self.pool(x).squeeze(-1))

class EEGConformer_Raw(nn.Module):
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
        h = self.bn2(self.spatial_conv(F.gelu(self.bn(h)))).squeeze(2)
        h = F.adaptive_avg_pool1d(h, 100)
        h = h.transpose(1, 2)
        T_out = h.size(1)
        pos = torch.arange(T_out, device=X.device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, h.size(-1), 2, device=X.device).float()
                        * (-math.log(10000.0) / h.size(-1)))
        pe = torch.zeros(1, T_out, h.size(-1), device=X.device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        h = self.transformer(h + pe)
        return self.clf(h.mean(dim=1))

class LMDA_Net(nn.Module):
    def __init__(self, C=62, nc=N_CLASSES, D=24, k=24):
        super().__init__()
        self.conv1 = nn.Conv2d(1, D, (1, 51), padding=(0, 25))
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, (C, 1), groups=D)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, k, (1, 1))
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
# 8. Data loader — raw windows + prediction pairs (w_k ↔ w_{k+2})
# =========================================================================
def load_raw_with_pairs(n_subjects=None, norm_sessions=None):
    """norm_sessions=None: 逐被试全会话统计 (LOSO 标准做法).
    norm_sessions=(1,2): 统计量仅来自指定会话 — 跨会话协议专用,
    防止测试会话统计量参与训练窗归一化 (2026-08-16 泄露审计修复)."""
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
                wids.append((session, t))  # (session, trial) — 修复跨session键碰撞

        if not trials:
            continue
        stats_trials = trials if norm_sessions is None else \
            [t for t, w in zip(trials, wids) if w[0] in norm_sessions]
        cat = np.concatenate(stats_trials, axis=1)
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
                all_wid.append((subj_id - 1, wid[0], wid[1], k))
                k += 1
        print(f"  [Data] subject {subj_id}: {len(all_X)} windows ({time.time() - t0:.0f}s)",
              flush=True)

    X = torch.stack(all_X)
    y = torch.tensor(all_y, dtype=torch.long)
    subj = torch.tensor(all_subj, dtype=torch.long)
    wid = torch.tensor([[w[0], w[1], w[2], w[3]] for w in all_wid],
                       dtype=torch.long)  # (subj, session, trial, k) — 跨会话划分用

    # 跨trial预测对 (iter4): 本trial末窗 → 同session下一trial首窗
    # 预测目标 = 情绪标签是否保持 (基线=真实保持率, 可证伪)
    pos = {}
    for i, key in enumerate(all_wid):
        s, sess, tr, k = key
        pos.setdefault((s, sess, tr), {})[k] = i
    by_sess = {}
    for (s, sess, tr), win_map in pos.items():
        by_sess.setdefault((s, sess), {})[tr] = win_map
    pair_a, pair_b = [], []
    for (s, sess), trials in by_sess.items():
        trs = sorted(trials)
        for t_cur, t_nxt in zip(trs[:-1], trs[1:]):
            pair_a.append(trials[t_cur][max(trials[t_cur].keys())])
            pair_b.append(trials[t_nxt][min(trials[t_nxt].keys())])
    pair_idx = torch.stack([torch.tensor(pair_a), torch.tensor(pair_b)], dim=1)
    keep_rate = float((y[pair_idx[:, 0]] == y[pair_idx[:, 1]]).float().mean())

    print(f"[Data] RAW SEED-IV: {X.shape[0]} windows | {len(subjects)} subjects | "
          f"classes={dict(Counter(all_y))} | {len(pair_idx)} cross-trial pairs "
          f"(label keep rate {keep_rate:.3f}) | prep {time.time() - t0:.0f}s", flush=True)
    return X, y, subj, pair_idx, wid

# =========================================================================
# 9. Training loops — 两遍训练 (Pass1全窗口自洽 + Pass2配对窗口真实预测)
# =========================================================================
def train_epoch_dame(model, X_train, y_train, pair_map, opt, bs=BATCH_SIZE):
    model.train()
    total_loss, n_b = 0.0, 0

    # Pass 1: all windows — CE + KL + mutual losses + 自洽耦合预测目标
    n = X_train.size(0)
    idx = torch.randperm(n)
    for i in range(0, n, bs):
        bidx = idx[i:i + bs]
        xb = X_train[bidx].to(DEVICE)
        yb = y_train[bidx].to(DEVICE)
        out = model(xb)
        loss = model.compute_loss(out, yb, None)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n_b += 1

    # Pass 2: prediction pairs — 预测4s后的耦合本征 (P4 预迁移)
    if model.use_pred and pair_map and getattr(model, "aux_losses", True):
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
            plv_next = out_b["plv"].detach()
            yb_next = y_train[torch.tensor(bb)].to(DEVICE)
            loss = model.compute_loss(out, yb, y_next=yb_next, plv_next=plv_next)
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
        if is_dame or is_dann:
            out = model(xb)["logits"]
        else:
            out = model(xb)
        preds.append(out.argmax(1).cpu())
    pred = torch.cat(preds)
    acc = (pred == y_test).float().mean().item()
    return acc, pred

@torch.no_grad()
def collect_diag(model, X_test, y_test, pair_map_test, bs=128):
    """机制级诊断 (留出被试上):
    1. 每类情绪PLV矩阵 (脑区联动耦合随情绪重组)
    2. 策略指纹: 门控均值 + 社区分配 (专家涌现/策略不动如山)
    3. 预测功能三指标: selfMSE(Z编码耦合) / predMSE(4s后) / trivMSE(不预测基线)"""
    model.eval()
    P = len(model.coupling.pair_i)
    plv_sums = torch.zeros(N_CLASSES, N_BANDS, P, T_COUP, device=DEVICE)
    plv_cnt = torch.zeros(N_CLASSES, device=DEVICE)
    gates_sum = torch.zeros(model.R if model.use_mutual else 1, device=DEVICE)
    omega_sum, omega_n = 0.0, 0
    self_mse_list = []

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
            if out.get("omega") is not None:
                omega_sum += out["omega"].sum().item(); omega_n += out["omega"].size(0)
        if model.use_pred and model.pred_head is not None:
            po = model.pred_head(out["Z_init"], out["A"])
            self_mse_list.append(
                F.mse_loss(po["c_self"], model.pred_head.embed(out["plv"]),
                           reduction='none').mean(-1).cpu())

    plv_per_class = (plv_sums / plv_cnt.clamp(min=1).view(-1, 1, 1, 1)).cpu().numpy()
    gates_mean = (gates_sum / max(X_test.size(0), 1)).cpu().numpy()
    self_mse = float(torch.cat(self_mse_list).mean()) if self_mse_list else float('nan')

    diag = {
        "plv_per_class": plv_per_class,
        "gates_mean": gates_mean,
        "self_pred_mse": self_mse,
        "community_ids": (model.mutual.community_ids.cpu().numpy()
                          if model.use_mutual else None),
        "W_mutual_norm": (model.mutual.W_mutual.data.norm().item()
                          if model.use_mutual else None),
        "omega_mean": omega_sum / max(omega_n, 1),
        "omega_d0": (float(model.mutual.omega_d0) if model.use_mutual else float('nan')),
    }

    # 预迁移诊断 (iter4): 跨trial情绪稳定性预测 (可证伪: vs 真实保持率)
    # + 跨trial耦合漂移 (参考: 情绪状态间耦合重组幅度, 论文生理学证据)
    if model.use_pred and model.pred_head is not None and pair_map_test:
        stab_hits, stab_n = 0, 0
        keep_sum, keep_n = 0, 0
        per_class = defaultdict(lambda: [0, 0])
        drift_mse, drift_cos = [], []
        a_list = list(pair_map_test.keys())
        for i in range(0, len(a_list), bs):
            ba = a_list[i:i + bs]
            out_a = model(X_test[ba].to(DEVICE))
            bb = [pair_map_test[a] for a in ba]
            out_b = model(X_test[bb].to(DEVICE))
            ya = y_test[ba]; yb = y_test[bb]
            po = model.pred_head(out_a["Z_init"], out_a["A"])
            pred_keep = po["stab_logits"].argmax(1).cpu()   # 1=保持, 0=切换
            target = (ya == yb).long()
            keep_sum += int(target.sum()); keep_n += len(target)
            stab_hits += int((pred_keep == target).sum()); stab_n += len(target)
            for c in range(N_CLASSES):
                m = ya == c
                if m.any():
                    per_class[c][0] += int((pred_keep[m] == target[m]).sum())
                    per_class[c][1] += int(m.sum())
            c_a = model.pred_head.embed(out_a["plv"])
            c_b = model.pred_head.embed(out_b["plv"])
            drift_mse.append(F.mse_loss(c_a, c_b, reduction='none').mean(-1).cpu())
            drift_cos.append(F.cosine_similarity(c_a, c_b, dim=-1).cpu())
        diag.update({
            "stab_acc": stab_hits / max(stab_n, 1),
            "stab_prior": keep_sum / max(keep_n, 1),
            "stab_per_class": np.array(
                [per_class[c][0] / max(per_class[c][1], 1) for c in range(N_CLASSES)]),
            "cross_trial_drift_mse": float(torch.cat(drift_mse).mean()),
            "cross_trial_drift_cos": float(torch.cat(drift_cos).mean()),
        })
    return diag

# =========================================================================
# 10. LOSO — cross-subject protocol with per-fold checkpoints
# =========================================================================
def loso_v5(model_factory, X, y, subj, pair_idx, seeds, epochs=EPOCHS,
            kind="dame", done_folds=None, save_fn=None, tag="", verbose=True,
            collector=None):
    """collector: 可选 [(key, acc, pred, y_test)] 收集器 — 弹窗工具取预测结果用."""
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

            train_glob = torch.nonzero(trm).flatten().tolist()
            g2l = {g: li for li, g in enumerate(train_glob)}
            pair_map = {g2l[a]: g2l[b] for a, b in pair_idx.tolist()
                        if a in g2l and b in g2l}

            # 互助强度初始化: 训练集实测PLV邻接 (机制绑定耦合)
            plv_adj = None
            if kind == "dame":
                groups = MODEL_SPECS.get(tag, {}).get("groups", REGION_GROUPS)
                plv_adj = compute_plv_adjacency(groups, X_train)
            model = model_factory(plv_adj).to(DEVICE)
            if kind == "dann":
                model._lam_epoch = 0
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

            t0 = time.time()
            for ep in range(epochs):
                # 看门狗心跳: 每epoch重arm (只有训练循环挂起时才触发dump+exit;
                # 修复2026-08-13包装器一次性看门狗600s无条件杀健康进程的伪崩溃)
                try:
                    import faulthandler as _fh
                    _fh.cancel_dump_traceback_later()
                    _fh.dump_traceback_later(600, exit=True)
                except Exception:
                    pass
                if kind == "dame":
                    model.set_epoch(ep, epochs)
                    loss = train_epoch_dame(model, X_train, y_train, pair_map, opt)
                elif kind == "plain":
                    loss = train_epoch_plain(model, X_train, y_train, opt)
                elif kind == "dann":
                    model._lam_epoch = ep
                    loss = train_epoch_dann(model, X_train, y_train, subj_train, opt)
                else:
                    loss = train_epoch_coral(model, X_train, y_train, X_test, opt)
                sched.step()
                if verbose and (ep % 5 == 4 or ep == epochs - 1):
                    acc, _ = evaluate(model, X_test, y_test, is_dame=(kind != "plain"),
                                      is_dann=(kind == "dann"))
                    print(f"    [{s + 1}/{Ns}] ep{ep + 1}: loss={loss:.3f} "
                          f"val_acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)

            # 场域路由阈值标定 (训练分布 95 分位, 无测试泄漏)
            if kind == "dame" and getattr(model, "mutual", None) is not None:
                model.mutual.calibrate_omega()

            acc, pred = evaluate(model, X_test, y_test, is_dame=(kind != "plain"),
                                 is_dann=(kind == "dann"))
            per_seed[key] = acc
            if collector is not None:
                collector.append((key, acc, pred.detach().cpu(), y_test.cpu()))
            per_class = {c: (pred == c).float().mean().item() for c in range(N_CLASSES)}
            print(f"  [{s + 1}/{Ns}] seed{seed} acc={acc:.4f} "
                  f"per_class={per_class} ({time.time() - t0:.0f}s)", flush=True)

            if kind == "dame" and isinstance(model, DAME_Coupling):
                test_glob = torch.nonzero(tm).flatten().tolist()
                g2l_t = {g: li for li, g in enumerate(test_glob)}
                pair_map_test = {g2l_t[a]: g2l_t[b] for a, b in pair_idx.tolist()
                                 if a in g2l_t and b in g2l_t}
                diag = collect_diag(model, X_test, y_test, pair_map_test)
                ddir = os.path.join(RESULTS_DIR, "diag_v5")
                os.makedirs(ddir, exist_ok=True)
                np.savez(os.path.join(ddir, f"{key}.npz"), **diag)
                active = int((diag["gates_mean"] > 0.5).sum()) if model.use_mutual else 0
                pstr = (f"stab={diag.get('stab_acc', float('nan')) * 100:.1f}% "
                        f"(prior={diag.get('stab_prior', float('nan')) * 100:.1f}%) "
                        f"selfMSE={diag['self_pred_mse']:.4f} "
                        f"driftCos={diag.get('cross_trial_drift_cos', float('nan')):.3f} "
                        f"active={active}/{model.R} "
                        f"omega={diag.get('omega_mean', float('nan')):.2f}")
                print(f"    [diag] {pstr}", flush=True)
            if save_fn:
                save_fn(key, acc)
            del model, opt
            torch.cuda.empty_cache()
    return per_seed

# =========================================================================
# 11. Cross-session protocol — 跨会话协议 (LOSO=跨被试; 本协议=跨会话小域差)
# =========================================================================
def session_run_v5(model_factory, X, y, subj, wid, pair_idx, seeds, epochs=EPOCHS,
                   kind="dame", done_folds=None, save_fn=None, tag="",
                   sessions_train=(1, 2), sessions_test=(3,), verbose=True,
                   collector=None):
    """跨会话协议: 训练=指定会话(默认1+2), 测试=指定会话(默认3), 全部被试合并.
    LOSO=跨被试(最大域差, 个体差异); 本协议=跨会话(同日间漂移, 小域差) —
    社会的泛化原始战场: 门控需"认出今天的耦合状态→选策略"."""
    st_t = torch.tensor(list(sessions_train), dtype=torch.long)
    se_t = torch.tensor(list(sessions_test), dtype=torch.long)
    tm = torch.isin(wid[:, 1], se_t)
    trm = torch.isin(wid[:, 1], st_t)
    X_train, y_train = X[trm], y[trm]
    X_test, y_test = X[tm], y[tm]
    sess_name = "".join(map(str, sessions_test))
    print(f"  [sess-split] train={sessions_train} ({X_train.size(0)} windows) "
          f"→ test={sessions_test} ({X_test.size(0)} windows)", flush=True)
    per_seed = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        key = f"{tag}_{kind}_sess{sess_name}_seed{seed}"
        if done_folds and key in done_folds:
            print(f"  [skip] {key} (done)")
            continue
        train_glob = torch.nonzero(trm).flatten().tolist()
        g2l = {g: li for li, g in enumerate(train_glob)}
        pair_map = {g2l[a]: g2l[b] for a, b in pair_idx.tolist()
                    if a in g2l and b in g2l}

        plv_adj = None
        if kind == "dame":
            groups = MODEL_SPECS.get(tag, {}).get("groups", REGION_GROUPS)
            plv_adj = compute_plv_adjacency(groups, X_train)
        model = model_factory(plv_adj).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

        t0 = time.time()
        for ep in range(epochs):
            try:
                import faulthandler as _fh
                _fh.cancel_dump_traceback_later()
                _fh.dump_traceback_later(600, exit=True)
            except Exception:
                pass
            if kind == "dame":
                model.set_epoch(ep, epochs)
                loss = train_epoch_dame(model, X_train, y_train, pair_map, opt)
            elif kind == "plain":
                loss = train_epoch_plain(model, X_train, y_train, opt)
            else:
                loss = train_epoch_coral(model, X_train, y_train, X_test, opt)
            sched.step()
            if verbose and (ep % 5 == 4 or ep == epochs - 1):
                acc, _ = evaluate(model, X_test, y_test, is_dame=(kind != "plain"),
                                  is_dann=(kind == "dann"))
                print(f"    [sess{sess_name}] ep{ep + 1}: loss={loss:.3f} "
                      f"val_acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)

        # 场域路由阈值标定 (训练分布 95 分位, 无测试泄漏)
        if kind == "dame" and getattr(model, "mutual", None) is not None:
            model.mutual.calibrate_omega()

        acc, pred = evaluate(model, X_test, y_test, is_dame=(kind != "plain"),
                             is_dann=(kind == "dann"))
        per_seed[key] = acc
        if collector is not None:
            collector.append((key, acc, pred.detach().cpu(), y_test.cpu()))
        per_class = {c: (pred == c).float().mean().item() for c in range(N_CLASSES)}
        print(f"  [sess{sess_name}] seed{seed} acc={acc:.4f} "
              f"per_class={per_class} ({time.time() - t0:.0f}s)", flush=True)

        if kind == "dame" and isinstance(model, DAME_Coupling):
            test_glob = torch.nonzero(tm).flatten().tolist()
            g2l_t = {g: li for li, g in enumerate(test_glob)}
            pair_map_test = {g2l_t[a]: g2l_t[b] for a, b in pair_idx.tolist()
                             if a in g2l_t and b in g2l_t}
            diag = collect_diag(model, X_test, y_test, pair_map_test)
            ddir = os.path.join(RESULTS_DIR, "diag_v5")
            os.makedirs(ddir, exist_ok=True)
            np.savez(os.path.join(ddir, f"{key}.npz"), **diag)
            active = int((diag["gates_mean"] > 0.5).sum()) if model.use_mutual else 0
            print(f"    [diag] active={active}/{model.R} "
                  f"driftCos={diag.get('cross_trial_drift_cos', float('nan')):.3f} "
                  f"omega={diag.get('omega_mean', float('nan')):.2f}",
                  flush=True)
        if save_fn:
            save_fn(key, acc)
        del model, opt
        torch.cuda.empty_cache()
    return per_seed

# =========================================================================
# 12. Model registry
# =========================================================================
def make_variant(**kw):
    def _f(plv_adj=None):
        return DAME_Coupling(plv_adj=plv_adj, **kw)
    return _f

DAME_ARMS = ["DAME-C5", "C5-NoPred", "C5-NoPredNoMutual", "C5-NoMutual",
             "C5-NoWater", "C5-NoReflux", "C5-Base", "C5-NoCoupling"]

NODE_GROUPS = [[i] for i in range(62)]

MODEL_SPECS = {
    "DAME-C5":      dict(fn=make_variant(), kind="dame", groups=REGION_GROUPS),
    "C5-NoPred":    dict(fn=make_variant(use_pred=False), kind="dame", groups=REGION_GROUPS),
    "C5-CEOnly":    dict(fn=make_variant(aux_losses=False), kind="dame",
                         groups=REGION_GROUPS),
    "C5-NoPredNoMutual": dict(fn=make_variant(use_pred=False, use_mutual=False),
                              kind="dame", groups=REGION_GROUPS),
    "C5-NoMutual":  dict(fn=make_variant(use_mutual=False), kind="dame", groups=REGION_GROUPS),
    "C5-NoWater":   dict(fn=make_variant(use_water=False), kind="dame", groups=REGION_GROUPS),
    "C5-NoWaterNoPred": dict(fn=make_variant(use_water=False, use_pred=False),
                             kind="dame", groups=REGION_GROUPS),
    "C5-NoReflux":  dict(fn=make_variant(use_reflux=False), kind="dame", groups=REGION_GROUPS),
    "C5-Base":      dict(fn=make_variant(use_water=False, use_mutual=False,
                                         use_pred=False, use_reflux=False),
                         kind="dame", groups=REGION_GROUPS),
    "C5-NoCoupling": dict(fn=make_variant(use_coupling=False), kind="dame",
                          groups=REGION_GROUPS),
    "C5-Node":      dict(fn=make_variant(groups=NODE_GROUPS),
                         kind="dame", groups=NODE_GROUPS),
    "EEGNet":         dict(fn=lambda plv_adj=None: EEGNet_Raw(), kind="plain"),
    "TSception":      dict(fn=lambda plv_adj=None: TSception_Raw(), kind="plain"),
    "DGCNN":          dict(fn=lambda plv_adj=None: DGCNN_Raw(), kind="plain"),
    "EEGConformer":   dict(fn=lambda plv_adj=None: EEGConformer_Raw(), kind="plain"),
    "LMDA-Net":       dict(fn=lambda plv_adj=None: LMDA_Net(), kind="plain"),
    "DANN":           dict(fn=lambda plv_adj=None: DANN_Raw(n_subjects=15), kind="dann"),
    "DeepCORAL":      dict(fn=lambda plv_adj=None: DeepCORAL_Raw(), kind="coral"),
}

# =========================================================================
# 12. Main — quick smoke / fast iteration / final full run
# =========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 subjects, 1 seed, 2 epochs")
    ap.add_argument("--fast", type=int, default=0,
                    help="fast iteration: use first N subjects (no global run)")
    ap.add_argument("--subjects", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="种子起点索引 (0=42, 1=123, 2=789; 配合--seeds 1 重跑特定种子)")
    ap.add_argument("--field-router", action="store_true",
                    help="启用场域路由器 (默认关闭 — 三判决门证伪后退役, 仅复现实验用)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--tag", type=str, default="v5")
    ap.add_argument("--session-split", action="store_true",
                    help="跨会话协议: 训练/测试按会话划分 (默认训练1+2, 测试3)")
    ap.add_argument("--sessions-train", type=str, default="1,2",
                    help="跨会话协议训练会话 (逗号分隔)")
    ap.add_argument("--sessions-test", type=str, default="3",
                    help="跨会话协议测试会话 (逗号分隔)")
    args = ap.parse_args()

    if args.quick:
        args.subjects, args.seeds, args.epochs = 2, 1, 2
        print("[QUICK MODE] 2 subjects, 1 seed, 2 epochs (smoke)")
    if args.fast:
        args.subjects = args.fast
        print(f"[FAST MODE] {args.fast} subjects (iteration, no global run)")

    seeds = ALL_SEEDS[args.seed_start: args.seed_start + args.seeds]
    if not seeds:
        raise SystemExit("--seed-start 越界 (ALL_SEEDS 共 3 个)")
    models = args.models.split(",") if args.models else DAME_ARMS
    out_path = os.path.join(RESULTS_DIR, f"eeg_{args.tag}_results.json")
    done = {}
    if os.path.exists(out_path):
        done = json.load(open(out_path))

    if args.field_router:
        global FIELD_ROUTER
        FIELD_ROUTER = True
        print("[ROUTER ON] 场域路由器启用 (证伪/复现实验)")

    st = se = None
    if args.session_split:
        st = [int(x) for x in args.sessions_train.split(",")]
        se = [int(x) for x in args.sessions_test.split(",")]

    # 跨会话协议: 标准化统计量仅来自训练会话 (泄露审计修复 2026-08-16)
    X, y, subj, pair_idx, wid = load_raw_with_pairs(
        n_subjects=args.subjects, norm_sessions=tuple(st) if st else None)

    def save_fn(key, acc):
        done[key] = acc
        json.dump(done, open(out_path, "w"), indent=2)

    for mname in models:
        spec = MODEL_SPECS[mname]
        print(f"\n{'=' * 60}\nMODEL: {mname} ({spec['kind']})\n{'=' * 60}", flush=True)
        if args.session_split:
            session_run_v5(spec["fn"], X, y, subj, wid, pair_idx, seeds,
                           epochs=args.epochs, kind=spec["kind"],
                           done_folds=done, save_fn=save_fn, tag=mname,
                           sessions_train=st, sessions_test=se)
        else:
            loso_v5(spec["fn"], X, y, subj, pair_idx, seeds,
                    epochs=args.epochs, kind=spec["kind"],
                    done_folds=done, save_fn=save_fn, tag=mname)

    # ---- Summary + ablation deltas ----
    stats = {}
    for mname in models:
        fold_accs = [v for k, v in done.items()
                     if k.startswith(f"{mname}_") and isinstance(v, float)
                     and ("_dame_s" in k or "_plain_s" in k or "_dann_s" in k
                          or "_coral_s" in k)]
        if not fold_accs:
            continue
        stats[mname] = {
            "mean": float(np.mean(fold_accs)), "std": float(np.std(fold_accs, ddof=1))
            if len(fold_accs) > 1 else 0.0, "n_folds": len(fold_accs),
            "folds": fold_accs,
        }

    print(f"\n{'=' * 70}\nV5 ITERATION SUMMARY — {args.tag} "
          f"({args.subjects}subj x {args.seeds}seed x {args.epochs}ep)\n{'=' * 70}")
    full_mean = None
    for mname in models:
        if mname in stats:
            if mname.startswith("DAME-C5"):
                full_mean = stats[mname]["mean"]
    for mname in models:
        if mname not in stats:
            continue
        s = stats[mname]
        delta = ""
        if full_mean is not None and mname != "DAME-C5":
            delta = f"  Δ={s['mean'] - full_mean:+.4f}"
        print(f"{mname:<16} {s['mean'] * 100:>6.2f}±{s['std'] * 100:.2f} "
              f"(n={s['n_folds']}){delta}")

    # 迭代裁决 (2026-08-13 新持平目标; 旧顶刊线已废止):
    #   硬门槛 = 耦合贡献证据 (NoCoupling 为负) + 完整版不输 Base
    #   持平判定 (vs 同协议基线) 在基线表补齐后单独裁决, 不在此处
    verdict = "FAIL"
    if full_mean is not None:
        checks = {
            "NoCoupling<0": ("C5-NoCoupling" in stats
                             and stats["C5-NoCoupling"]["mean"] < full_mean),
            "vsBase>=0": ("C5-Base" in stats
                          and full_mean >= stats["C5-Base"]["mean"]),
        }
        info = {
            "NoWater": ("C5-NoWater" in stats
                        and stats["C5-NoWater"]["mean"] < full_mean),
            "NoReflux": ("C5-NoReflux" in stats
                         and stats["C5-NoReflux"]["mean"] < full_mean),
            "NoPred": ("C5-NoPred" in stats
                       and stats["C5-NoPred"]["mean"] < full_mean),
            "NoMutual": ("C5-NoMutual" in stats
                         and stats["C5-NoMutual"]["mean"] < full_mean),
        }
        passed = [k for k, v in checks.items() if v]
        failed = [k for k, v in checks.items() if not v]
        verdict = "PASS" if not failed else "FAIL"
        print(f"\n迭代裁决(持平目标): {verdict}  (硬门槛 {len(passed)}/{len(checks)})")
        if failed:
            print(f"  未过: {failed}")
        print(f"  参考: " + " ".join(
            f"{k}={'✓' if v else '✗'}" for k, v in info.items()))

    # ---- 落盘迭代记录 (回溯) ----
    it_path = os.path.join(RESULTS_DIR, "v5_iterations.jsonl")
    entry = {
        "tag": args.tag, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subjects": args.subjects, "seeds": args.seeds, "epochs": args.epochs,
        "models": stats, "verdict": verdict,
    }
    with open(it_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[iter log] {it_path}")

if __name__ == "__main__":
    main()
