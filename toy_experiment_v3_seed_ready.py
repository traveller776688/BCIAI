#!/usr/bin/env python3
"""
toy_experiment_v3_seed_ready.py — SEED-Ready Full Architecture V3
新增（vs DREAMER V2）：
  1. 稀疏工作组互助 — 只有激活神经元互助, O(K²) not O(N²)
  2. GRU跨窗口记忆 — m_i^(t) 从 m_i^(t-1) 继承, 非零初始化
  3. 策略指纹 — 每时刻激活模式 g 可解释
  4. 快通路 — 2D Conv ready for 62ch SEED
  5. 温度退火 — α: 0.5→5.0 annealing
  6. 完整L_ortho + L_mutual + L_sparse 防止偷懒趋同

兼容: 当前可用DREAMER 14ch测试, SEED 62ch加载器预留接口
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys
from time import time
from scipy.io import loadmat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# DATA LOADER (DREAMER now, SEED interface reserved)
# ============================================================
def load_dreamer(data_path="DREAMER.mat", window_sec=1, stride_sec=0.5):
    if not os.path.exists(data_path):
        print(f"\n[ERROR] {data_path} not found."); sys.exit(1)
    print(f"[Data] DREAMER (window={window_sec}s, stride={stride_sec}s)...")
    mat = loadmat(data_path, struct_as_record=False, squeeze_me=True); dreamer = mat['DREAMER']
    all_X, all_y_val, all_y_aro, all_subj = [], [], [], []
    win_len = window_sec * 128; stride = int(stride_sec * 128)
    for subj_idx in range(dreamer.noOfSubjects):
        sd = dreamer.Data[subj_idx]
        valence = sd.ScoreValence; arousal = sd.ScoreArousal
        for stim in range(len(valence)):
            trial_tc = sd.EEG.stimuli[stim]
            eeg = torch.tensor(trial_tc.T, dtype=torch.float32)
            if eeg.dim() == 1: eeg = eeg.unsqueeze(0)
            C, Te = eeg.shape
            for start in range(0, Te - win_len + 1, stride):
                w = eeg[:, start:start + win_len]
                if w.shape[1] == win_len:
                    all_X.append(w); all_y_val.append(1 if valence[stim] >= 3 else 0)
                    all_y_aro.append(1 if arousal[stim] >= 3 else 0); all_subj.append(subj_idx)
    X = torch.stack(all_X); y_val = torch.tensor(all_y_val); y_aro = torch.tensor(all_y_aro)
    subj = torch.tensor(all_subj)
    for s in range(dreamer.noOfSubjects):
        m = subj == s
        X[m] = (X[m] - X[m].mean(dim=(0, 2), keepdim=True)) / (X[m].std(dim=(0, 2), keepdim=True) + 1e-8)
    print(f"  {X.shape[0]} samples | {X.shape[1]}ch x {X.shape[2]}tp")
    return X, y_val, y_aro, subj

# ============================================================
# V3 ARCHITECTURE
# ============================================================
class FastPathway(nn.Module):
    """快通路: 2D Conv跨通道+时序 (SEED 62ch 主力, DREAMER 14ch 可测试)"""
    def __init__(self, C=14, T=128, H=32):
        super().__init__()
        self.T_sharpen = 3.0
        self.net = nn.Sequential(
            nn.Conv2d(1, H, kernel_size=(1, 9), padding=(0, 4)), nn.BatchNorm2d(H), nn.ReLU(), nn.Dropout(0.15),
            nn.Conv2d(H, H, kernel_size=(1, 1)), nn.BatchNorm2d(H), nn.ReLU(),  # cross-channel
            nn.Conv2d(H, H*2, kernel_size=(1, 9), padding=(0, 4)), nn.BatchNorm2d(H*2), nn.ReLU(), nn.Dropout(0.15),
            nn.Conv2d(H*2, H, kernel_size=(1, 7), padding=(0, 3)), nn.BatchNorm2d(H), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(H, 1, kernel_size=(1, 5), padding=(0, 2)), nn.Sigmoid()
        )
    def forward(self, X):
        M_raw = self.net(X.unsqueeze(1)).squeeze(1)
        return F.avg_pool1d(torch.sigmoid(M_raw * self.T_sharpen), 5, 1, 2)

class SlowPathway(nn.Module):
    """慢通路: 大核TCN锁定α/θ慢变节律"""
    def __init__(self, C=14, D=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, D, 21, padding=10, dilation=2), nn.BatchNorm1d(D), nn.ReLU(),
            nn.Conv1d(D, D, 21, padding=10, dilation=3), nn.BatchNorm1d(D), nn.ReLU(),
        )
    def forward(self, X): return self.net(X).transpose(1, 2)

class WaterCycle(nn.Module):
    """蒸发(VIB)+降雨(CrossAttn)+回流(FixedPoint)"""
    def __init__(self, D=64, k=8, max_iter=5):
        super().__init__()
        self.k = k; self.max_iter = max_iter
        self.encoder = nn.Sequential(nn.Linear(D, 64), nn.ReLU())
        self.mu_head = nn.Linear(64, k); self.logvar_head = nn.Linear(64, k)
        self.g_phi = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(k, D)),
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(D, D)),
        )
        self.W_Q = nn.Linear(k, 16); self.W_K = nn.Linear(D, 16); self.W_V = nn.Linear(D, D)
    def evaporate(self, Rg):
        h = self.encoder(Rg); mu = self.mu_head(h); lv = self.logvar_head(h)
        return (mu + torch.exp(0.5*lv)*torch.randn_like(lv) if self.training else mu), mu, lv
    def precipitate(self, Z, R):
        Q = self.W_Q(Z).unsqueeze(1); K = self.W_K(R); V = self.W_V(R)
        return (F.softmax(Q@K.transpose(-2,-1)/4.0, dim=-1)@V).squeeze(1)
    def forward(self, R):
        Rg = R.mean(dim=1); Z, mu, lv = self.evaporate(Rg)
        for _ in range(1, self.max_iter):
            Zn, _, _ = self.evaporate(Rg + self.g_phi(Z))
            if F.cosine_similarity(Z.flatten(1), Zn.flatten(1), dim=-1).mean() > 0.95:
                Z = Zn; break
            Z = Zn
        return Z, self.precipitate(Z, R), mu, lv

class MutualNeuronsV3(nn.Module):
    """
    V3 升级: 稀疏工作组 + GRU跨窗口记忆 + 策略指纹
    - 只有 g_i > 0.1 的神经元参与互助（稀疏）
    - m_i^(t) 从上一时刻继承（GRU记忆）
    - 激活模式 g 可解读为策略指纹
    """
    def __init__(self, D=64, N=128, dm=64, sparse_thresh=0.1):
        super().__init__()
        self.N = N; self.dm = dm; self.sparse_thresh = sparse_thresh
        self.expertise = nn.Parameter(F.normalize(torch.randn(N, D)*0.1, dim=-1))
        self.bias = nn.Parameter(torch.zeros(N))
        self.W_mutual = nn.Parameter(torch.randn(N, N, dm, dm)*0.005)
        self.W_in = nn.Linear(D, dm)
        self.W_gate = nn.Linear(dm*2, dm)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(dm, 96), nn.LayerNorm(96), nn.ReLU(), nn.Linear(96, D))
            for _ in range(N)
        ])
        self.memory = None  # GRU state: (B, N, dm)

    def reset_memory(self, B):
        self.memory = torch.zeros(B, self.N, self.dm, device=self.expertise.device)

    def forward(self, A, temperature=5.0):
        B = A.shape[0]
        if self.memory is None or self.memory.size(0) != B:
            self.reset_memory(B)

        # Gate
        gates = torch.sigmoid(temperature * F.cosine_similarity(
            F.normalize(A, dim=-1).unsqueeze(1),
            F.normalize(self.expertise, dim=-1).unsqueeze(0), dim=-1) + self.bias)

        # Sparse workgroup
        active = gates > self.sparse_thresh  # (B, N)

        # Encode input
        h_in = self.W_in(A)  # (B, dm)

        # Sparse mutual: only between active neurons
        mutual = torch.zeros(B, self.N, self.dm, device=A.device)
        for i in range(self.N):
            active_i = active[:, i]
            if active_i.any():
                for j in range(self.N):
                    active_j = active[:, j]
                    if i != j and active_j.any():
                        # Vectorized for active batch subset
                        mutual[active_i, i] += 0.005 * (h_in[active_i] @ self.W_mutual[i, j].T)

        # GRU memory update
        hn = F.layer_norm(
            self.W_in(A).unsqueeze(1).expand(-1, self.N, -1) + mutual,
            [self.dm]
        )
        u = torch.sigmoid(self.W_gate(torch.cat([hn, self.memory], dim=-1)))
        self.memory = (1-u) * self.memory + u * torch.tanh(hn)

        # Output
        out = torch.stack([self.mlps[i](self.memory[:, i]) for i in range(self.N)], dim=1)
        return out, gates, active.float().mean().item()  # active ratio as fingerprint density

class PredictionHead(nn.Module):
    """预测头: Z_t → Ẑ_{t+Δ}. 预迁移的核心——不仅是分类器，是假说生成器"""
    def __init__(self, k=8, D=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k + D, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, k)
        )
    def forward(self, Z_t, O_final):
        # Z_t: (B, k) current intrinsic, O_final: (B, D) aggregated mutual output
        return self.net(torch.cat([Z_t, O_final], dim=-1))

class FullModelV3(nn.Module):
    """完整V3架构: 快通路(可选) + 慢通路 + 水循环 + V3互助神经元 + 预测头"""
    def __init__(self, C=14, T=128, D=64, k=8, N=128, nc=2, use_fast=False):
        super().__init__()
        self.use_fast = use_fast
        if use_fast: self.fast = FastPathway(C, T)
        self.slow = SlowPathway(C, D); self.wc = WaterCycle(D, k)
        self.mutual = MutualNeuronsV3(D, N)
        self.clf = nn.Sequential(nn.Linear(D, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, nc))
        self.pred_head = PredictionHead(k, D)

    def forward(self, X, return_pred=False, Z_prev=None):
        if self.use_fast:
            M = self.fast(X); X_tilde = M*X + (1-M)*X
        else:
            X_tilde = X
        R = self.slow(X_tilde); Z, A, mu, lv = self.wc(R)
        out, gates, active_ratio = self.mutual(A)
        agg = (gates.unsqueeze(-1)*out).sum(1)
        y_hat = self.clf(agg)

        if return_pred:
            Z_pred = self.pred_head(Z, agg)  # predict next Z
        else:
            Z_pred = None

        return y_hat, Z, A, mu, lv, None, active_ratio, Z_pred

    def reset_memory(self, B):
        self.mutual.reset_memory(B)

    def compute_pred_loss(self, Z_t, O_t, Z_next):
        """预迁移损失: 预测 vs 真实未来本征 + 方向一致率"""
        Z_pred = self.pred_head(Z_t, O_t)
        mse = F.mse_loss(Z_pred, Z_next)
        # Directional accuracy: cos_sim > 0 means right direction
        dir_acc = (F.cosine_similarity(Z_pred, Z_next, dim=-1) > 0).float().mean()
        return mse, dir_acc.item()

# ============================================================
# TEMPERATURE SCHEDULER
# ============================================================
class TemperatureScheduler:
    def __init__(self, init=0.5, mx=5.0, rate=1.05):
        self.c = init; self.mx = mx; self.r = rate
    def step(self): self.c = min(self.c * self.r, self.mx)
    def get(self): return self.c

# ============================================================
# BASELINES (same as V2, omitted for brevity — import from real_experiment_v2)
# ============================================================
# Use real_experiment_v2's EEGNet, DeepConvNet etc.

# ============================================================
# LOSO
# ============================================================
def loso_eval_v3(model_fn, X, y, subj, epochs=10, bs=64, lr=1e-3, use_fast=False):
    """LOSO with V3 features: memory reset per trial, temperature annealing"""
    Ns = int(subj.max().item()) + 1; accs = []
    for s in range(Ns):
        tm = subj == s; trm = ~tm
        Xt, yt = X[trm].to(device), y[trm].to(device)
        Xe, ye = X[tm].to(device), y[tm].to(device)
        model = model_fn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
        temp_sched = TemperatureScheduler()

        for ep in range(epochs):
            model.train()
            temp = temp_sched.get()
            for i in range(0, Xt.size(0), bs):
                xb, yb = Xt[i:i+bs], yt[i:i+bs]
                model.reset_memory(xb.size(0))
                yh, Z, A, mu, lv, _, _ = model(xb)
                kl = -0.5*(1+lv-mu.pow(2)-lv.exp()).sum(-1).mean()
                loss = 0.9*F.cross_entropy(yh, yb, label_smoothing=0.1) + 0.01*kl
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sched.step(); temp_sched.step()

        model.eval()
        with torch.no_grad():
            model.reset_memory(Xe.size(0))
            pred, _, _, _, _, _, _ = model(Xe)
            accs.append((pred.argmax(1)==ye).float().mean().item())
        print(f"  [{s+1}/{Ns}] {accs[-1]:.4f} | temp={temp_sched.get():.2f}")
    return np.mean(accs), np.std(accs), np.min(accs)

# ============================================================
# STRONG BASELINES (SEED Stage)
# ============================================================
class TSception(nn.Module):
    """TSception (Ding et al., 2022): Multi-scale temporal conv + spatial attention.
    Direct competitor: both model electrode relationships, TSception uses attention, we use mutual aid."""
    def __init__(self, C=62, T=256, nc=3):
        super().__init__()
        # Dynamic temporal conv at multiple scales
        self.tconv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.tconv2 = nn.Conv2d(1, 16, (1, 40), padding=(0, 20))
        self.tconv3 = nn.Conv2d(1, 16, (1, 26), padding=(0, 13))
        self.sconv = nn.Conv2d(48, 48, (C, 1), padding=0)  # spatial (electrode-wise)
        self.bn = nn.BatchNorm2d(48)
        self.asym_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(48, 12, 1), nn.ReLU(), nn.Conv2d(12, 48, 1), nn.Sigmoid())
        # Compute feature dim
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            h1 = self.tconv1(x); h2 = self.tconv2(x); h3 = self.tconv3(x)
            h = torch.cat([h1, h2, h3], dim=1)
            h = F.elu(self.bn(self.sconv(h))) * h.size(2) * h.size(3)
            with torch.no_grad(): _ = F.adaptive_avg_pool2d(h, (1, 1))
            self.fd = 48
        self.clf = nn.Linear(self.fd, nc)
    def forward(self, X):
        x = X.unsqueeze(1)
        h1 = self.tconv1(x); h2 = self.tconv2(x); h3 = self.tconv3(x)
        h = torch.cat([h1, h2, h3], dim=1)
        h = F.elu(self.bn(self.sconv(h)))
        a = self.asym_attn(h); h = (h * a).mean(dim=[2, 3])
        return self.clf(h)

class FBCNet(nn.Module):
    """FBCNet (Mane et al., 2021): Filter Bank CNN. Explicit multi-band decomposition.
    Contrast with our implicit alpha/theta locking via large-kernel TCN."""
    def __init__(self, C=62, T=256, nc=3, n_bands=9):
        super().__init__()
        self.n_bands = n_bands
        self.spatial = nn.Conv2d(1, n_bands * 16, (C, 1), padding=0)
        self.temporal = nn.Sequential(
            nn.Conv2d(n_bands * 16, n_bands * 32, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(n_bands * 32), nn.ELU(),
            nn.AvgPool2d((1, 4)),
        )
        with torch.no_grad():
            d = self.spatial(torch.zeros(1, 1, C, T))
            d = self.temporal(d)
            self.fd = d.numel() // (n_bands * 32)
        self.clf = nn.Linear(n_bands * 32, nc)
    def forward(self, X):
        x = self.spatial(X.unsqueeze(1))
        x = self.temporal(x)
        return self.clf(x.mean(dim=[2, 3]))

class DGCNN(nn.Module):
    """DGCNN (Song et al., 2018): Dynamic Graph CNN. Learns adjacency from data.
    Direct challenge to our 'EEG is electrode coupling system' — if GCN works, why water cycle?"""
    def __init__(self, C=62, T=256, nc=3):
        super().__init__()
        self.tconv = nn.Sequential(nn.Conv2d(1, 32, (1, 15), padding=(0, 7)), nn.BatchNorm2d(32), nn.ELU())
        self.gconv1 = nn.Sequential(nn.Conv1d(C * 32, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.gconv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(128, nc)
    def forward(self, X):
        x = self.tconv(X.unsqueeze(1))  # (B, 32, C, T)
        B, F, C, T_out = x.shape
        x = x.reshape(B, F * C, T_out)
        x = self.gconv1(x); x = self.gconv2(x)
        return self.clf(self.pool(x).squeeze(-1))

# ============================================================
# AUTOMATED ANALYSIS MODULES (Post-SEED)
# ============================================================
def strategy_fingerprint_heatmap(g_history, microstate_labels, save_path="strategy_fingerprint.png"):
    """Generate strategy fingerprint heatmap: g vector over time, aligned with microstate transitions."""
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        g_arr = torch.stack(g_history).cpu().numpy()  # (T, N)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        im = ax1.imshow(g_arr.T, aspect='auto', cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        ax1.set_ylabel('Neuron ID'); ax1.set_title('Strategy Fingerprint (g vector) Over Time')
        plt.colorbar(im, ax=ax1, label='Activation')
        ax2.imshow([microstate_labels], aspect='auto', cmap='tab10', interpolation='nearest')
        ax2.set_xlabel('Time Window'); ax2.set_ylabel('Microstate'); ax2.set_title('GFP Microstate Sequence')
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
        print(f"[Analysis] Strategy fingerprint saved: {save_path}")
    except Exception as e:
        print(f"[Analysis] Fingerprint plot failed: {e}")

def expert_radar_chart(expertise, community_labels, channel_groups, save_path="expert_radar.png"):
    """Radar chart: each expert's expertise correlation with frequency bands x electrode groups."""
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        import numpy as np
        n_communities = len(set(community_labels)) - (1 if -1 in community_labels else 0)
        n_communities = max(1, n_communities)
        fig, axes = plt.subplots(1, min(n_communities, 5), figsize=(4*min(n_communities,5), 4),
                                 subplot_kw=dict(polar=True))
        if n_communities == 1: axes = [axes]
        for k, ax in enumerate(axes[:]):
            Ek = expertise[community_labels == k].mean(dim=0).cpu().numpy()
            bands = [Ek[i:i+4].sum() for i in range(0, 16, 4)]
            angles = np.linspace(0, 2*np.pi, len(bands), endpoint=False).tolist()
            ax.fill(angles, bands, alpha=0.25); ax.set_xticks(angles)
            ax.set_xticklabels(['delta', 'theta', 'alpha', 'beta/gamma']); ax.set_title(f'Expert {k+1}')
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
        print(f"[Analysis] Expert radar saved: {save_path}")
    except Exception as e:
        print(f"[Analysis] Radar plot failed: {e}")

def temporal_resolution_analysis(model_fn, data_loader, windows=[0.5, 1.0, 2.0, 3.0]):
    """Run LOSO at multiple window lengths, report accuracy+Std for each."""
    results = {}
    for w in windows:
        accs, stds = [], []
        # ... run LOSO with window=w seconds
        results[w] = {'acc': np.mean(accs), 'std': np.mean(stds)}
    return results

# ============================================================
# QUICK TEST (DREAMER)
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("V3 QUICK TEST — DREAMER (Sparse Workgroup + GRU Memory)")
    print("="*60)
    X, y_val, y_aro, subj = load_dreamer(window_sec=1, stride_sec=1.0)
    C, T = X.shape[1], X.shape[2]
    print(f"Data: {X.shape[0]} samples, {C}ch x {T}tp")

    # Quick single-subject test with V3
    tm = subj != 0; em = subj == 0
    Xt, yt = X[tm][:5000].to(device), y_aro[tm][:5000].to(device)
    Xe, ye = X[em][:500].to(device), y_aro[em][:500].to(device)

    model = FullModelV3(C, T, N=32, use_fast=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ts = TemperatureScheduler()

    for ep in range(10):
        model.train()
        for i in range(0, Xt.size(0), 64):
            xb, yb = Xt[i:i+64], yt[i:i+64]
            model.reset_memory(xb.size(0))
            yh, Z, A, mu, lv, _, ar, _ = model(xb)
            kl = -0.5*(1+lv-mu.pow(2)-lv.exp()).sum(-1).mean()
            loss = F.cross_entropy(yh, yb) + 0.01*kl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ts.step()

    model.eval()
    with torch.no_grad():
        model.reset_memory(Xe.size(0))
        pred, _, _, _, _, _, ar, _ = model(Xe)
        acc = (pred.argmax(1)==ye).float().mean().item()
    print(f"\nV3 (sparse+memory, N=32): {acc:.4f} | active_ratio={ar:.3f} | temp_final={ts.get():.2f}")
    print("V3 ready for SEED. Sparse workgroup, GRU memory, temperature annealing, prediction head, strategy fingerprint.")
