#!/usr/bin/env python3
"""real_experiment_v2.py — DREAMER Arousal/Valence LOSO benchmark. Phase 1: water cycle + mutual neurons (fast pathway skipped)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, json
from time import time
from scipy.io import loadmat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
def load_dreamer(data_path="DREAMER.mat", window_sec=2):
    if not os.path.exists(data_path):
        print(f"\n[ERROR] {data_path} not found."); sys.exit(1)
    print(f"[Data] Loading DREAMER (window={window_sec}s)...")
    mat = loadmat(data_path, struct_as_record=False, squeeze_me=True)
    dreamer = mat['DREAMER']
    all_X, all_y_val, all_y_aro, all_subj = [], [], [], []
    win_len = window_sec * 128; stride = win_len // 2  # 50% overlap = more data
    n_subjects = dreamer.noOfSubjects
    for subj_idx in range(n_subjects):
        sd = dreamer.Data[subj_idx]
        valence = sd.ScoreValence; arousal = sd.ScoreArousal
        n_stimuli = len(valence)
        for stim in range(n_stimuli):
            trial_tc = sd.EEG.stimuli[stim]
            eeg = torch.tensor(trial_tc.T, dtype=torch.float32)
            if eeg.dim() == 1: eeg = eeg.unsqueeze(0)
            C, Te = eeg.shape
            for start in range(0, Te - win_len + 1, stride):
                w = eeg[:, start:start + win_len]
                if w.shape[1] == win_len:
                    all_X.append(w)
                    all_y_val.append(1 if valence[stim] >= 3 else 0)
                    all_y_aro.append(1 if arousal[stim] >= 3 else 0)
                    all_subj.append(subj_idx)
    X = torch.stack(all_X); y_val = torch.tensor(all_y_val); y_aro = torch.tensor(all_y_aro)
    subj = torch.tensor(all_subj)
    for s in range(n_subjects):
        m = subj == s
        X[m] = (X[m] - X[m].mean(dim=(0, 2), keepdim=True)) / (X[m].std(dim=(0, 2), keepdim=True) + 1e-8)
    print(f"  Samples: {X.shape[0]} | {X.shape[1]}ch x {X.shape[2]}tp")
    print(f"  Valence pos: {y_val.float().mean():.2%} | Arousal pos: {y_aro.float().mean():.2%}")
    return X, y_val, y_aro, subj

# ============================================================
class SlowPathway(nn.Module):
    def __init__(self, C=14, D=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, D, 21, padding=10, dilation=2), nn.BatchNorm1d(D), nn.ReLU(),
            nn.Conv1d(D, D, 21, padding=10, dilation=3), nn.BatchNorm1d(D), nn.ReLU(),
        )
    def forward(self, X): return self.net(X).transpose(1, 2)

class WaterCycle(nn.Module):
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
        return (mu + torch.exp(0.5 * lv) * torch.randn_like(lv) if self.training else mu), mu, lv

    def precipitate(self, Z, R):
        Q = self.W_Q(Z).unsqueeze(1); K = self.W_K(R); V = self.W_V(R)
        attn = F.softmax(Q @ K.transpose(-2, -1) / 4.0, dim=-1)
        return (attn @ V).squeeze(1)

    def forward(self, R):
        Rg = R.mean(dim=1); Z, mu, lv = self.evaporate(Rg)
        for _ in range(1, self.max_iter):
            Zn, _, _ = self.evaporate(Rg + self.g_phi(Z))
            if F.cosine_similarity(Z.flatten(1), Zn.flatten(1), dim=-1).mean() > 0.95:
                Z = Zn; break
            Z = Zn
        return Z, self.precipitate(Z, R), mu, lv

class MutualNeurons(nn.Module):
    def __init__(self, D=64, N=32, dm=64):
        super().__init__()
        self.N = N
        self.expertise = nn.Parameter(F.normalize(torch.randn(N, D) * 0.1, dim=-1))
        self.bias = nn.Parameter(torch.zeros(N))
        self.W_mutual = nn.Parameter(torch.randn(N, N, dm, dm) * 0.005)
        self.W_in = nn.Linear(D, dm); self.W_gate = nn.Linear(dm * 2, dm)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(dm, 96), nn.LayerNorm(96), nn.ReLU(), nn.Linear(96, D))
            for _ in range(N)
        ])

    def forward(self, A, temp=5.0):
        B = A.shape[0]
        gates = torch.sigmoid(temp * F.cosine_similarity(
            F.normalize(A, dim=-1).unsqueeze(1),
            F.normalize(self.expertise, dim=-1).unsqueeze(0), dim=-1) + self.bias)
        h = self.W_in(A).unsqueeze(1).expand(-1, self.N, -1)
        mutual = 0.005 * torch.einsum('bjd,ijde->bie', h, self.W_mutual)
        mutual = mutual - 0.005 * torch.einsum('bid,iide->bie', h, self.W_mutual)
        hn = F.layer_norm(h + mutual, [h.size(-1)])
        u = torch.sigmoid(self.W_gate(torch.cat([hn, torch.zeros_like(hn)], dim=-1)))
        mem = u * torch.tanh(hn)
        out = torch.stack([self.mlps[i](mem[:, i]) for i in range(self.N)], dim=1)
        return out, gates

class FullModel(nn.Module):
    def __init__(self, C=14, T=768, D=64, k=8, N=32, nc=2):
        super().__init__()
        self.slow = SlowPathway(C, D); self.wc = WaterCycle(D, k); self.mutual = MutualNeurons(D, N)
        self.clf = nn.Sequential(nn.Linear(D, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, nc))

    def forward(self, X):
        R = self.slow(X); Z, A, mu, lv = self.wc(R)
        out, gates = self.mutual(A)
        agg = (gates.unsqueeze(-1) * out).sum(1)
        return self.clf(agg), Z, A, mu, lv, None

class EEGNet(nn.Module):
    def __init__(self, C=14, T=768, nc=2):
        super().__init__()
        F1, Dv, F2 = 8, 2, 16
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, 64), padding=(0, 32)), nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * Dv, (C, 1), groups=F1), nn.BatchNorm2d(F1 * Dv),
                                nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * Dv, F1 * Dv, (1, 16), padding=(0, 8), groups=F1 * Dv),
                                nn.Conv2d(F1 * Dv, F2, (1, 1)), nn.BatchNorm2d(F2),
                                nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.25))
        with torch.no_grad(): self.fd = self.b2(self.b1(torch.zeros(1, 1, C, T))).numel()
        self.clf = nn.Linear(self.fd, nc)

    def forward(self, X): return self.clf(self.b2(self.b1(X.unsqueeze(1))).flatten(1))

class DeepConvNet(nn.Module):
    def __init__(self, C=14, T=256, nc=2):
        super().__init__()
        # T=256 → /3=85 → /3=28 → /2=14 → /2=7  (padded convs preserve T through each block)
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), padding=(0, 5)), nn.Conv2d(25, 25, (C, 1)),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(0.25),
            nn.Conv2d(25, 50, (1, 10), padding=(0, 5)), nn.BatchNorm2d(50), nn.ELU(),
            nn.MaxPool2d((1, 3)), nn.Dropout(0.25),
            nn.Conv2d(50, 100, (1, 10), padding=(0, 5)), nn.BatchNorm2d(100), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.25),
            nn.Conv2d(100, 200, (1, 10), padding=(0, 5)), nn.BatchNorm2d(200), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.25))
        with torch.no_grad(): self.fd = self.net(torch.zeros(1, 1, C, T)).numel()
        self.clf = nn.Linear(self.fd, nc)

    def forward(self, X): return self.clf(self.net(X.unsqueeze(1)).flatten(1))

class MLPBaseline(nn.Module):
    def __init__(self, C=14, T=128, nc=2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(C*T, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc))
    def forward(self, X): return self.net(X)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return grad.neg() * ctx.lam, None

class DANN(nn.Module):
    def __init__(self, C=14, T=128, nc=2, n_dom=23):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(1,16,(1,15),padding=(0,7)),nn.BatchNorm2d(16),nn.ReLU(),
                                     nn.Conv2d(16,32,(1,15),padding=(0,7)),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((1,16)))
        with torch.no_grad(): self.fd = self.feature(torch.zeros(1,1,C,T)).numel()
        self.task_clf = nn.Linear(self.fd, nc); self.domain_clf = nn.Sequential(nn.Linear(self.fd,64),nn.ReLU(),nn.Linear(64,n_dom))
    def forward(self, X, dom=None, lam=1.0):
        f = self.feature(X.unsqueeze(1)).flatten(1)
        if dom is not None: rev = GradReverse.apply(f, lam); return self.task_clf(f), self.domain_clf(rev)
        return self.task_clf(f)

class DeepCORAL(nn.Module):
    def __init__(self, C=14, T=128, nc=2):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(1,32,(1,15),padding=(0,7)),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((1,16)))
        with torch.no_grad(): self.fd = self.feature(torch.zeros(1,1,C,T)).numel()
        self.clf = nn.Linear(self.fd, nc)
    def forward(self, X): return self.clf(self.feature(X.unsqueeze(1)).flatten(1))
    def coral_loss(self, xs, xt):
        d = xs.size(1); cs = (xs.T@xs)/(xs.size(0)-1); ct = (xt.T@xt)/(xt.size(0)-1)
        return (cs-ct).pow(2).sum()/(4*d*d)

class SoftMoE_EEG(nn.Module):
    def __init__(self, C=14, T=128, D=64, K=16, nc=2):
        super().__init__(); self.K = K
        self.encoder = nn.Sequential(nn.Conv1d(C,D,15,padding=7),nn.BatchNorm1d(D),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.router = nn.Linear(D,K); self.experts = nn.ModuleList([nn.Sequential(nn.Linear(D,D),nn.ReLU(),nn.Linear(D,D)) for _ in range(K)])
        self.clf = nn.Linear(D, nc)
    def forward(self, X):
        f = self.encoder(X); w = F.softmax(self.router(f), dim=-1)
        eo = torch.stack([e(f) for e in self.experts], dim=1)
        return self.clf((w.unsqueeze(-1)*eo).sum(1))

# ============================================================
# LOSO EVALUATION (supports Ours / EEGNet / DeepConvNet / DANN / CORAL)
def loso_eval(model_fn, X, y, subj, epochs=5, bs=64, lr=1e-3, is_ours=False, is_dann=False, is_coral=False):
    Ns = int(subj.max().item()) + 1
    accs, yt_all, yp_all = [], [], []
    for s in range(Ns):
        tm = subj == s; trm = ~tm
        Xt, yt = X[trm].to(device), y[trm].to(device)
        Xe, ye = X[tm].to(device), y[tm].to(device)
        # --- remap domain labels for DANN (training subjects → 0..N-2) ---
        if is_dann:
            train_subj = subj[trm]
            dom_map = {oid.item(): nid for nid, oid in enumerate(train_subj.unique().sort()[0])}
            dt = torch.tensor([dom_map[oid.item()] for oid in train_subj], device=device)
            n_dom = len(dom_map)
        # --- build model ---
        model = model_fn(n_dom=n_dom).to(device) if is_dann else model_fn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
        for ep in range(epochs):
            model.train()
            # DANN lambda schedule: ramp from 0→1 over first half of epochs
            dann_lam = min(1.0, 2.0 * ep / max(1, epochs - 1)) if is_dann else 0
            for i in range(0, Xt.size(0), bs):
                xb, yb = Xt[i:i + bs], yt[i:i + bs]
                if is_ours:
                    yh, _, _, mu, lv, _ = model(xb)
                    kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(-1).mean()
                    loss = 0.9 * F.cross_entropy(yh, yb, label_smoothing=0.1) + 0.01 * kl
                elif is_dann:
                    db = dt[i:i + bs]
                    yh, dom_pred = model(xb, dom=db, lam=dann_lam)
                    loss = F.cross_entropy(yh, yb, label_smoothing=0.1) \
                         + 0.1 * F.cross_entropy(dom_pred, db)
                else:
                    loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            if is_ours:
                pred = model(Xe)[0].argmax(1)
            elif is_dann:
                pred = model(Xe).argmax(1)  # eval mode: no dom → task_clf only
            else:
                pred = model(Xe).argmax(1)
            accs.append((pred == ye).float().mean().item())
            yt_all.append(ye.cpu()); yp_all.append(pred.cpu())
        print(f"  [{s + 1}/{Ns}] {accs[-1]:.4f}")
    yt_all = torch.cat(yt_all); yp_all = torch.cat(yp_all)
    from sklearn.metrics import f1_score, cohen_kappa_score
    return np.mean(accs), np.std(accs), np.min(accs), f1_score(yt_all, yp_all, average='macro'), cohen_kappa_score(yt_all, yp_all)

# ============================================================
print("\n" + "=" * 60)
print("REAL EXPERIMENT — DREAMER (Water Cycle + Mutual Neurons, Fast=Off)")
print("=" * 60)
X, y_val, y_aro, subj = load_dreamer()
C, T_actual = X.shape[1], X.shape[2]
print(f"Data shape: {X.shape[0]} samples, {C}ch, {T_actual}tp")
results = {}; timing = {}
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dreamer_loso.json")
if os.path.exists(SAVE_PATH):
    saved = json.load(open(SAVE_PATH))
    for k, v in saved.items():
        if isinstance(v, list) and len(v) == 5:
            results[k] = tuple(v)
            timing[k] = 0.0
    print(f"[Resume] Loaded {len(results)} completed results from {SAVE_PATH}")

def save_results():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    json.dump({k: list(v) for k, v in results.items()}, open(SAVE_PATH, "w"), indent=2)

for task_name, y_target in [("Arousal", y_aro), ("Valence", y_val)]:
    print(f"\n{'=' * 60}\nTASK: {task_name} ({X.shape[0]} samples)\n{'=' * 60}")
    for mname, mfn, is_ours, is_dann in [
        ("Ours",       lambda: FullModel(C, T_actual, N=32), True,  False),
        ("EEGNet",     lambda: EEGNet(C, T_actual),          False, False),
        ("DeepConvNet", lambda: DeepConvNet(C, T_actual),     False, False),
        ("DANN",       lambda n_dom=22: DANN(C, T_actual, n_dom=n_dom), False, True),
    ]:
        key = f'{mname}_{task_name}'
        if key in results:
            mu, std, mi, f1, kp = results[key]
            print(f"\n[{mname}] (cached) {mu * 100:.1f}+/-{std * 100:.1f}%")
            continue
        print(f"\n[{mname}]")
        t0 = time()
        mu, std, mi, f1, kp = loso_eval(mfn, X, y_target, subj, epochs=10, is_ours=is_ours, is_dann=is_dann)
        results[key] = (mu, std, mi, f1, kp)
        timing[key] = (time() - t0) / 60
        save_results()
        print(f"  {mu * 100:.1f}+/-{std * 100:.1f}% | F1={f1:.3f} | Kappa={kp:.3f} | Worst={mi * 100:.1f}% | {timing[key]:.0f}min")

print(f"\n{'=' * 70}")
print(f"FINAL RESULTS — DREAMER (Water Cycle + Mutual Neurons)")
print(f"{'=' * 70}")
print(f"{'Method':<14} {'Task':<9} {'Acc(%)':<12} {'F1':<7} {'Kappa':<7} {'Worst(%)':<10} {'Time':<8}")
print(f"{'-' * 70}")
for task in ['Arousal', 'Valence']:
    for m in ['Ours', 'EEGNet', 'DeepConvNet', 'DANN']:
        k = f'{m}_{task}'
        if k in results:
            mu, std, mi, f1, kp = results[k]
            print(f"{m:<14} {task:<9} {mu * 100:>6.2f}+/-{std * 100:.1f}  {f1:>5.3f}  {kp:>5.3f}  {mi * 100:>6.1f}     {timing[k]:>5.0f}m")
    if task == 'Arousal':
        oa = results.get('Ours_Arousal', (0,))[0]
        ea = results.get('EEGNet_Arousal', (0,))[0]
        da = results.get('DeepConvNet_Arousal', (0,))[0]
        dna = results.get('DANN_Arousal', (0,))[0]
        print(f"{'':->70}")
        parts = []
        if 'Ours_Arousal' in results and 'EEGNet_Arousal' in results:
            parts.append(f"Ours-EEGNet: {oa - ea:+.1%}")
        if 'DeepConvNet_Arousal' in results:
            parts.append(f"Ours-DeepConv: {oa - da:+.1%}")
        if 'DANN_Arousal' in results:
            parts.append(f"Ours-DANN: {oa - dna:+.1%}")
        if parts: print(f"  {' | '.join(parts)}")
print(f"\nTotal: {sum(timing.values()):.0f} min (~{sum(timing.values()) / 60:.1f}h)")
print("NO fast pathway. Phase 2: add fast pathway back after adversarial pretraining.")
