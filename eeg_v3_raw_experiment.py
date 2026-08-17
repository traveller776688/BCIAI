"""EEG V3-RAW: DAME on SEED-IV raw waveforms.

Motivation: the DE-feature protocol has a ~40% ceiling (recent protocol-evaluation
literature: DGCNN ≈ 39.5% unseen-subject on DE features), leaving no headroom for
DAME mechanisms. Raw signal restores the encoding stage — WaterCycle/MutualSociety
must now extract their own representations, giving the mechanisms real work.

Design: same DAME components (WaterCycleV2 / MutualSocietyV2 / PredictionHeadV2)
and same ablation matrix as eeg_v3_experiment.py — the ONLY change is the input:
raw 800Hz EEG (decimated to 200Hz) → RawEncoder conv stem → H_seq.

Ablation matrix (unchanged):
  DAME-EEG-Raw:  water=1 reflux=1 mutual=1  (full)
  DAME-NoReflux: water=1 reflux=0 mutual=1
  DAME-NoMutual: water=1 reflux=1 mutual=0
  DAME-NoWater:  water=0 reflux=0 mutual=1
  DAME-Base:     water=0 reflux=0 mutual=0  (encoder + classifier)

Usage:
  python -u eeg_v3_raw_experiment.py --quick    # 3 subj × 1 seed × 3 ep, sanity
  python -u eeg_v3_raw_experiment.py --level2   # 5 subj × 5 seeds × 8 ep (default)
  python -u eeg_v3_raw_experiment.py --level2 --model DAME-EEG-Raw
"""
import os
import sys
import json
import time
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat
from scipy.signal import decimate
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eeg_v3_experiment import (
    WaterCycleV2, MutualSocietyV2, PredictionHeadV2,
    D_MODEL, K_LATENT, N_NEURONS, N_COMMUNITIES, D_MEM,
    KL_W, ORTHO_W, MUTUAL_W, SPEC_W, REFLUX_W, GATE_ENTROPY_W, PRED_W,
    TEMP_INIT, TEMP_FINAL, TEMP_ANNEAL_EPOCHS, KL_WARMUP_EPOCHS,
    LR, N_CLASSES, SESSION_LABELS,
    train_epoch_dame, evaluate,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================
# CONFIG — raw-signal protocol
# =========================================================================
SEED_DIR = r"C:/Users/LENOVO/Desktop/BCIAI/SEED_IV"
RAW_DIR = os.path.join(SEED_DIR, "eeg_raw_data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FS_RAW = 800         # SEED-IV raw sampling rate
FS_TARGET = 200      # decimated rate (standard for EEG emotion)
WINDOW_SAMPLES = 4 * FS_TARGET    # 800 = 4s window (SEED convention)
WINDOW_STRIDE = 2 * FS_TARGET     # 400 = 2s hop (50% overlap)

BATCH_SIZE = 64
QUICK_SUBJECTS = 3
QUICK_EPOCHS = 3
LEVEL2_SUBJECTS = 5
LEVEL2_EPOCHS = 12   # raw encoder trains from scratch → more epochs than DE features
ALL_SEEDS = [42, 123, 789, 456, 999]


# =========================================================================
# 1. Raw Data Loader
# =========================================================================

def load_seed_iv_raw(n_subjects=None):
    """Load SEED-IV raw EEG: 800Hz → decimate 4× → 200Hz → 4s windows (2s hop).

    Per-subject z-score normalization. Labels from ReadMe per-session arrays
    (fixed for all subjects). Returns X float16 (N, 62, 800), y, subj.
    """
    subjects = sorted({int(f.split('_')[0])
                       for s in [1, 2, 3]
                       for f in os.listdir(os.path.join(RAW_DIR, str(s)))
                       if f.endswith('.mat')})
    if n_subjects:
        subjects = subjects[:n_subjects]

    all_X, all_y, all_subj = [], [], []
    t0 = time.time()

    for subj_id in subjects:
        trials, labels = [], []
        for session in [1, 2, 3]:
            sess_dir = os.path.join(RAW_DIR, str(session))
            fname = next((f for f in os.listdir(sess_dir)
                          if f.startswith(f"{subj_id}_") and f.endswith('.mat')), None)
            if fname is None:
                continue
            data = loadmat(os.path.join(sess_dir, fname))
            sess_labels = SESSION_LABELS[session]
            # Key prefix varies per subject (tyc_eegN / cz_eegN / ...) — match by suffix
            for t in range(24):
                var = next((k for k in data
                            if not k.startswith('__') and k.endswith(f'_eeg{t + 1}')), None)
                if var is None:
                    continue
                raw = np.nan_to_num(data[var]).astype(np.float32)  # (62, T@800)
                ds = decimate(raw, 4, axis=1)                      # (62, T@200)
                trials.append(ds)
                labels.append(sess_labels[t])

        if not trials:
            continue

        # Per-subject z-score over all samples
        cat = np.concatenate(trials, axis=1)
        mu = cat.mean(axis=1, keepdims=True)
        sd = cat.std(axis=1, keepdims=True) + 1e-8
        del cat

        for ds, lab in zip(trials, labels):
            ds = (ds - mu) / sd
            T = ds.shape[1]
            for st in range(0, T - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
                all_X.append(torch.from_numpy(ds[:, st:st + WINDOW_SAMPLES].copy()))
                all_y.append(lab)
                all_subj.append(subj_id - 1)

        print(f"  [Data] subject {subj_id}: {len(all_X)} windows so far "
              f"({time.time() - t0:.0f}s)", flush=True)

    X = torch.stack(all_X)                                  # (N, 62, 800)
    y = torch.tensor(all_y, dtype=torch.long)
    subj = torch.tensor(all_subj, dtype=torch.long)

    print(f"[Data] RAW SEED-IV: {X.shape[0]} windows | {len(subjects)} subjects | "
          f"{WINDOW_SAMPLES} samples ({WINDOW_SAMPLES // FS_TARGET}s) | "
          f"classes={dict(Counter(all_y))} | prep {time.time() - t0:.0f}s", flush=True)
    return X, y, subj


# =========================================================================
# 2. Raw Encoder + DAME-EEG-Raw (same components, new input stage)
# =========================================================================

class RawEncoder(nn.Module):
    """Raw EEG conv stem @200Hz: (B, 62, 800) → (B, 25, D)."""

    def __init__(self, C=62, T=WINDOW_SAMPLES, D=D_MODEL):
        super().__init__()
        # 800 → 100 (k=51, s=8)
        self.s1 = nn.Sequential(
            nn.Conv1d(C, 128, 51, stride=8, padding=25, bias=False),
            nn.BatchNorm1d(128), nn.GELU())
        # 100 → 50
        self.s2 = nn.Sequential(
            nn.Conv1d(128, 192, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(192), nn.GELU())
        # 50 → 25
        self.s3 = nn.Sequential(
            nn.Conv1d(192, D, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(D), nn.GELU())

    def forward(self, X):  # (B, C, T)
        h = self.s1(X)
        h = self.s2(h)
        h = self.s3(h)
        return h.transpose(1, 2)  # (B, T'=25, D)


class DAME_EEG_Raw(nn.Module):
    """DAME-EEG on raw waveforms: RawEncoder → [WaterCycleV2] → [MutualSocietyV2] → Classifier.

    Same component flags / ablation matrix / loss terms as DAME_EEG (DE version).
    Interface matches train_epoch_dame / evaluate from eeg_v3_experiment.
    """

    def __init__(self, C=62, T=WINDOW_SAMPLES, nc=N_CLASSES,
                 use_water=True, use_reflux=True, use_mutual=True):
        super().__init__()
        self.use_water = use_water
        self.use_reflux = use_reflux and use_water
        self.use_mutual = use_mutual

        # P1: raw signal → temporal representation
        self.encoder = RawEncoder(C, T, D_MODEL)

        # P2+P5: water cycle
        if use_water:
            self.water = WaterCycleV2(D_MODEL, K_LATENT, use_reflux=self.use_reflux)
        else:
            self.water = None

        # P3: mutual society
        if use_mutual:
            self.mutual = MutualSocietyV2(N_NEURONS, D_MODEL, D_MEM,
                                           n_communities=N_COMMUNITIES)
        else:
            self.mutual = None

        # P4: pre-migration (only with reflux)
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
        self._total_epochs = 1

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
        """X: (B, C, T) raw EEG"""
        # P1: encoder → H_seq (temporal representation)
        H_seq = self.encoder(X)
        H_pooled = H_seq.mean(dim=1)  # (B, D) — global state

        if self.use_water:
            # P2+P5: water cycle — VIB → CrossAttn → [Reflux]
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
            # P3: mutual society on anchor A + raw features H_pooled
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
            "reflux_mag": reflux_mag if isinstance(reflux_mag, torch.Tensor)
            else torch.tensor(reflux_mag, device=X.device),
        }

    def compute_loss(self, out, labels, subject_ids=None, Z_next=None, Z_next2=None):
        """Same loss structure as DAME_EEG: CE + component-gated regularization."""
        loss = F.cross_entropy(out["logits"], labels, label_smoothing=0.1)

        # P5: KL warmup (only with WaterCycle)
        if self.use_water:
            warmup = min(1.0, (self._current_epoch + 1) / max(KL_WARMUP_EPOCHS, 1))
            kl_val = out["kl_loss"]
            if isinstance(kl_val, torch.Tensor):
                loss = loss + KL_W * warmup * kl_val

        # P3: mutual constraints
        if self.use_mutual:
            gates = out["gates"]
            loss = loss + ORTHO_W * self.mutual.ortho_loss()
            loss = loss + MUTUAL_W * self.mutual.mutual_loss()
            loss = loss + SPEC_W * self.mutual.specialization_loss(gates)
            loss = loss + GATE_ENTROPY_W * self.mutual.gate_entropy_loss(gates)

        # P4: pre-migration (only when reflux is active)
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

            # P2: encourage meaningful reflux displacement
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


# =========================================================================
# 3. LOSO evaluation (raw)
# =========================================================================

MODEL_SPECS = {
    "DAME-EEG-Raw": (True, True, True),
    "DAME-NoReflux": (True, False, True),
    "DAME-NoMutual": (True, True, False),
    "DAME-NoWater": (False, False, True),
    "DAME-Base": (False, False, False),
}


def loso_raw(X, y, subj, spec, epochs, seed, log=None, done_folds=None, save_fn=None):
    """LOSO with one DAME variant. Returns (mean_acc, std_acc, fold_accs).

    done_folds: {fold_idx: acc} already computed (resume support) — skipped.
    save_fn: called with fold_accs after each fold for incremental checkpointing.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_water, use_reflux, use_mutual = spec

    n_subj = int(subj.max().item()) + 1
    accs = []
    t_start = time.time()

    for s in range(n_subj):
        if done_folds and s in done_folds:
            accs.append(done_folds[s])
            print(f"  [{s + 1}/{n_subj}] acc={done_folds[s]:.4f} (resumed)", flush=True)
            if log:
                log.write(f"  [{s + 1}/{n_subj}] acc={done_folds[s]:.4f} (resumed)\n")
                log.flush()
            continue
        te = subj == s
        tr = ~te

        model = DAME_EEG_Raw(use_water=use_water, use_reflux=use_reflux,
                             use_mutual=use_mutual).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=2)

        tr_ds = torch.utils.data.TensorDataset(X[tr], y[tr], subj[tr])
        tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True)
        te_ds = torch.utils.data.TensorDataset(X[te], y[te], torch.full_like(y[te], s))
        te_dl = DataLoader(te_ds, BATCH_SIZE * 2, shuffle=False)

        for ep in range(epochs):
            model.set_epoch(ep, epochs)
            tl, tstats = train_epoch_dame(model, tr_dl, opt)
            sched.step()
            if ep == epochs - 1 or ep == 0 or ep == epochs // 2:
                print(f"    ep{ep + 1}/{epochs}: loss={tl:.3f} kl={tstats['kl']:.3f} "
                      f"active={tstats['n_active']:.1f}", flush=True)

        acc, per_class, preds, labels = evaluate(model, te_dl, is_dame=True)
        accs.append(acc)
        if save_fn:
            save_fn(accs)

        line = (f"  [{s + 1}/{n_subj}] acc={acc:.4f} "
                f"per_class={ {k: f'{v:.3f}' for k, v in per_class.items()} } "
                f"({time.time() - t_start:.0f}s)")
        print(line, flush=True)
        if log:
            log.write(line + "\n")
            log.flush()

    return float(np.mean(accs)), float(np.std(accs)), accs


# =========================================================================
# 4. Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="EEG V3-RAW: DAME on SEED-IV raw waveforms")
    parser.add_argument("--quick", action="store_true", help="3 subj × 1 seed × 3 epochs")
    parser.add_argument("--level2", action="store_true", help="5 subj × 5 seeds × 12 epochs")
    parser.add_argument("--model", type=str, default=None, help="Run single model only")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--subjects", type=int, default=None, help="Subject count override")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds (first N of ALL_SEEDS)")
    parser.add_argument("--seed-offset", type=int, default=0, help="Skip first K seeds of ALL_SEEDS")
    args = parser.parse_args()

    if args.quick:
        n_subjects, epochs, seeds = QUICK_SUBJECTS, QUICK_EPOCHS, [42]
        tag = "quick"
    else:
        n_subjects, epochs, seeds = LEVEL2_SUBJECTS, LEVEL2_EPOCHS, ALL_SEEDS
        tag = "level2"
    if args.epochs:
        epochs = args.epochs
    if args.subjects:
        n_subjects = args.subjects
    if args.seeds:
        seeds = ALL_SEEDS[:args.seeds]
    if args.seed_offset:
        seeds = ALL_SEEDS[args.seed_offset:args.seed_offset + (args.seeds or len(ALL_SEEDS))]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"raw_{tag}_run.log")
    log = open(log_path, "w", encoding="utf-8")

    print("=" * 74)
    print("  EEG V3-RAW — DAME on SEED-IV raw waveforms (800→200Hz, 4s windows)")
    print(f"  {len(seeds)} seed(s) × {n_subjects} subjects LOSO × {epochs} epochs × "
          f"{len(MODEL_SPECS)} DAME variants")
    print(f"  D={D_MODEL} K={K_LATENT} N={N_NEURONS} C={N_COMMUNITIES} | device={DEVICE}")
    print("=" * 74)
    print(f"[1/3] Loading raw data ({n_subjects} subjects)...", flush=True)
    X, y, subj = load_seed_iv_raw(n_subjects)

    models_to_run = [args.model] if args.model else list(MODEL_SPECS.keys())

    # Resume support: load partial results from a previous killed run
    out_path = os.path.join(RESULTS_DIR, f"eeg_v3_raw_{tag}.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
        if "results" in prev:
            results = prev["results"]
            print(f"[resume] loaded partial results: "
                  f"{ {m: list(prev['results'][m].keys()) for m in prev['results']} }",
                  flush=True)

    n_subj_total = int(subj.max().item()) + 1

    def save_now():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"config": {"n_subjects": n_subjects, "epochs": epochs,
                                  "seeds": seeds, "window_samples": WINDOW_SAMPLES,
                                  "stride": WINDOW_STRIDE, "fs": FS_TARGET},
                       "results": results}, f, indent=2)

    print(f"\n[2/3] Running LOSO experiments...", flush=True)
    for name in models_to_run:
        spec = MODEL_SPECS[name]
        print(f"\n--- {name} (water={spec[0]}, reflux={spec[1]}, mutual={spec[2]}) ---",
              flush=True)
        log.write(f"\n--- {name} ---\n")
        results.setdefault(name, {})
        for seed in seeds:
            done_folds = {}
            prev_entry = results[name].get(str(seed))
            if prev_entry and len(prev_entry.get("folds", [])) >= n_subj_total:
                print(f"  seed={seed} already complete — skip", flush=True)
                log.write(f"  seed={seed} already complete — skip\n")
                continue
            if prev_entry:
                done_folds = {i: a for i, a in enumerate(prev_entry["folds"])}
                print(f"  seed={seed} (resuming {len(done_folds)}/{n_subj_total} folds)",
                      flush=True)
            else:
                print(f"  seed={seed}", flush=True)
            log.write(f"  seed={seed}\n")

            def save_fn(fold_accs, _name=name, _seed=seed):
                mean = float(np.mean(fold_accs))
                std = float(np.std(fold_accs))
                results[_name][str(_seed)] = {"mean": mean, "std": std,
                                              "folds": [float(a) for a in fold_accs]}
                save_now()

            mean_acc, std_acc, fold_accs = loso_raw(
                X, y, subj, spec, epochs, seed, log, done_folds, save_fn)
            results[name][str(seed)] = {"mean": mean_acc, "std": std_acc,
                                        "folds": [float(a) for a in fold_accs]}
            save_now()
            print(f"  → {name} seed={seed}: {mean_acc:.4f} ± {std_acc:.4f}", flush=True)
            log.write(f"  → {mean_acc:.4f} ± {std_acc:.4f}\n")

    # ---- Statistical analysis over seeds ----
    print("\n[3/3] Statistical analysis (paired t-test over seeds)...", flush=True)
    full_name = "DAME-EEG-Raw"
    summary = {}
    for name, per_seed in results.items():
        accs = [v["mean"] for v in per_seed.values()]
        arr = np.array(accs)
        mean, std = arr.mean(), arr.std(ddof=1)
        ci = 2.776 * std / np.sqrt(len(arr))  # t(0.025, 4) = 2.776
        summary[name] = {"mean": mean, "std": std, "ci95": ci, "accs": accs}
        print(f"  {name:16s} {mean:.4f} ± {std:.4f}  (95% CI ±{ci:.4f})", flush=True)

    if full_name in summary and len(summary) > 1:
        print("\n  ---- Ablation deltas vs DAME-EEG-Raw ----", flush=True)
        full_accs = summary[full_name]["accs"]
        for name, s in summary.items():
            if name == full_name:
                continue
            t, p = sps.ttest_rel(full_accs, s["accs"])
            delta = s["mean"] - summary[full_name]["mean"]
            sig = "★" if p < 0.05 else "(n.s.)"
            print(f"  {name:16s} {delta:+.4f}  t={t:.2f}  p={p:.4f}  {sig}", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"n_subjects": n_subjects, "epochs": epochs,
                              "seeds": seeds, "window_samples": WINDOW_SAMPLES,
                              "stride": WINDOW_STRIDE, "fs": FS_TARGET},
                   "results": results, "summary": summary}, f, indent=2)

    print(f"\nDONE. Results: {out_path}", flush=True)
    log.close()


if __name__ == "__main__":
    main()
