#!/usr/bin/env python3
import os, json, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============ CONFIG ============
MODES_PATH      = "modes/U_modes_tol_1e-05.npy"      # pick one saved POD basis
DATA_DIR        = "../FEM/fem_training_data"
PREFIX          = "fem_simulation_"
TIME_NORM       = "index"  # "index" or "vector" (if you have a per-sim t vector)
T_VECTOR_PATH   = None

# Train/val split by FILES (grouped to avoid leakage across time of same μ)
VAL_FRACTION    = 0.2
RANDOM_SEED     = 42

# MLP
HIDDEN_SIZES    = [32, 64, 128]
ACTIVATION      = "elu"   # "relu", "gelu", "elu"
DROPOUT         = 0.0

# Optimization
EPOCHS          = 1000
BATCH_SIZE      = 64
LR              = 1e-3
WEIGHT_DECAY    = 1e-6
PATIENCE        = 20       # early stopping

OUT_DIR         = "ann_models"
# ================================

MU_RE = re.compile(r"mu1_([0-9]+(?:\.[0-9]+)?)_mu2_([0-9]+(?:\.[0-9]+)?)")

def parse_mus(fname: str):
    m = MU_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse mu1/mu2 from '{fname}'. Expected ..._mu1_<float>_mu2_<float>...")
    return float(m.group(1)), float(m.group(2))

def load_modes(path):
    U = np.load(path)  # [M, n]
    return U

def build_Z_Q(modes_path, data_dir, prefix, time_norm, t_vector_path):
    U_modes = load_modes(modes_path)               # [M, n]
    M, n = U_modes.shape

    # keep file order deterministic (match POD stacking if available)
    stack_order = "modes/stack_order.json"
    if os.path.exists(stack_order):
        files = json.load(open(stack_order))["files"]
    else:
        files = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith(".npy")])

    # optional global time vector (if all sims share it)
    t_norm = None
    if time_norm == "vector":
        if not (t_vector_path and os.path.exists(t_vector_path)):
            raise ValueError("TIME_NORM='vector' requires valid T_VECTOR_PATH")
        t = np.load(t_vector_path).ravel()
        t = t - t[0]
        T = t[-1] if t[-1] != 0 else 1.0
        t_norm = t / T

    Z_per_file, Q_per_file = [], []
    for fname in files:
        mu1, mu2 = parse_mus(fname)
        S = np.load(os.path.join(data_dir, fname))  # [M, Nt]
        assert S.ndim == 2 and S.shape[0] == M, f"{fname}: expected [M,Nt], got {S.shape}"
        Nt = S.shape[1]
        tau = (np.linspace(0,1,Nt) if time_norm == "index" else t_norm)
        assert len(tau) == Nt, f"{fname}: tau len {len(tau)} != Nt {Nt}"

        Z_traj = np.column_stack([np.full(Nt, mu1), np.full(Nt, mu2), tau])  # [Nt,3]
        Q_traj = (S.T @ U_modes)                                             # [Nt,n]

        Z_per_file.append(Z_traj)
        Q_per_file.append(Q_traj)

    return U_modes, files, Z_per_file, Q_per_file

def split_by_files(files, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(len(files)*val_fraction)) if len(files) > 1 else 0
    val_idx = set(idx[:n_val])
    train_files = [files[i] for i in range(len(files)) if i not in val_idx]
    val_files   = [files[i] for i in range(len(files)) if i in val_idx]
    return train_files, val_files, val_idx

def stack_selected(files, Z_per_file, Q_per_file, selected):
    Z_list, Q_list = [], []
    for i, (Z, Q) in enumerate(zip(Z_per_file, Q_per_file)):
        if i in selected:
            Z_list.append(Z); Q_list.append(Q)
    return (np.vstack(Z_list) if Z_list else np.empty((0,3))), (np.vstack(Q_list) if Q_list else np.empty((0,Q_per_file[0].shape[1])))

class NumpyScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_  = X.std(axis=0, keepdims=True)
        self.std_[self.std_==0] = 1.0
        return self
    def transform(self, X): return (X - self.mean_) / self.std_
    def inverse_transform(self, Xs): return Xs * self.std_ + self.mean_

def make_mlp(in_dim, out_dim, hidden, act, dropout):
    if act == "relu": A = nn.ReLU
    elif act == "gelu": A = nn.GELU
    else: A = nn.ELU
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), A()]
        if dropout>0: layers += [nn.Dropout(dropout)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class ZQDataset(Dataset):
    def __init__(self, Zs, Q): self.Zs = torch.from_numpy(Zs).float(); self.Q = torch.from_numpy(Q).float()
    def __len__(self): return self.Zs.shape[0]
    def __getitem__(self, i): return self.Zs[i], self.Q[i]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    U_modes, files, Z_per_file, Q_per_file = build_Z_Q(MODES_PATH, DATA_DIR, PREFIX, TIME_NORM, T_VECTOR_PATH)
    M, n = U_modes.shape

    # split by files (grouped)
    train_files, val_files, val_idx = split_by_files(files, VAL_FRACTION, RANDOM_SEED)
    train_idx = set(i for i in range(len(files)) if i not in val_idx)

    Z_tr, Q_tr = stack_selected(files, Z_per_file, Q_per_file, train_idx)
    Z_va, Q_va = stack_selected(files, Z_per_file, Q_per_file, val_idx)
    print(f"Train samples: {Z_tr.shape[0]} | Val samples: {Z_va.shape[0]} | n_modes={n}")

    # scale inputs only
    scaler = NumpyScaler().fit(Z_tr)
    Zs_tr = scaler.transform(Z_tr); Zs_va = scaler.transform(Z_va)

    train_ds = ZQDataset(Zs_tr, Q_tr)
    val_ds   = ZQDataset(Zs_va, Q_va)
    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # model, loss, opt
    model = make_mlp(in_dim=3, out_dim=n, hidden=HIDDEN_SIZES, act=ACTIVATION, dropout=DROPOUT)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # (optional) SV weighting to balance modes by energy
    # w = 1.0 / (np.arange(1, n+1))**0  # =1 => plain MSE
    w = torch.ones(n, device=device)

    criterion = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf"); patience = PATIENCE; best_state = None
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for Zb, Qb in tr_loader:
            Zb, Qb = Zb.to(device), Qb.to(device)
            optim.zero_grad()
            Qp = model(Zb)
            loss = criterion(Qp, Qb)
            loss.backward()
            optim.step()
            tr_loss += loss.item()*Zb.size(0)
        tr_loss /= max(1,len(train_ds))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Zb, Qb in va_loader:
                Zb, Qb = Zb.to(device), Qb.to(device)
                Qp = model(Zb); va_loss += criterion(Qp, Qb).item()*Zb.size(0)
        va_loss /= max(1,len(val_ds))

        if va_loss < best_val - 1e-6:
            best_val = va_loss; patience = PATIENCE; best_state = model.state_dict()
        else:
            patience -= 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:04d} | train {tr_loss:.3e} | val {va_loss:.3e} | best {best_val:.3e} | patience {patience}")
        if patience <= 0:
            print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # save artifacts
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "ann_model.pt"))
    np.save(os.path.join(OUT_DIR, "U_modes.npy"), U_modes)
    np.savez(os.path.join(OUT_DIR, "scaler_z.npz"), mean=scaler.mean_, std=scaler.std_)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({
            "modes_path": MODES_PATH,
            "hidden": HIDDEN_SIZES,
            "activation": ACTIVATION,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "val_fraction": VAL_FRACTION,
            "time_norm": TIME_NORM
        }, f, indent=2)

    print(f"[OK] Saved model to {OUT_DIR}/ann_model.pt")

if __name__ == "__main__":
    main()
