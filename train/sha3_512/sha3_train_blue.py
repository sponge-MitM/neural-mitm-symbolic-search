'Module object.'
import os
import sys
import time
import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


def pairwise_ranking_loss(pred, target, margin=2.0):
    'Function pairwise ranking loss.'
    n = pred.shape[0]
    half = max(1, n // 2)
    idx = torch.randperm(n, device=pred.device)[:half]
    jdx = torch.randperm(n, device=pred.device)[:half]
    pi, pj = pred[idx], pred[jdx]
    ti, tj = target[idx], target[jdx]
    valid = ti < tj
    if valid.sum() == 0:
        return torch.zeros((), device=pred.device)
    pi, pj = pi[valid], pj[valid]
    return torch.clamp(margin - (pj - pi), min=0).mean()


_current_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_current_dir))
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'sha3_512')
RESULT_DIR = PROCESSED_DIR


class BlueMLP(nn.Module):
    'Class BlueMLP.'
    def __init__(self, input_dim, hidden_dims=(256, 128, 64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.reg = nn.Sequential(nn.Linear(prev_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.rank = nn.Sequential(nn.Linear(prev_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.encoder(x)
        return {'reg': self.reg(h).squeeze(-1), 'rank': self.rank(h).squeeze(-1)}


def find_data(base_name):
    'Function find data.'
    for d in (PROCESSED_DIR, RESULT_DIR):
        feats = os.path.join(d, f'{base_name}_features.npy')
        labels = os.path.join(d, f'{base_name}_labels.npy')
        if os.path.exists(feats) and os.path.exists(labels):
            return feats, labels, d
    raise FileNotFoundError(f"Not found {base_name}_features.npy / _labels.npy "
                            f"in {PROCESSED_DIR} / {RESULT_DIR}")


def stratified_split(labels, n_bins=10, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    quantiles = np.percentile(labels, np.linspace(0, 100, n_bins + 1))
    bin_ids = np.clip(np.digitize(labels, quantiles[:-1]) - 1, 0, n_bins - 1)
    train_idx, val_idx, test_idx = [], [], []
    for b in range(n_bins):
        bin_idx = np.where(bin_ids == b)[0]
        rng.shuffle(bin_idx)
        nb = len(bin_idx)
        ntr, nva = int(nb * train_ratio), int(nb * val_ratio)
        train_idx.append(bin_idx[:ntr])
        val_idx.append(bin_idx[ntr:ntr + nva])
        test_idx.append(bin_idx[ntr + nva:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def main():
    ap = argparse.ArgumentParser(description='SHA3-512 blue->min_u2 neural network training')
    ap.add_argument('--data-dir', type=str, default=None, help='feature file directory')
    ap.add_argument('--base-name', type=str, default=None,
                    help='feature file name prefix (default: latest sha3_512_blue_labels_*)')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--ranking-weight', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()


    if args.base_name is None:
        candidates = []
        for d in (PROCESSED_DIR, RESULT_DIR):
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.startswith('sha3_512_blue_labels_') and f.endswith('_features.npy'):
                    candidates.append(os.path.join(d, f))
        if not candidates:
            raise FileNotFoundError("sha3_512_blue_labels_*_features.npy not found; "
                                    "run sha3_blue_label.py + sha3_blue_features.py first")
        candidates.sort(key=os.path.getmtime)
        args.base_name = os.path.basename(candidates[-1]).replace('_features.npy', '')
        args.data_dir = os.path.dirname(candidates[-1])
        print(f"Auto-selected: {args.base_name} @ {args.data_dir}")

    feats_path, labels_path, found_dir = find_data(args.base_name)
    if args.data_dir is None:
        args.data_dir = found_dir

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    X_raw = np.load(feats_path).astype(np.float32)
    y = np.load(labels_path).astype(np.float32)
    print(f"Data: X{X_raw.shape}, y[{y.min():.0f}..{y.max():.0f}], "
          f"mean={y.mean():.1f}, std={y.std():.1f}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    y_mean, y_std = y.mean(), y.std()
    y_norm = ((y - y_mean) / y_std).astype(np.float32)

    tr, va, te = stratified_split(y, seed=args.seed)
    print(f"Splits: train={len(tr)}, val={len(va)}, test={len(te)}")

    tr_ds = TensorDataset(torch.FloatTensor(X[tr]), torch.FloatTensor(y_norm[tr]))
    va_ds = TensorDataset(torch.FloatTensor(X[va]), torch.FloatTensor(y_norm[va]))
    te_ds = TensorDataset(torch.FloatTensor(X[te]), torch.FloatTensor(y_norm[te]))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = BlueMLP(input_dim=X.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BlueMLP {X.shape[1]}->256-128-64-32 (reg+rank), params {n_params:,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_loss, best_state, counter = float('inf'), None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x_b, y_b in tr_ld:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt.zero_grad()
            out = model(x_b)
            loss = criterion(out['reg'], y_b)
            if args.ranking_weight > 0:
                loss = loss + args.ranking_weight * pairwise_ranking_loss(out['rank'], y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for x_b, y_b in va_ld:
                out = model(x_b.to(device))['reg']
                va_loss += criterion(out, y_b.to(device)).item() * len(x_b)
                va_n += len(x_b)
        va_loss /= va_n

        if va_loss < best_loss - 1e-5:
            best_loss, counter, best_state = va_loss, 0, copy.deepcopy(model.state_dict())
        else:
            counter += 1
        if ep % 20 == 19 or ep == 0:
            print(f"  Ep {ep+1:3d}: va_loss={va_loss:.4f} best={best_loss:.4f} "
                  f"({time.time()-t0:.0f}s)")
        if counter >= args.patience:
            print(f"  Early stop @ epoch {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x_b, y_b in te_ld:
            p = model(x_b.to(device))['reg'].cpu().numpy() * y_std + y_mean
            all_pred.append(p)
            all_true.append(y_b.numpy() * y_std + y_mean)
    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    spear, _ = spearmanr(preds, trues)

    print(f"\nTest: MAE={mae:.3f}, RMSE={rmse:.3f}, Spearman={spear:.4f}")

    save_path = os.path.join(PROCESSED_DIR, 'model_blue_mlp.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'y_mean': float(y_mean),
        'y_std': float(y_std),
        'scalar_dim': int(X.shape[1]),
        'test_mae': mae,
        'test_rmse': rmse,
        'test_spearman': spear,
        'n_params': n_params,
        'source': args.base_name,
    }, save_path)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
