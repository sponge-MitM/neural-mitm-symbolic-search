'Module object.'
import os, sys, time, copy, json, argparse, gc
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def pairwise_ranking_loss(pred, target, margin=2.0):
    """Pairwise margin ranking loss: for random pairs (i,j) with target_i < target_j,
    penalize pred_i >= pred_j by margin - (pred_j - pred_i)."""
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

from scipy.stats import spearmanr

_model_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _model_dir)
from model import CyclicConv3d

REPO_ROOT = os.path.dirname(os.path.dirname(_model_dir))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'sha3_384')
DATA_PATH = os.path.join(DATA_DIR, 'sha3_384_train_stageB_pi1.npz')

def _resolve_data_path(arg):
    return arg or DATA_PATH

SEED = 42
FEAT_DIM = 3840


class MLP_stageB(nn.Module):
    'Class MLP stageB.'
    def __init__(self, input_dim=3840, hidden=(512,256,128), dropout=0.3):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.reg = nn.Sequential(nn.Linear(prev,32), nn.ReLU(), nn.Linear(32,1))
        self.rank = nn.Sequential(nn.Linear(prev,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x, _=None):
        h = self.encoder(x)
        return {'reg': self.reg(h).squeeze(-1), 'rank': self.rank(h).squeeze(-1)}


class CNN_stageB(nn.Module):
    'Class CNN stageB.'
    def __init__(self):
        super().__init__()
        self.conv1 = CyclicConv3d(2, 16, kernel_z=3, kernel_x=3, kernel_y=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = CyclicConv3d(16, 32, kernel_z=3, kernel_x=3, kernel_y=2)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = CyclicConv3d(32, 64, kernel_z=3, kernel_x=3, kernel_y=2)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.AdaptiveAvgPool3d((1, 16, 2))
        self.scalar_enc = nn.Sequential(nn.Linear(640, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(2112, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU())
        self.reg = nn.Linear(64, 1)
        self.rank = nn.Linear(64, 1)
    def forward(self, x_raw, _=None):
        B = x_raw.shape[0]
        pi0_r = x_raw[:, 640:2240].reshape(B, 1, 5, 64, 5)
        pi0_b = x_raw[:, 2240:3840].reshape(B, 1, 5, 64, 5)
        t = torch.cat([pi0_r, pi0_b], dim=1)
        t = F.relu(self.bn1(self.conv1(t)))
        t = F.relu(self.bn2(self.conv2(t)))
        t = F.relu(self.bn3(self.conv3(t)))
        t = self.pool(t).flatten(1)
        s = self.scalar_enc(x_raw[:, :640])
        h = self.fc(torch.cat([t, s], 1))
        return {'reg': self.reg(h).squeeze(-1), 'rank': self.rank(h).squeeze(-1)}


def load_data(max_samples=200000, data_path=None):
    data = np.load(_resolve_data_path(data_path))
    total = len(data['y'])
    actual = min(max_samples, total)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(total, actual, replace=False)
    idx.sort()
    X = data['X'][idx].astype(np.float32)
    y = data['y'][idx].astype(np.float32)
    print(f"Loaded {actual:,}/{total:,} samples, y range=[{y.min():.0f},{y.max():.0f}], mean={y.mean():.1f}")
    return X, y


def create_splits(y, train_ratio=0.7, val_ratio=0.15):
    rng = np.random.RandomState(SEED)
    n = len(y)
    quantiles = np.percentile(y, np.linspace(0, 100, 11))
    bin_ids = np.clip(np.digitize(y, quantiles[:-1]) - 1, 0, 9)
    tr, va, te = [], [], []
    for b in range(10):
        mask = bin_ids == b
        indices = np.where(mask)[0]
        rng.shuffle(indices)
        nb = len(indices)
        ntr, nva = int(nb*train_ratio), int(nb*val_ratio)
        tr.append(indices[:ntr]); va.append(indices[ntr:ntr+nva]); te.append(indices[ntr+nva:])
    tr = np.concatenate(tr); rng.shuffle(tr)
    va = np.concatenate(va); rng.shuffle(va)
    te = np.concatenate(te); rng.shuffle(te)
    return {'train': tr, 'val': va, 'test': te}


def train_model(model, name, X, y, splits, device, epochs=200, batch_size=128, lr=1e-3,
                patience=30, ranking_weight=0.3):
    print(f"\n{'='*60}\nTraining {name} (ranking_weight={ranking_weight})\n{'='*60}")
    tr, va, te = splits['train'], splits['val'], splits['test']
    X_tr = torch.FloatTensor(X[tr].astype(np.float32))
    X_va = torch.FloatTensor(X[va].astype(np.float32))
    X_te = torch.FloatTensor(X[te].astype(np.float32))
    y_tr = y[tr].astype(np.float32); y_va = y[va].astype(np.float32); y_te = y[te].astype(np.float32)
    y_mean, y_std = y_tr.mean(), y_tr.std()
    y_tr_n = (y_tr - y_mean) / y_std; y_va_n = (y_va - y_mean) / y_std
    print(f"  train={len(tr):,}, val={len(va):,}, test={len(te):,}, y_mean={y_mean:.1f}")

    tr_ds = TensorDataset(X_tr, torch.FloatTensor(y_tr_n))
    va_ds = TensorDataset(X_va, torch.FloatTensor(y_va_n))
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size*2, shuffle=False, pin_memory=True)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss, best_state, counter = float('inf'), None, 0
    t0 = time.time(); history = []

    for ep in range(epochs):
        model.train()
        for x_b, y_b in tr_ld:
            opt.zero_grad()
            out = model(x_b.to(device))
            mse = nn.MSELoss()(out['reg'], y_b.to(device))
            rank = pairwise_ranking_loss(out['rank'], y_b.to(device))
            loss = mse + ranking_weight * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for x_b, y_b in va_ld:
                out = model(x_b.to(device))['reg']
                va_loss += nn.MSELoss()(out, y_b.to(device)).item() * len(x_b)
                va_n += len(x_b)
        va_loss /= va_n; history.append(va_loss)

        if va_loss < best_loss - 1e-5:
            best_loss, counter, best_state = va_loss, 0, copy.deepcopy(model.state_dict())
        else:
            counter += 1
        if ep % 20 == 19 or ep == 0:
            elapsed = time.time() - t0
            print(f"  Ep {ep+1:3d}: va_loss={va_loss:.4f} best={best_loss:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if counter >= patience:
            print(f"  Early stop @ epoch {ep+1}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        preds_list = []
        te_bs = 2048
        for i in range(0, len(X_te), te_bs):
            preds_list.append(
                model(X_te[i:i+te_bs].to(device))['reg'].cpu().numpy())
        preds = np.concatenate(preds_list) * y_std + y_mean
    mae = float(np.mean(np.abs(preds - y_te)))
    spear, _ = spearmanr(preds, y_te)
    pearson = float(np.corrcoef(preds, y_te)[0,1])
    elapsed = time.time() - t0
    print(f"  Test: MAE={mae:.2f}, Spearman={spear:.4f}, Pearson={pearson:.4f}")
    print(f"  Time: {elapsed:.0f}s")

    save_path = os.path.join(PROCESSED_DIR, f'model_{name}.pt')
    torch.save({'model_state': best_state, 'y_mean': float(y_mean), 'y_std': float(y_std),
                'test_mae': mae, 'test_spearman': spear, 'test_pearson': pearson,
                'n_params': n_params, 'history': history}, save_path)
    print(f"  Saved: {save_path}")

    del X_tr, X_va, X_te
    if device != 'cpu': torch.cuda.empty_cache()
    gc.collect()
    return {'mae': mae, 'spearman': spear, 'pearson': pearson, 'n_params': n_params}


def main():
    parser = argparse.ArgumentParser(description='SHA3-384 StageB model training')
    parser.add_argument('--model', type=str, default='all', choices=['mlp','cnn','all'])
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--ranking-weight', type=float, default=0.3,
                        help='ranking loss weight (0=MSE only)')
    parser.add_argument('--max-samples', type=int, default=200000)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X, y = load_data(args.max_samples, args.data_path)
    splits = create_splits(y)
    results = {}

    if args.model in ('mlp', 'all'):
        results['mlp_stageB'] = train_model(MLP_stageB(), 'mlp_stageB', X, y, splits, device,
                                            args.epochs, min(args.batch_size,256), args.lr,
                                            args.patience, args.ranking_weight)
    if args.model in ('cnn', 'all'):
        results['cnn_stageB'] = train_model(CNN_stageB(), 'cnn_stageB', X, y, splits, device,
                                            args.epochs, args.batch_size, args.lr,
                                            args.patience, args.ranking_weight)

    print(f"\n{'='*60}\nTraining complete")
    for name, r in results.items():
        print(f"  {name}: Spearman={r['spearman']:.4f}, MAE={r['mae']:.2f}, Pearson={r['pearson']:.4f}")

    result_path = os.path.join(PROCESSED_DIR, 'training_results_stageB.json')
    with open(result_path, 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv,(np.floating,np.integer)) else vv
                       for kk,vv in v.items()} for k,v in results.items()}, f, indent=2)
    print(f"Results saved: {result_path}")


if __name__ == '__main__':
    main()
