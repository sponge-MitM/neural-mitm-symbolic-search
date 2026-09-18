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
from sklearn.preprocessing import StandardScaler

_model_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _model_dir)
from model import CyclicConv3d

REPO_ROOT = os.path.dirname(os.path.dirname(_model_dir))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
RESULT_DIR = DATA_DIR
PROCESSED_DIR = os.path.join(DATA_DIR, 'sha3_512')
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEED = 42
FEAT_DIM = 3712              # init(512) + pi1_r(1600) + pi1_b(1600)
INIT_DIM = 512               # init_red(256) + init_blue(256)
PI1_DIM = 3200               # pi1_r(1600) + pi1_b(1600)


# ============================================================

# ============================================================

class MLP_stageB(nn.Module):
    'Class MLP stageB.'
    def __init__(self, input_dim=FEAT_DIM, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.reg = nn.Sequential(nn.Linear(prev, 32), nn.ReLU(), nn.Linear(32, 1))
        self.rank = nn.Sequential(nn.Linear(prev, 32), nn.ReLU(), nn.Linear(32, 1))

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
        self.scalar_enc = nn.Sequential(
            nn.Linear(INIT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(2112, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU())
        self.reg = nn.Linear(64, 1)
        self.rank = nn.Linear(64, 1)

    def forward(self, x_raw, _=None):
        B = x_raw.shape[0]
        # init_r: [0:256], init_b: [256:512], pi1_r: [512:2112], pi1_b: [2112:3712]
        pi1_r = x_raw[:, 512:2112].reshape(B, 1, 5, 64, 5)   # (B,1,5,64,5)
        pi1_b = x_raw[:, 2112:3712].reshape(B, 1, 5, 64, 5)
        t = torch.cat([pi1_r, pi1_b], dim=1)  # (B,2,5,64,5)
        t = F.relu(self.bn1(self.conv1(t)))
        t = F.relu(self.bn2(self.conv2(t)))
        t = F.relu(self.bn3(self.conv3(t)))
        t = self.pool(t).flatten(1)
        s = self.scalar_enc(x_raw[:, :INIT_DIM])
        h = self.fc(torch.cat([t, s], 1))
        return {'reg': self.reg(h).squeeze(-1), 'rank': self.rank(h).squeeze(-1)}


# ============================================================

# ============================================================

def find_data_path():
    'Function find data path.'
    candidates = []
    for f in os.listdir(RESULT_DIR):
        if f.startswith('sha3_512_train_stageB_') and f.endswith('.npz'):
            candidates.append(os.path.join(RESULT_DIR, f))
    if not candidates:
        # fallback to work1 default
        default = os.path.join(RESULT_DIR, 'sha3_512_train_1000000.npz')
        if os.path.exists(default):
            print(f"[Warning] StageB data not found, using Work 1 data: {default}")
            return default
        raise FileNotFoundError(f"StageB training data not found in {RESULT_DIR}")

    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


def load_data(data_path, max_samples=200000):
    data = np.load(data_path)
    total = len(data['y'])
    actual = min(max_samples, total)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(total, actual, replace=False)
    idx.sort()
    X = data['X'][idx].astype(np.float32)
    y = data['y'][idx].astype(np.float32)
    print(f"Loaded {actual:,}/{total:,} samples, y range=[{y.min():.0f},{y.max():.0f}], "
          f"mean={y.mean():.1f}, std={y.std():.1f}")
    return X, y


def create_stratified_splits(y, train_ratio=0.7, val_ratio=0.15):
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
        ntr, nva = int(nb * train_ratio), int(nb * val_ratio)
        tr.append(indices[:ntr]); va.append(indices[ntr:ntr+nva])
        te.append(indices[ntr+nva:])
    tr = np.concatenate(tr); rng.shuffle(tr)
    va = np.concatenate(va); rng.shuffle(va)
    te = np.concatenate(te); rng.shuffle(te)
    return {'train': tr, 'val': va, 'test': te}


# ============================================================

# ============================================================

def train_model(model, train_loader, val_loader, device, lr=1e-3, epochs=200,
                patience=30, weight_decay=1e-4, ranking_weight=0.3):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out['reg'], y_batch)
            loss = loss + ranking_weight * pairwise_ranking_loss(out['rank'], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(device)
                y_batch = batch[1].to(device)
                out = model(x_batch)
                loss = criterion(out['reg'], y_batch)
                val_loss += loss.item() * x_batch.size(0)
                all_preds.append(out['reg'].cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        spear, _ = spearmanr(preds, targets)

        scheduler.step()

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, Spearman={spear:.4f}")

        if counter >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_loss


def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    criterion = nn.MSELoss()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            out = model(x_batch)
            loss = criterion(out['reg'], y_batch)
            test_loss += loss.item() * x_batch.size(0)
            all_preds.append(out['reg'].cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    spear, _ = spearmanr(preds, targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return {'loss': test_loss, 'mae': mae, 'rmse': rmse, 'spearman': spear}


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SHA3-512 StageB model training')
    parser.add_argument('--model', type=str, default='all',
                        choices=['mlp', 'cnn', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--max-samples', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--ranking-weight', type=float, default=0.3,
                        help='ranking loss weight (0=MSE only)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = args.data_path or find_data_path()
    print(f"Data: {data_path}")

    X, y = load_data(data_path, args.max_samples)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    y_mean, y_std = y.mean(), y.std()
    y_norm = ((y - y_mean) / y_std).astype(np.float32)

    splits = create_stratified_splits(y)
    print(f"Splits: train={len(splits['train']):,} val={len(splits['val']):,} "
          f"test={len(splits['test']):,}")

    results = {}

    models_to_train = ['mlp', 'cnn'] if args.model == 'all' else [args.model]

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} StageB")
        print(f"{'='*60}")

        if model_name == 'mlp':
            model = MLP_stageB()
        else:
            model = CNN_stageB()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        tr_idx = splits['train']; va_idx = splits['val']; te_idx = splits['test']

        tr_ds = TensorDataset(torch.from_numpy(X_scaled[tr_idx]),
                              torch.from_numpy(y_norm[tr_idx]))
        va_ds = TensorDataset(torch.from_numpy(X_scaled[va_idx]),
                              torch.from_numpy(y_norm[va_idx]))
        te_ds = TensorDataset(torch.from_numpy(X_scaled[te_idx]),
                              torch.from_numpy(y_norm[te_idx]))

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               pin_memory=True, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               pin_memory=True, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                               pin_memory=True, num_workers=0)

        t0 = time.time()
        model, best_val_loss = train_model(
            model, tr_loader, va_loader, device,
            lr=args.lr, epochs=args.epochs, patience=args.patience,
            ranking_weight=args.ranking_weight)
        train_time = time.time() - t0
        print(f"Training time: {train_time:.0f}s")

        eval_result = evaluate(model, te_loader, device)
        eval_result['train_time'] = train_time
        eval_result['n_params'] = n_params
        results[model_name] = eval_result

        print(f"  Test set: MAE={eval_result['mae']:.2f}, RMSE={eval_result['rmse']:.2f}, "
              f"Spearman={eval_result['spearman']:.4f}")


        save_path = os.path.join(PROCESSED_DIR, f'model_{model_name}_stageB.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'y_mean': float(y_mean), 'y_std': float(y_std),
            'eval': eval_result,
            'config': vars(args),
        }, save_path)
        print(f"  Model saved: {save_path}")


    summary_path = os.path.join(PROCESSED_DIR, 'training_results_stageB.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults summary: {summary_path}")

    return results


if __name__ == '__main__':
    main()
