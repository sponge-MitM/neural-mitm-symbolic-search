'Module object.'

import os
import sys
import json
import time
import copy
import argparse
import warnings
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from ascon_propagator import (
    extract_all_features, get_feature_vector,
    extract_milp_linear_features,
    STATE_GROUPS, parse_all_states
)
from ascon_model import AsconMLP, AsconCNN, pairwise_ranking_loss


# ============================================================

# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Ascon Work 1 training and evaluation (revised)')
    p.add_argument('--mode', choices=['quick', 'full', 'custom'],
                   default='quick',
                   help='quick=2000 samples, full=all, custom=user-defined')
    p.add_argument('--n-sample', type=int, default=50000,
                   help='number of samples (mode=custom)')
    p.add_argument('--model', choices=['mlp', 'cnn', 'all'],
                   default='mlp', help='model type')
    p.add_argument('--epochs', type=int, default=0,
                   help='training epochs (0=auto)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--skip-gb', action='store_true',
                   help='skip slow GB baseline')
    p.add_argument('--extract-rules', action='store_true',
                   help='stage 3: extract heuristic rules')
    p.add_argument('--compare', action='store_true',
                   help='stage 4: comparative evaluation (raw vs NN vs rules)')
    p.add_argument('--data-dir',
                    default=os.path.join(_script_dir, '..', '..', 'data', 'ascon'),
                    help='data directory')
    p.add_argument('--output-dir',
                    default=os.path.join(_script_dir, '..', '..', 'data', 'ascon'),
                    help='output directory')
    p.add_argument('--device', default='auto',
                   help='device (auto/cpu/cuda)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude-z-details', action='store_true', default=True,
                    help='exclude per-z count features (default on)')
    p.add_argument('--ridge-top-k', type=int, default=0,
                    help='Ridge keep Top-K rules (0=all, recommended 6-12)')
    p.add_argument('--output-milp-code', action='store_true',
                    help='output v1-style MILP code snippet, embeddable in attack scripts')
    p.add_argument('--quick-ridge', action='store_true',
                    help='fast Ridge mode: skip MLP/CNN, GB+Ridge rule extraction only')
    p.add_argument('--ranking-weight', type=float, default=0.3,
                    help='ranking loss weight (0=MSE only, recommended 0.1-0.3)')
    p.add_argument('--feature-set', choices=['v2', 'milp_linear'],
                    default='v2',
                    help='feature set: v2=statistical (proxy), milp_linear=MILP-exact')
    return p.parse_args()


def get_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def _find_data_file(data_dir):
    'Function find data file.'
    if os.path.isfile(data_dir):
        return data_dir

    if os.path.isdir(data_dir):
        candidates = []
        for f in os.listdir(data_dir):
            if f.startswith('ascon_train_') and f.endswith('.npz'):
                candidates.append(os.path.join(data_dir, f))
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if candidates:
            return candidates[0]


    result_dir = os.path.join(os.path.dirname(data_dir), 'result')
    if os.path.isdir(result_dir):
        candidates = []
        for f in os.listdir(result_dir):
            if f.startswith('ascon_train_') and f.endswith('.npz'):
                candidates.append(os.path.join(result_dir, f))
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if candidates:
            return candidates[0]

    return None


# ============================================================

# ============================================================

def load_and_extract_features(data_path, n_sample, seed,
                               cache_path=None, exclude_z=True,
                               feature_set='v2'):
    'Function load and extract features.'
    if cache_path and os.path.exists(cache_path):
        print(f"[data] loaded from cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        X = cached['features']
        y = cached['targets']
        print(f"  features: {X.shape}, targets: {y.shape}")
        return X, y, cached.get('feat_names', None)

    print(f"[data] loading raw data: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    r_all = data['r']
    b_all = data['b']
    targets_all = data['targets']
    N_total = len(targets_all)
    feat_dim = r_all.shape[1]

    print(f"  total samples: {N_total}, feature dims: r={feat_dim}, b={feat_dim}")

    if feat_dim == 480:
        print("  WARNING: v1 format detected (480 dims), limited feature info")
        print("    regenerate v2 data (1280 dims) with the revised Ascon_XOF_MitM.py")
    elif feat_dim == 1280:
        print("  OK v2 format (1280 dims); using full intermediate states")

    if n_sample and n_sample < N_total:
        np.random.seed(seed)
        indices = sorted(np.random.choice(N_total, size=n_sample, replace=False))
        print(f"  sampled: {n_sample}")
    else:
        indices = list(range(N_total))
        n_sample = N_total
        print(f"  using all: {n_sample}")

    use_milp = (feature_set == 'milp_linear')
    feat_label = 'MILP-linearizable' if use_milp else 'v2'
    print(f"[features] extracting {feat_label} features (n={n_sample})...")
    X_list, y_list = [], []
    t0 = time.time()

    for idx, i in enumerate(indices):
        if use_milp:
            feats = extract_milp_linear_features(r_all[i], b_all[i])
        else:
            feats = extract_all_features(r_all[i], b_all[i])
        X_list.append(get_feature_vector(feats, exclude_z_details=exclude_z))
        y_list.append(float(targets_all[i]))
        if (idx + 1) % 20000 == 0:
            print(f"  progress: {idx+1}/{n_sample} ({100*(idx+1)/n_sample:.0f}%)")

    elapsed = time.time() - t0
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  done in {elapsed:.1f}s ({n_sample/elapsed:.0f} samples/s)")
    print(f"  feature matrix: {X.shape}")


    assert np.isnan(X).sum() == 0, "features contain NaN!"
    assert np.isinf(X).sum() == 0, "features contain Inf!"


    feat_names_out = None
    if use_milp:
        sample_feats = extract_milp_linear_features(r_all[0], b_all[0])
    else:
        sample_feats = extract_all_features(r_all[0], b_all[0])
    feat_names_full = sorted(sample_feats.keys())
    feat_names_compact = [k for k in feat_names_full
                          if not (k.count('_z') > 0 and k.split('_z')[-1].isdigit())]
    feat_names_out = feat_names_compact

    if cache_path:
        print(f"[cache] saved to: {cache_path}")
        np.savez_compressed(cache_path, features=X, targets=y,
                           feat_names=np.array(feat_names_compact))

    return X, y, feat_names_out


def split_data(X, y, seed):
    'Function split data.'
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15/0.85, random_state=seed)
    print(f"[split] tr={len(X_tr)}, va={len(X_va)}, te={len(X_te)}")
    return (X_tr, X_va, X_te), (y_tr, y_va, y_te)


# ============================================================

# ============================================================

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_mlp(X_tr_s, y_tr_n, X_va_s, y_va_n, args, device):
    'Function train mlp.'
    print(f"\n{'='*50}\nStage 2: training MLP\n{'='*50}")
    input_dim = X_tr_s.shape[1]
    lambda_rank = args.ranking_weight
    use_ranking = lambda_rank > 0
    print(f"  input dim: {input_dim}")
    if use_ranking:
        print(f"  joint loss: MSE + {lambda_rank}xRankingLoss")

    model = AsconMLP(input_dim, hidden_dims=[256, 128, 64], dropout=0.2).to(device)

    tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr_n))
    va_ds = TensorDataset(torch.FloatTensor(X_va_s), torch.FloatTensor(y_va_n))
    tr_loader = DataLoader(tr_ds, args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=torch.cuda.is_available())
    va_loader = DataLoader(va_ds, args.batch_size * 2, shuffle=False,
                           num_workers=2, pin_memory=torch.cuda.is_available())

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    early = EarlyStopping(args.patience)
    history = []

    for ep in range(args.epochs):
        model.train()
        tr_loss, tr_mse, tr_rank = 0.0, 0.0, 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            out = model(xb.to(device))
            mse = nn.functional.mse_loss(out['reg'], yb.to(device))
            loss = mse
            tr_mse += mse.item()
            if use_ranking:

                yb_raw = yb * args.y_std + args.y_mean
                rank = pairwise_ranking_loss(out['rank'], yb_raw.to(device), margin=2.0)
                loss = mse + lambda_rank * rank
                tr_rank += rank.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item()
        sched.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                out = model(xb.to(device))
                va_loss += nn.functional.mse_loss(
                    out['reg'], yb.to(device)).item()
        va_loss /= len(va_loader)
        history.append({
            'epoch': ep + 1,
            'train_loss': tr_loss / len(tr_loader),
            'val_loss': va_loss,
        })

        if (ep + 1) % 10 == 0:
            msg = f"  Epoch {ep+1:4d}: val_mse={va_loss:.4f}"
            if use_ranking:
                msg += f" (tr_mse={tr_mse/len(tr_loader):.4f} rank={tr_rank/len(tr_loader):.4f})"
            print(msg)

        if early.step(va_loss, model):
            print(f"  early stopping at epoch {ep+1}, best_val={early.best:.4f}")
            break

    model.load_state_dict(early.best_state)
    return model, history


def train_cnn(X_tr_s, y_tr_n, X_va_s, y_va_n, args, device):
    'Function train cnn.'
    print(f"\n{'='*50}\nStage 2: training CNN (v2 has no tensor input; equivalent to MLP)\n{'='*50}")
    print("  note: v2 features have no 2D tensors; CNN degenerates to MLP")
    return train_mlp(X_tr_s, y_tr_n, X_va_s, y_va_n, args, device)


# ============================================================

# ============================================================

def evaluate_model(model, model_type, X_te_s, y_te, y_mean, y_std, device):
    'Function evaluate model.'
    model.eval()
    reg_preds, rank_preds = [], []
    with torch.no_grad():
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_te_s)), 256,
            num_workers=2, pin_memory=torch.cuda.is_available())
        for (xb,) in loader:
            out = model(xb.to(device))
            reg_preds.extend(out['reg'].cpu().numpy().tolist())
            rank_preds.extend(out['rank'].cpu().numpy().tolist())

    reg_r = np.array(reg_preds) * y_std + y_mean
    rank_r = np.array(rank_preds) * y_std + y_mean
    mae = float(np.mean(np.abs(reg_r - y_te)))
    rmse = float(np.sqrt(np.mean((reg_r - y_te) ** 2)))
    sp, sp_p = spearmanr(reg_r, y_te)
    pr, pr_p = pearsonr(reg_r, y_te)
    rank_sp, _ = spearmanr(rank_r, y_te)

    result = {
        'mae': round(mae, 4), 'rmse': round(rmse, 4),
        'spearman': round(float(sp), 4), 'spearman_p': float(sp_p),
        'pearson': round(float(pr), 4), 'pearson_p': float(pr_p),
    }
    if rank_sp is not None:
        result['rank_spearman'] = round(float(rank_sp), 4)
    return result


def eval_gb_baseline(X_tr_s, y_tr, X_te_s, y_te):
    """GB baseline (single-core, reduced trees to keep it fast)."""
    gb = GradientBoostingRegressor(
        n_estimators=50, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gb.fit(X_tr_s, y_tr)
    preds = gb.predict(X_te_s)
    sp, _ = spearmanr(preds, y_te)
    mae = float(np.mean(np.abs(preds - y_te)))
    return {
        'mae': round(mae, 4),
        'spearman': round(float(sp), 4),
        'feature_importances': gb.feature_importances_.tolist(),
    }


# ============================================================

# ============================================================

def _write_milp_code(output_dir, weights, feat_names_used, topk_sp, full_sp,
                     top_k, n_total):
    'Function write milp code.'
    milp_path = os.path.join(output_dir, 'ascon_milp_rules_v1.py')
    lines = []
    lines.append('"""')
    lines.append(f'Ascon v1-style Ridge rules - MILP code snippet')
    lines.append(f'Auto-generated at: {time.strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'Training data: {n_total} samples')
    lines.append(f'Ridge Top-{top_k} Spearman: {topk_sp:.4f} (full: {full_sp:.4f})')
    lines.append(f'')
    lines.append(f'Usage:')
    lines.append(f'  1. Copy the weight constants below into the attack script (Ascon_XOF_3_preimage.py, etc.)')
    lines.append(f'  2. Hand-write the v1-style MILP proxy variables (active_z, z_adj_pairs, etc.)')
    lines.append(f'  3. Replace the heuristic_bonus expression')
    lines.append('"""')
    lines.append('')
    lines.append(f'# === Ridge Top-{top_k} rule weights ===')
    lines.append(f'# Full Spearman={full_sp:.4f}, Top-{top_k} Spearman={topk_sp:.4f}')
    lines.append(f'RIDGE_SCALE = 0.0015  # global scale (v1 conservative value, tunable)')
    lines.append('')

    for i, w in enumerate(weights):
        name_safe = w['feature'].replace('-', '_').replace('.', '_')
        lines.append(f'# {i+1}. {w["feature"]} ({w["direction"]}, weight={w["weight"]:.4f})')
        lines.append(f'W_{name_safe.upper()[:40]} = {w["weight"]:.4f}')
    lines.append('')
    lines.append('# === Build heuristic_bonus (added to the MAXIMIZE objective) ===')
    lines.append('# Direction: (+) = MAXIMIZE (reward), (-) = MINIMIZE (penalty)')
    lines.append('# MILP proxy variables must be built manually (see v1 Ascon_XOF_3_preimage.py)')
    lines.append('heuristic_bonus = (')
    for w in weights:
        dir_sign = '+' if w['direction'] == 'MAXIMIZE' else '-'
        name_safe = w['feature'].replace('-', '_').replace('.', '_')[:40]
        lines.append(f'    {dir_sign} RIDGE_SCALE * W_{name_safe.upper()}'
                     f'  # {w["feature"]} ({w["direction"]})')
    lines.append(')')

    with open(milp_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  OK v1-style MILP code saved to: {milp_path}")


def extract_heuristic_rules(X_tr_raw, y_tr, X_te_raw, y_te,
                             feat_names, output_dir,
                             ridge_top_k=0, output_milp_code=False):
    'Function extract heuristic rules.'
    print(f"\n{'='*50}\nStage 3: rule extraction\n{'='*50}")

    if feat_names is None:
        print("  WARNING: no feature names; skipping rule extraction")
        return {}

    # ensure feat_names is a plain Python list (can be numpy array from cache)
    feat_names = list(feat_names)

    results = {'feature_names': feat_names}


    valid_mask = []
    for i in range(X_te_raw.shape[1]):
        col = X_te_raw[:, i]
        valid_mask.append(np.std(col) > 1e-8)
    n_valid = sum(valid_mask)
    n_total = len(valid_mask)
    print(f"[rules] valid features: {n_valid}/{n_total} (dropped {n_total-n_valid} constant features)")

    def _is_valid_sp(x):
        """True if x is a valid finite float."""
        if x is None:
            return False
        try:
            return not np.isnan(x)
        except (TypeError, ValueError):
            return False


    print("[rules] marginal correlations...")
    corrs = []
    for i, name in enumerate(feat_names):
        if not valid_mask[i]:
            continue  # skip constant features
        try:
            sp, _ = spearmanr(X_te_raw[:, i], y_te)
            pr, _ = pearsonr(X_te_raw[:, i], y_te)
            if _is_valid_sp(sp):
                corrs.append({
                    'name': name,
                    'spearman': round(float(sp), 4),
                    'pearson': round(float(pr), 4),
                    'abs_sp': abs(float(sp)),
                })
        except:
            pass
    corrs.sort(key=lambda x: x['abs_sp'], reverse=True)
    results['marginal_correlations'] = corrs[:30]


    print("[rules] decision tree distillation...")

    valid_corrs = [c for c in corrs if _is_valid_sp(c['spearman'])]
    top_n = min(12, len(valid_corrs))
    top_names = [c['name'] for c in valid_corrs[:top_n]]
    top_idx = [feat_names.index(n) for n in top_names]

    dt = DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=300, random_state=42)
    dt.fit(X_tr_raw[:, top_idx], y_tr)
    dt_preds = dt.predict(X_te_raw[:, top_idx])
    dt_sp, _ = spearmanr(dt_preds, y_te)

    results['decision_tree'] = {
        'spearman': round(float(dt_sp), 4),
        'mae': round(float(np.mean(np.abs(dt_preds - y_te))), 4),
        'features_used': top_names,
        'feature_importances': {
            n: round(float(imp), 4)
            for n, imp in zip(top_names, dt.feature_importances_)
        },
        'rules_text': export_text(
            dt, feature_names=top_names, max_depth=4,
            decimals=2, spacing=3),
    }


    print("[rules] Ridge weighting...")

    ridge_features = []
    for c in corrs:
        sp = c['spearman']
        if not _is_valid_sp(sp):
            continue
        if abs(sp) < 0.02:
            continue
        direction = -1 if sp < 0 else 1
        ridge_features.append((c['name'], direction))
    print(f"  Ridge features used: {len(ridge_features)}")

    if len(ridge_features) < 3:
        print("  WARNING: not enough valid features; skipping Ridge")
        return results

    ridge_names = [n for n, _ in ridge_features]
    ridge_dirs = [d for _, d in ridge_features]
    ridge_idx = [feat_names.index(n) for n in ridge_names]


    X_ridge = np.column_stack([
        X_tr_raw[:, ridge_idx[i]] * ridge_dirs[i]
        for i in range(len(ridge_names))
    ])
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_ridge, y_tr)
    X_ridge_te = np.column_stack([
        X_te_raw[:, ridge_idx[i]] * ridge_dirs[i]
        for i in range(len(ridge_names))
    ])
    ridge_preds = ridge.predict(X_ridge_te)
    ridge_sp, _ = spearmanr(ridge_preds, y_te)


    total_w = np.sum(np.abs(ridge.coef_))
    if total_w < 1e-8:
        total_w = 1.0
    milp_weights = []
    for i, name in enumerate(ridge_names):
        raw = ridge.coef_[i]
        norm = abs(raw) / total_w
        sign = '-' if ridge_dirs[i] == -1 else '+'
        milp_weights.append({
            'feature': name,
            'direction': 'MAXIMIZE' if ridge_dirs[i] == -1 else 'MINIMIZE',
            'weight': round(float(norm), 4),
            'raw_coef': round(float(raw), 4),
            'milp_code': f"{sign}{norm:.4f} * {name}",
        })
    milp_weights.sort(key=lambda x: -x['weight'])

    results['ridge'] = {
        'spearman': round(float(ridge_sp), 4),
        'mae': round(float(np.mean(np.abs(ridge_preds - y_te))), 4),
        'weights': milp_weights,
    }


    if ridge_top_k > 0 and ridge_top_k < len(milp_weights):
        topk = milp_weights[:ridge_top_k]
        topk_names = [w['feature'] for w in topk]
        topk_idx = [feat_names.index(n) for n in topk_names]
        topk_dirs = [ridge_dirs[ridge_names.index(n)] for n in topk_names]

        X_topk = np.column_stack([
            X_tr_raw[:, topk_idx[i]] * topk_dirs[i]
            for i in range(len(topk_names))
        ])
        ridge_topk = Ridge(alpha=5.0)
        ridge_topk.fit(X_topk, y_tr)
        X_topk_te = np.column_stack([
            X_te_raw[:, topk_idx[i]] * topk_dirs[i]
            for i in range(len(topk_names))
        ])
        topk_preds = ridge_topk.predict(X_topk_te)
        topk_sp, _ = spearmanr(topk_preds, y_te)
        total_w_topk = np.sum(np.abs(ridge_topk.coef_))
        if total_w_topk < 1e-8:
            total_w_topk = 1.0

        topk_weights = []
        for i, name in enumerate(topk_names):
            raw = ridge_topk.coef_[i]
            norm = abs(raw) / total_w_topk
            sign = '-' if topk_dirs[i] == -1 else '+'
            topk_weights.append({
                'feature': name,
                'direction': 'MAXIMIZE' if topk_dirs[i] == -1 else 'MINIMIZE',
                'weight': round(float(norm), 4),
                'raw_coef': round(float(raw), 4),
                'milp_code': f"{sign}{norm:.4f} * {name}",
            })
        topk_weights.sort(key=lambda x: -x['weight'])

        results['ridge_topk'] = {
            'top_k': ridge_top_k,
            'spearman': round(float(topk_sp), 4),
            'mae': round(float(np.mean(np.abs(topk_preds - y_te))), 4),
            'weights': topk_weights,
        }
        print(f"  Ridge Top-{ridge_top_k} Spearman: {topk_sp:.4f} "
              f"(full: {ridge_sp:.4f}, loss: {ridge_sp - topk_sp:.4f})")


        if output_milp_code:
            n_total = len(y_tr) + len(y_te)
            _write_milp_code(output_dir, topk_weights, topk_names,
                            topk_sp, ridge_sp, ridge_top_k, n_total)
    elif output_milp_code and ridge_top_k == 0:

        n_total = len(y_tr) + len(y_te)
        _write_milp_code(output_dir, milp_weights, ridge_names,
                        ridge_sp, ridge_sp, len(milp_weights), n_total)


    print("[rules] good/bad distribution comparison...")
    median = np.median(y_te)
    good = y_te <= median
    bad = ~good
    comparisons = []
    for c in corrs[:15]:
        if c['spearman'] is None:
            continue
        idx = feat_names.index(c['name'])
        gm = float(np.mean(X_te_raw[good, idx]))
        bm = float(np.mean(X_te_raw[bad, idx]))
        comparisons.append({
            'feature': c['name'],
            'good_mean': round(gm, 4),
            'bad_mean': round(bm, 4),
            'diff': round(gm - bm, 4),
        })
    results['good_vs_bad'] = comparisons

    return results


# ============================================================

# ============================================================

def compare_strategies(all_results, X_te_raw, y_te, feat_names):
    'Function compare strategies.'
    print(f"\n{'='*60}")
    print("Stage 4: comparative evaluation")
    print(f"{'='*60}")


    baseline_mean = float(np.mean(y_te))
    baseline_std = float(np.std(y_te))
    baseline_median = float(np.median(y_te))
    n_below_baseline_median = int(np.sum(y_te < baseline_median))

    comparison = {
        'test_size': len(y_te),
        'baseline': {
            'mean': round(baseline_mean, 2),
            'std': round(baseline_std, 2),
            'median': round(baseline_median, 2),
            'min': int(np.min(y_te)),
            'max': int(np.max(y_te)),
        },
        'strategies': {}
    }


    gb = all_results.get('gb_baseline', {})
    if gb and not gb.get('skipped'):
        comparison['strategies']['gb_baseline'] = {
            'spearman': gb.get('spearman', None),
            'mae': gb.get('mae', None),
        }

    # MLP
    mlp = all_results.get('mlp_eval', {})
    if mlp:
        comparison['strategies']['mlp'] = {
            'spearman': mlp.get('spearman', None),
            'mae': mlp.get('mae', None),
            'rmse': mlp.get('rmse', None),
        }

    # CNN
    cnn = all_results.get('cnn_eval', {})
    if cnn:
        comparison['strategies']['cnn'] = {
            'spearman': cnn.get('spearman', None),
            'mae': cnn.get('mae', None),
            'rmse': cnn.get('rmse', None),
        }


    rules = all_results.get('rules', {})
    ridge = rules.get('ridge', {})
    if ridge:
        comparison['strategies']['ridge_rules'] = {
            'spearman': ridge.get('spearman', None),
            'mae': ridge.get('mae', None),
            'n_features': len(ridge.get('weights', [])),
        }


    dt = rules.get('decision_tree', {})
    if dt:
        comparison['strategies']['decision_tree'] = {
            'spearman': dt.get('spearman', None),
            'mae': dt.get('mae', None),
            'n_features': len(dt.get('features_used', [])),
        }


    print(f"\n  {'strategy':25s} {'Spearman':>10s} {'MAE':>10s} {'note'}")
    print(f"  {'-'*55}")
    print(f"  {'raw MILP (random mean)':25s} {'N/A':>10s} "
          f"{baseline_mean:>10.2f}  mean")

    for name, info in comparison['strategies'].items():
        sp = info.get('spearman', 'N/A')
        mae = info.get('mae', 'N/A')
        extra = ''
        if 'n_features' in info:
            extra = f"  ({info['n_features']} features)"
        print(f"  {name:25s} {str(sp):>10s} {str(mae):>10s}{extra}")


    print(f"\n  improvement (Spearman gain):")
    best_sp = 0
    best_name = ''
    for name, info in comparison['strategies'].items():
        sp = info.get('spearman', 0) or 0
        if sp > best_sp:
            best_sp = sp
            best_name = name
    if best_sp > 0:
        print(f"  best strategy: {best_name} (Spearman={best_sp:.4f})")

    return comparison


# ============================================================

# ============================================================

def main():
    args = parse_args()


    if args.mode == 'quick':
        n_sample = 2000
        if args.epochs == 0:
            args.epochs = 30
    elif args.mode == 'full':
        n_sample = None
        if args.epochs == 0:
            args.epochs = 200
    else:
        n_sample = args.n_sample
        if args.epochs == 0:
            args.epochs = 100

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Ascon Work 1 - training and evaluation (v1 improved)")
    print("=" * 60)
    print(f"  mode: {args.mode}")
    print(f"  model: {args.model}")
    print(f"  samples: {n_sample if n_sample else 'all'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  device: {device}")
    print(f"  seed: {args.seed}")
    print(f"  Ridge Top-K: {args.ridge_top_k if args.ridge_top_k > 0 else 'all'}")
    print(f"  MILP code output: {'yes' if args.output_milp_code else 'no'}")
    print(f"  fast Ridge: {'yes' if args.quick_ridge else 'no'}")
    print(f"  ranking weight: {args.ranking_weight}")
    feat_desc = 'MILP-linearizable (counts/AND/diff)' if args.feature_set == 'milp_linear' else 'v2 statistical features'
    print(f"  feature set: {feat_desc}")



    data_path = _find_data_file(args.data_dir)
    if data_path is None:
        print(f"\nx no training data file found in {args.data_dir}")
        print("run ascon_gen_training.py first to generate training data")
        print("or copy result/ascon_train_*.npz into the ascon/data/ directory")
        sys.exit(1)
    print(f"[data] using file: {os.path.basename(data_path)}")

    feat_prefix = 'milp' if args.feature_set == 'milp_linear' else 'v2'
    cache_name = f"ascon_features_{feat_prefix}_{n_sample if n_sample else 'full'}.npz"
    cache_path = os.path.join(args.output_dir, cache_name)

    X, y, feat_names = load_and_extract_features(
        data_path, n_sample, args.seed, cache_path,
        exclude_z=args.exclude_z_details,
        feature_set=args.feature_set
    )

    (X_tr, X_va, X_te), (y_tr, y_va, y_te) = split_data(X, y, args.seed)


    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_tr_n = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_va_n = ((y_va - y_mean) / y_std).astype(np.float32)
    args.y_mean = y_mean  # for ranking loss denormalization
    args.y_std = y_std

    print(f"[preprocess] target: mean={y_mean:.2f}, std={y_std:.2f}")
    print(f"[preprocess] feature dim: {X_tr_s.shape[1]}")

    all_results = {}
    all_results['config'] = {
        'mode': args.mode,
        'n_sample': len(X),
        'n_features': X.shape[1],
        'n_train': len(X_tr),
        'n_val': len(X_va),
        'n_test': len(X_te),
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'device': str(device),
        'seed': args.seed,
        'feature_version': args.feature_set,
        'ranking_weight': args.ranking_weight,
    }


    if not args.skip_gb:
        print(f"\n{'='*50}\nGB baseline\n{'='*50}")
        gb_result = eval_gb_baseline(X_tr_s, y_tr, X_te_s, y_te)
        print(f"  GB MAE={gb_result['mae']:.4f}, Spearman={gb_result['spearman']:.4f}")
        all_results['gb_baseline'] = gb_result
    else:
        print(f"\n{'='*50}\nGB baseline: skipped (--skip-gb)\n{'='*50}")
        all_results['gb_baseline'] = {'skipped': True}


    if not args.quick_ridge:
        for model_type in (['mlp', 'cnn'] if args.model == 'all' else [args.model]):
            print(f"\n{'#'*60}")
            print(f"# Stage 2: training {model_type.upper()}")
            print(f"{'#'*60}")

            if model_type == 'mlp':
                model, history = train_mlp(
                    X_tr_s, y_tr_n, X_va_s, y_va_n, args, device)
            else:
                model, history = train_cnn(
                    X_tr_s, y_tr_n, X_va_s, y_va_n, args, device)


            eval_result = evaluate_model(
                model, model_type, X_te_s, y_te, y_mean, y_std, device)
            all_results[f'{model_type}_eval'] = eval_result
            print(f"  {model_type.upper()}: MAE={eval_result['mae']:.4f}, "
                  f"RMSE={eval_result['rmse']:.4f}, Spearman={eval_result['spearman']:.4f}")


            model_path = os.path.join(
                args.output_dir, f'ascon_model_{model_type}_v2.pt')
            save_dict = {
                'model_state': model.state_dict(),
                'model_type': model_type,
                'scalar_dim': X.shape[1],
                'n_aux_tasks': 0,
                'y_mean': y_mean,
                'y_std': y_std,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
            }
            torch.save(save_dict, model_path)
            print(f"  model saved to: {model_path}")
    else:
        print(f"\n{'#'*60}")
        print(f"# Stage 2: skipped (--quick-ridge)")
        print(f"{'#'*60}")


    if args.extract_rules:
        rules = extract_heuristic_rules(
            X_tr, y_tr, X_te, y_te, feat_names, args.output_dir,
            ridge_top_k=args.ridge_top_k,
            output_milp_code=args.output_milp_code)
        all_results['rules'] = rules


        print(f"\n{'='*50}")
        print("extracted heuristic rule summary")
        print(f"{'='*50}")
        if 'ridge' in rules:
            print("\nrecommended MILP objective weights:")
            for w in rules['ridge']['weights'][:8]:
                print(f"  {w['milp_code']}")
            print(f"  (combined Spearman={rules['ridge']['spearman']:.4f})")

        if 'decision_tree' in rules:
            print(f"\ndecision tree rules (Spearman={rules['decision_tree']['spearman']:.4f}):")
            lines = rules['decision_tree']['rules_text'].split('\n')[:12]
            for line in lines:
                print(f"  {line}")
            if len(lines) >= 12:
                print(f"  ... (full rules in JSON)")


    if args.compare:

        if feat_names is None:
            print("\nWARNING: no feature names (loaded from cache); skipping stage 4 comparison")
        else:
            comparison = compare_strategies(
                all_results, X_te, y_te, feat_names)
            all_results['comparison'] = comparison


    results_path = os.path.join(args.output_dir, 'ascon_eval_results_v2.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nevaluation results saved to: {results_path}")


    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    gb = all_results.get('gb_baseline', {})
    if not gb.get('skipped'):
        print(f"  GB baseline:          Spearman={gb.get('spearman', 0):.4f}")
    else:
        print(f"  GB baseline:          skipped")
    if not args.quick_ridge:
        for mt in (['mlp', 'cnn'] if args.model == 'all' else [args.model]):
            if f'{mt}_eval' in all_results:
                ev = all_results[f'{mt}_eval']
                sp = ev['spearman']
                print(f"  {mt.upper():6s}:             Spearman={sp:.4f}", end='')
                if 'rank_spearman' in ev:
                    print(f" (rank_head={ev['rank_spearman']:.4f})")
                else:
                    print()
    else:
        print(f"  MLP/CNN:              skipped (--quick-ridge)")
    if args.extract_rules and 'ridge' in all_results.get('rules', {}):
        rsp = all_results['rules']['ridge']['spearman']
        dt_sp = all_results['rules']['decision_tree']['spearman']
        print(f"  Ridge rules (full):   Spearman={rsp:.4f}")
        print(f"  decision tree rules:  Spearman={dt_sp:.4f}")
        if 'ridge_topk' in all_results['rules']:
            tk = all_results['rules']['ridge_topk']
            print(f"  Ridge Top-{tk['top_k']}:          Spearman={tk['spearman']:.4f}")

    print(f"\nOK Done!")


if __name__ == '__main__':
    main()
