"""
distill/proxy_metrics.py - proxy-metric comparison experiment (paper material: baseline vs rules vs NN)

Protocol: for each primitive, compare the three models on the proxy metric N_mix over the training dataset's configuration pool:
  - baseline : configurations produced by the original MILP search (mean/std of y over the full dataset)
  - rules    : mean/std of y for the top-k configurations ranked by distilled-rule scores (Ridge fitted to QP soft labels)
  - NN       : mean/std of y for the top-k configurations ranked by QP soft labels

Output: a per-primitive comparison table (top-1% / top-5%), for use in the paper's applications section.

Usage: python distill/proxy_metrics.py [--k1 0.01 --k2 0.05]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
OUT = os.path.join(REPO_ROOT, 'proxy_metrics_results.json')


def topk_stats(y, score, k):
    n = max(1, int(len(y) * k))
    idx = np.argsort(score)[:n]  # score is lower-better (smaller N_mix is better)
    sel = y[idx]
    return float(sel.mean()), float(sel.std()), float(sel.min()), float(sel.max())


def report(name, y_all, score_rules, score_nn, k1, k2):
    base_m, base_s = float(y_all.mean()), float(y_all.std())
    print('=' * 78)
    print('%-12s  baseline: mean=%.2f std=%.2f  (n=%d, y[%d..%d])' % (
        name, base_m, base_s, len(y_all), y_all.min(), y_all.max()))
    for k in (k1, k2):
        m_r, s_r, mn_r, mx_r = topk_stats(y_all, score_rules, k)
        m_n, s_n, mn_n, mx_n = topk_stats(y_all, score_nn, k)
        print('  top-%.1f%% : rules mean=%.2f std=%.2f (Delta%.1f%%) | NN mean=%.2f std=%.2f (Delta%.1f%%)' % (
            k * 100, m_r, s_r, (m_r - base_m) / base_m * 100,
            m_n, s_n, (m_n - base_m) / base_m * 100))
    print()
    return {
        'baseline_mean': base_m, 'baseline_std': base_s, 'n': len(y_all),
        'top1pct': {'rules': topk_stats(y_all, score_rules, k1), 'nn': topk_stats(y_all, score_nn, k1)},
        'top5pct': {'rules': topk_stats(y_all, score_rules, k2), 'nn': topk_stats(y_all, score_nn, k2)},
    }


@torch.no_grad()
def nn_soft(model, X, y_mean, y_std, batch=2048):
    model.to(DEVICE)
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.FloatTensor(X[i:i + batch]).to(DEVICE)
        o = model(xb)
        out.append((o['reg'] if isinstance(o, dict) else o).cpu().numpy().ravel())
    return np.concatenate(out) * y_std + y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k1', type=float, default=0.01)
    ap.add_argument('--k2', type=float, default=0.05)
    ap.add_argument('--n-sample', type=int, default=0)
    args = ap.parse_args()
    results = {}

    # ---------- Ascon (73-dim milp_linear rule features; NN = AsconMLP 216-dim) ----------
    sys.path.insert(0, os.path.join(REPO_ROOT, 'train', 'ascon'))
    from ascon_model import AsconMLP
    dv = np.load(os.path.join(DATA_DIR, 'ascon', 'ascon_features_v2_full.npz'), allow_pickle=True)
    dm = np.load(os.path.join(DATA_DIR, 'ascon', 'ascon_features_milp_full.npz'), allow_pickle=True)
    y = dv['targets'].astype(float)
    Xv = dv['features'].astype(np.float32)
    Xm = dm['features'].astype(np.float32)
    rng = np.random.default_rng(0)
    if len(y) > 200000:
        idx = rng.choice(len(y), 200000, replace=False)
        y, Xv, Xm = y[idx], Xv[idx], Xm[idx]
    ckpt = torch.load(os.path.join(DATA_DIR, 'ascon', 'ascon_model_mlp_v2.pt'), map_location='cpu', weights_only=False)
    sc = StandardScaler()
    sc.mean_, sc.scale_ = np.array(ckpt['scaler_mean']), np.array(ckpt['scaler_scale'])
    m = AsconMLP(input_dim=int(ckpt['scalar_dim']))
    m.load_state_dict(ckpt['model_state'])
    soft = nn_soft(m, sc.transform(Xv).astype(np.float32), float(ckpt['y_mean']), float(ckpt['y_std']))
    tr, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42)
    scm = StandardScaler().fit(Xm[tr])
    ridge = Ridge(alpha=5.0).fit(scm.transform(Xm[tr]), soft[tr])
    score_r = ridge.predict(scm.transform(Xm))
    results['ascon'] = report('Ascon', y, score_r, soft, args.k1, args.k2)

    # ---------- SHA3-512 (stageB pi1 features; NN = MLP_stageB, same scope as the paper rules) ----------
    sys.path.insert(0, os.path.join(REPO_ROOT, 'train', 'sha3_512'))
    from train_stageB_512 import MLP_stageB
    d = np.load(os.path.join(DATA_DIR, 'sha3_512_train_stageB_200000.npz'), allow_pickle=True)
    X, y = d['X'].astype(np.float32), d['y'].astype(float)
    rng = np.random.default_rng(0)
    if len(y) > 200000:
        idx = rng.choice(len(y), 200000, replace=False)
        X, y = X[idx], y[idx]
    ckpt = torch.load(os.path.join(DATA_DIR, 'sha3_512', 'model_mlp_stageB.pt'), map_location='cpu', weights_only=False)
    m = MLP_stageB()
    m.load_state_dict(ckpt['model_state_dict'])
    soft = nn_soft(m, X, float(ckpt['y_mean']), float(ckpt['y_std']))
    tr, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42)
    scm = StandardScaler().fit(X[tr])
    ridge = Ridge(alpha=10.0).fit(scm.transform(X[tr]), soft[tr])
    score_r = ridge.predict(scm.transform(X))
    results['sha3_512'] = report('SHA3-512(stageB)', y, score_r, soft, args.k1, args.k2)

    # ---------- SHA3-384 (stageB pi1 features; NN = MLP_stageB) ----------
    sys.path.insert(0, os.path.join(REPO_ROOT, 'train', 'sha3_384'))
    from train_stageB import MLP_stageB
    d = np.load(os.path.join(DATA_DIR, 'sha3_384_train_stageB_pi1.npz'), allow_pickle=True)
    X, y = d['X'].astype(np.float32), d['y'].astype(float)
    rng = np.random.default_rng(0)
    if len(y) > 200000:
        idx = rng.choice(len(y), 200000, replace=False)
        X, y = X[idx], y[idx]
    ckpt = torch.load(os.path.join(DATA_DIR, 'sha3_384', 'model_mlp_stageB.pt'), map_location='cpu', weights_only=False)
    m = MLP_stageB()
    m.load_state_dict(ckpt['model_state'])
    soft = nn_soft(m, X, float(ckpt['y_mean']), float(ckpt['y_std']))
    tr, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42)
    scm = StandardScaler().fit(X[tr])
    ridge = Ridge(alpha=10.0).fit(scm.transform(X[tr]), soft[tr])
    score_r = ridge.predict(scm.transform(X))
    results['sha3_384'] = report('SHA3-384(stageB)', y, score_r, soft, args.k1, args.k2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print('[saved]', OUT)


if __name__ == '__main__':
    main()
