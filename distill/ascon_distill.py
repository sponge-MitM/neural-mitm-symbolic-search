"""
distill/ascon_distill.py - Ascon distillation rules (Ridge fitted to the QP soft labels)

Soft label source (NN2, AsconMLP): input 216-dim statistical features, predicts u2
  - Data: ascon_features_v2_full.npz (1,000,000 default, 216 feat_names)
  - Model: ascon_model_mlp_v2.pt (AsconMLP, scaler_mean/scale + y_mean/y_std)

Fitting: Ridge vs soft labels, features = 73-dim milp_linear embedded features (ascon_features_milp_full.npz)
  -> Spearman(rules, soft labels) [distillation fidelity] + Spearman(rules, raw labels) [proxy quality]
  -> rule table (73 feature names, all MILP-linearly-expressible)
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

import torch  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, 'distill', 'rules')
DATA_DIR = os.path.join(REPO_ROOT, 'data')

FEAT_V2 = os.path.join(DATA_DIR, 'ascon', 'ascon_features_v2_full.npz')
FEAT_MILP = os.path.join(DATA_DIR, 'ascon', 'ascon_features_milp_full.npz')
CKPT = os.path.join(DATA_DIR, 'ascon', 'ascon_model_mlp_v2.pt')
MODEL_DIR = os.path.join(REPO_ROOT, 'train', 'ascon')


def load_nn():
    sys.path.insert(0, MODEL_DIR)
    from ascon_model import AsconMLP
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = AsconMLP(input_dim=int(ckpt['scalar_dim']))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt['scaler_mean'])
    scaler.scale_ = np.array(ckpt['scaler_scale'])
    return model, scaler, float(ckpt['y_mean']), float(ckpt['y_std'])


@torch.no_grad()
def predict_soft(model, X_s, y_mean, y_std, batch_size=8192):
    model.to(DEVICE)
    preds = []
    for i in range(0, len(X_s), batch_size):
        xb = torch.FloatTensor(X_s[i:i + batch_size]).to(DEVICE)
        out = model(xb)
        pred = out['reg'] if isinstance(out, dict) else out
        preds.append(pred.cpu().numpy().ravel())
    return np.concatenate(preds) * y_std + y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-sample', type=int, default=0)
    args = ap.parse_args()

    print(f"[Ascon distill] device={DEVICE}")
    d_v2 = np.load(FEAT_V2, allow_pickle=True)
    d_milp = np.load(FEAT_MILP, allow_pickle=True)
    X_v2 = d_v2['features'].astype(np.float32)
    X_milp = d_milp['features'].astype(np.float32)
    y_raw = d_v2['targets'].astype(float)
    names_milp = [str(n) for n in d_milp['feat_names']]
    assert len(X_v2) == len(X_milp) == len(y_raw)
    print(f"  data: v2 features{X_v2.shape}, milp features{X_milp.shape}, y[{y_raw.min()}..{y_raw.max()}]")

    if args.n_sample and args.n_sample < len(y_raw):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_raw), args.n_sample, replace=False)
        X_v2, X_milp, y_raw = X_v2[idx], X_milp[idx], y_raw[idx]

    model, scaler, y_mean, y_std = load_nn()
    print(f"  model: AsconMLP, y_mean={y_mean:.2f}, y_std={y_std:.2f}")
    X_s = scaler.transform(X_v2).astype(np.float32)
    print("  generating soft labels (CUDA)...")
    y_soft = predict_soft(model, X_s, y_mean, y_std)

    idx_all = np.arange(len(y_raw))
    tr, te = train_test_split(idx_all, test_size=0.15, random_state=args.seed)
    tr, va = train_test_split(tr, test_size=0.15 / 0.85, random_state=args.seed)
    # NN2 Spearman reported on the test split (avoid training leakage)
    sp_model = float(spearmanr(y_soft[te], y_raw[te]).statistic)
    print(f"  NN2 Spearman(soft, raw) [test] = {sp_model:.4f}")

    scaler_m = StandardScaler()
    Xm_tr = scaler_m.fit_transform(X_milp[tr])
    Xm_te = scaler_m.transform(X_milp[te])
    ridge = Ridge(alpha=args.alpha)
    ridge.fit(Xm_tr, y_soft[tr])
    preds_te = ridge.predict(Xm_te)
    sp_soft = float(spearmanr(preds_te, y_soft[te]).statistic)
    sp_raw = float(spearmanr(preds_te, y_raw[te]).statistic)
    print(f"  [Ridge vs soft]  Spearman(vs soft)={sp_soft:.4f}  (distillation fidelity)")
    print(f"  [Ridge vs soft]  Spearman(vs raw)={sp_raw:.4f}  (proxy quality)")

    rows = []
    for i, n in enumerate(names_milp):
        w = float(ridge.coef_[i])
        if abs(w) < 1e-12:
            continue
        rows.append({'name': n, 'w': w, 'max_val': float(np.max(np.abs(X_milp[te, i])))})
    rows.sort(key=lambda r: -abs(r['w']))
    total = sum(abs(r['w']) for r in rows) or 1e-8
    table = []
    for r in rows[:15]:
        sign = -1 if r['w'] < 0 else 1
        table.append({'name': r['name'], 'sign': sign,
                      'direction': 'MAXIMIZE' if sign == -1 else 'MINIMIZE',
                      'w_norm': round(abs(r['w']) / total, 6),
                      'w_raw': round(r['w'], 6),
                      'max_val': round(r['max_val'], 4)})
    print(f"  Top-10 rules:")
    for r in table[:10]:
        print(f"    {r['name']:30s} {r['direction']:8s} w={r['w_norm']:.6f} max={r['max_val']}")

    payload = {
        'primitive': 'ascon', 'stage': 2,
        'ckpt': 'see README (checkpoints not distributed)',
        'soft_label_source': 'AsconMLP',
        'nn_spearman_vs_raw': sp_model,
        'ridge': {'spearman_vs_soft': sp_soft, 'spearman_vs_raw': sp_raw},
        'alpha': args.alpha, 'n_rules': len(table), 'rules': table,
    }
    out = os.path.join(OUT_DIR, 'rules_ascon.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[done] rules -> {out}")


if __name__ == '__main__':
    main()
