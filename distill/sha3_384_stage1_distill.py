"""
SHA3-384 Stage-1 rule distillation (Ridge fitted to the NN1 soft labels).

Soft-label source (NN1, BlueMLP): 33-dim aggregate features (SHA3-384, 5 columns) of the blue scheme,
predicting the minimal reachable Stage-2 N_mix (min_u2).
  - Data:  sha3_384_blue_labels_*_features.npy (7936 blue schemes x 31 features)
  - Model: model_blue_mlp.pt (BlueMLP, dual-head)

Fitting: ridge vs soft labels over the MILP-expressible features, after removing
exact duplicates (n_blue == init_blue_total, blue_x{k}_count == init_blue_x{k});
active_z features (blue_z_active, pi1_blue_active_z) are excluded from the embedded
rule set because they require OR-variable linearization.
  -> Spearman(rules, soft) [distillation fidelity]
  -> Spearman(rules, raw)  [proxy quality]

Usage:
  python sha3_stage1_distill.py [--alpha 10.0]
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

DATA_F = os.path.join(DATA_DIR, 'sha3_384', 'sha3_384_blue_labels_8000_t5_features.npy')
DATA_L = os.path.join(DATA_DIR, 'sha3_384', 'sha3_384_blue_labels_8000_t5_labels.npy')
CKPT = os.path.join(DATA_DIR, 'sha3_384', 'model_blue_mlp.pt')
MODEL_DIR = os.path.join(REPO_ROOT, 'train', 'sha3_384')

# Features not exactly expressible in the MILP (excluded from the embedded rule set)
EXCLUDE_SUBSTR = ('std', 'volatility', 'adj', 'over_', 'ratio', 'max_per_z')


def load_blue_model():
    sys.path.insert(0, MODEL_DIR)
    from sha3_train_blue import BlueMLP
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = BlueMLP(input_dim=int(ckpt['scalar_dim']))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt['scaler_mean'])
    scaler.scale_ = np.array(ckpt['scaler_scale'])
    return model, scaler, float(ckpt['y_mean']), float(ckpt['y_std'])


@torch.no_grad()
def predict_soft(model, X_s, y_mean, y_std, batch_size=8192):
    model.to(DEVICE)
    model.eval()
    preds = []
    for i in range(0, len(X_s), batch_size):
        xb = torch.FloatTensor(X_s[i:i + batch_size]).to(DEVICE)
        out = model(xb)
        p = out['reg'] if isinstance(out, dict) else out
        preds.append(p.cpu().numpy())
    return np.concatenate(preds) * y_std + y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--top-n', type=int, default=15)
    args = ap.parse_args()

    print(f"[SHA3-512 stage1 distill (NN1 BlueMLP)] device={DEVICE}")
    X_raw = np.load(DATA_F).astype(np.float32)
    y_raw = np.load(DATA_L).astype(float)
    print(f"  data: X{X_raw.shape}, min_u2[{y_raw.min():.0f}..{y_raw.max():.0f}], "
          f"mean={y_raw.mean():.2f}")

    model, scaler, y_mean, y_std = load_blue_model()
    X_s = scaler.transform(X_raw).astype(np.float32)
    print("  generating soft labels (CUDA)...")
    y_soft = predict_soft(model, X_s, y_mean, y_std)

    idx_all = np.arange(len(y_raw))
    tr, te = train_test_split(idx_all, test_size=0.15, random_state=args.seed)
    tr, va = train_test_split(tr, test_size=0.15 / 0.85, random_state=args.seed)
    # NN Spearman reported on the test split (avoid training leakage)
    sp_model = float(spearmanr(y_soft[te], y_raw[te]).statistic)
    print(f"  NN1 Spearman(soft, raw) [test] = {sp_model:.4f}")

    meta_path = DATA_F.replace('_features.npy', '_meta.json')
    with open(meta_path) as f:
        feat_names = json.load(f)['feature_names']
    print(f"  features: {len(feat_names)}")

    # MILP-expressible whitelist
    keep = [i for i, n in enumerate(feat_names)
            if not any(s in n for s in EXCLUDE_SUBSTR)]
    keep_names = [feat_names[i] for i in keep]

    # Deduplicate exact aliases (n_blue == init_blue_total, blue_x{k} == init_blue_x{k})
    dup_aliases = {'n_blue', 'blue_x0_count', 'blue_x1_count',
                   'blue_x2_count', 'blue_x3_count', 'blue_x4_count'}
    dedup = [i for i in keep if feat_names[i] not in dup_aliases]
    keep_names = [feat_names[i] for i in dedup]
    print(f"  MILP-expressible: {len(keep)}/{len(feat_names)} "
          f"(after dedup {len(dedup)}): {keep_names}")
    keep = dedup

    scaler2 = StandardScaler()
    X_tr_s = scaler2.fit_transform(X_raw[tr][:, keep])
    X_te_s = scaler2.transform(X_raw[te][:, keep])

    ridge = Ridge(alpha=args.alpha)
    ridge.fit(X_tr_s, y_soft[tr])
    preds_te = ridge.predict(X_te_s)
    sp_soft = float(spearmanr(preds_te, y_soft[te]).statistic)
    sp_raw = float(spearmanr(preds_te, y_raw[te]).statistic)
    print(f"  [Ridge vs soft]  Spearman(vs soft)={sp_soft:.4f}  (distillation fidelity)")
    print(f"  [Ridge vs soft]  Spearman(vs raw)={sp_raw:.4f}  (proxy quality)")

    rows = []
    for i, name in enumerate(keep_names):
        w = float(ridge.coef_[i])
        if abs(w) < 1e-12:
            continue
        max_val = float(np.max(np.abs(X_raw[te][:, keep][:, i])))
        rows.append({'name': name, 'agg_w': w, 'max_val': max_val})
    rows.sort(key=lambda r: -abs(r['agg_w']))
    total = sum(abs(r['agg_w']) for r in rows) or 1e-8
    table = []
    for r in rows[:args.top_n]:
        sign = -1 if r['agg_w'] < 0 else 1
        table.append({'name': r['name'], 'sign': sign,
                      'direction': 'MAXIMIZE' if sign == -1 else 'MINIMIZE',
                      'w_norm': round(abs(r['agg_w']) / total, 6),
                      'w_agg': round(r['agg_w'], 6),
                      'max_val': round(r['max_val'], 4)})
    print(f"  Top-{min(10, len(table))} rules:")
    for r in table[:10]:
        print(f"    {r['name']:24s} {r['direction']:8s} w={r['w_norm']:.6f} "
              f"max={r['max_val']}")

    payload = {
        'primitive': 'sha3_384', 'stage': 1,
        'ckpt': 'see README (checkpoints not distributed)',
        'soft_label_source': 'BlueMLP_NN1',
        'nn_spearman_vs_raw': sp_model,
        'ridge': {'spearman_vs_soft': sp_soft, 'spearman_vs_raw': sp_raw},
        'alpha': args.alpha, 'n_rules': len(table), 'rules': table,
        'feature_names': keep_names,
    }
    out = os.path.join(OUT_DIR, 'rules_sha3_384_stage1.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[done] rules -> {out}")


if __name__ == '__main__':
    main()
