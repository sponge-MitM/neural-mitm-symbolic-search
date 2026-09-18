"""
distill/sha3_distill.py - SHA3-384 / SHA3-512 two-stage distillation rules
(Ridge fitted to the QP soft labels)

Soft label source (NN2, stageB model): input init + pi1 mask, predicts u2
  - SHA3-512: MLP_stageB (3712) @ model_mlp_stageB.pt, data sha3_512_train_stageB_200000.npz
  - SHA3-384: MLP_stageB (3840) @ model_mlp_stageB.pt, data sha3_384_train_stageB_pi1.npz

stage2 rules: Ridge fit on all pi1 features (init + pi1_r + pi1_b) vs soft labels
  -> Spearman(rules, soft labels) [distillation fidelity] + Spearman(rules, raw labels) [proxy quality]
stage1 rules: Ridge fit on the blue-feature subset (init_blue + pi1_blue) vs soft labels -> blue rule table

Rule table: generated from coefficients aggregated per aggregate feature (y row / x column / total), aligned with the attack script's feature_map.

Usage:
  python distill/sha3_distill.py --variant 512 --stage 2
  python distill/sha3_distill.py --variant 384 --stage 1
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

# Feature layout (z-major)
# 512: init_r[0:256] init_b[256:512] pi1_r[512:2112] pi1_b[2112:3712]
# 384: init_r[0:320] init_b[320:640] pi1_r[640:2240] pi1_b[2240:3840]

VARIANTS = {
    512: dict(init_x=4, data=os.path.join(DATA_DIR, 'sha3_512_train_stageB_200000.npz'),
              ckpt=os.path.join(DATA_DIR, 'sha3_512', 'model_mlp_stageB.pt'),
              model_dir=os.path.join(REPO_ROOT, 'train', 'sha3_512'),
              st_key='model_state_dict', alpha=10.0),
    384: dict(init_x=5, data=os.path.join(DATA_DIR, 'sha3_384_train_stageB_pi1.npz'),
              ckpt=os.path.join(DATA_DIR, 'sha3_384', 'model_mlp_stageB.pt'),
              model_dir=os.path.join(REPO_ROOT, 'train', 'sha3_384'),
              st_key='model_state', alpha=10.0),
}


def load_nn1(variant, cfg):
    sys.path.insert(0, cfg['model_dir'])
    ckpt = torch.load(cfg['ckpt'], map_location='cpu', weights_only=False)
    if variant == 512:
        from train_stageB_512 import MLP_stageB
    else:
        from train_stageB import MLP_stageB
    model = MLP_stageB()
    model.load_state_dict(ckpt[cfg['st_key']])
    model.eval()
    scaler = None
    if 'scaler_mean' in ckpt and 'scaler_scale' in ckpt:
        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt['scaler_mean'])
        scaler.scale_ = np.array(ckpt['scaler_scale'])
    return model, scaler, float(ckpt['y_mean']), float(ckpt['y_std'])


@torch.no_grad()
def predict_soft(model, X, y_mean, y_std, batch_size=2048):
    model.to(DEVICE)
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
        out = model(xb)['reg'].cpu().numpy()
        preds.append(out)
    return np.concatenate(preds) * y_std + y_mean


def layout(variant):
    """Return (init_red_slice, init_blue_slice, pi1_red_slice, pi1_blue_slice, nz, ix)."""
    ix = VARIANTS[variant]['init_x']
    d_init = 64 * ix
    return (slice(0, d_init), slice(d_init, 2 * d_init),
            slice(2 * d_init, 2 * d_init + 1600), slice(2 * d_init + 1600, 2 * d_init + 3200),
            64, ix)


def agg_defs(stage, variant):
    """Aggregate feature definitions (based on the 3712/3840-dim coefficient indices)."""
    ir, ib, p1r, p1b, nz, ix = layout(variant)
    defs = []
    if stage == 2:
        for j in range(ix):
            defs.append({'name': f'init_red_x{j}', 'slices': [slice(ir.start + j, ir.stop, ix)]})
            defs.append({'name': f'init_blue_x{j}', 'slices': [slice(ib.start + j, ib.stop, ix)]})
        for y in range(5):
            sl_r = [p1r.start + z * 25 + y * 5 + x for z in range(64) for x in range(5)]
            sl_b = [p1b.start + z * 25 + y * 5 + x for z in range(64) for x in range(5)]
            defs.append({'name': f'pi1_red_y{y}', 'slices': [sl_r]})
            defs.append({'name': f'pi1_blue_y{y}', 'slices': [sl_b]})
        defs.append({'name': 'pi1_red_total', 'slices': [p1r]})
        defs.append({'name': 'pi1_blue_total', 'slices': [p1b]})
    else:
        defs.append({'name': 'init_blue_total', 'slices': [ib]})
        for j in range(ix):
            defs.append({'name': f'init_blue_x{j}', 'slices': [slice(ib.start + j, ib.stop, ix)]})
        for y in range(5):
            sl_b = [p1b.start + z * 25 + y * 5 + x for z in range(64) for x in range(5)]
            defs.append({'name': f'pi1_blue_y{y}', 'slices': [sl_b]})
        defs.append({'name': 'pi1_blue_total', 'slices': [p1b]})
    return defs


def build_rule_table(coef, defs, F_te, top_n=15):
    rows = []
    for i, d in enumerate(defs):
        w = 0.0
        for sl in d['slices']:
            w += float(np.sum(coef[sl]))
        if abs(w) < 1e-12:
            continue
        rows.append({'name': d['name'], 'agg_w': w,
                     'max_val': float(np.max(np.abs(F_te[:, i])))})
    rows.sort(key=lambda r: -abs(r['agg_w']))
    total = sum(abs(r['agg_w']) for r in rows) or 1e-8
    table = []
    for r in rows[:top_n]:
        sign = -1 if r['agg_w'] < 0 else 1
        table.append({'name': r['name'], 'sign': sign,
                      'direction': 'MAXIMIZE' if sign == -1 else 'MINIMIZE',
                      'w_norm': round(abs(r['agg_w']) / total, 6),
                      'w_agg': round(r['agg_w'], 6),
                      'max_val': round(r['max_val'], 4)})
    return table


def compute_agg(X, defs):
    F = np.zeros((X.shape[0], len(defs)), dtype=np.float64)
    for i, d in enumerate(defs):
        acc = np.zeros(X.shape[0], dtype=np.float64)
        for sl in d['slices']:
            acc += X[:, sl].sum(axis=1)
        F[:, i] = acc
    return F


def feature_slices(stage, variant):
    """Feature columns to fit. stage2: all features; stage1: init_blue + pi1_blue."""
    ir, ib, p1r, p1b, nz, ix = layout(variant)
    if stage == 2:
        cols = np.arange(ir.stop + 3200)
    else:
        cols = np.concatenate([np.arange(ib.start, ib.stop), np.arange(p1b.start, p1b.stop)])
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', type=int, choices=[384, 512], default=512)
    ap.add_argument('--stage', type=int, choices=[1, 2], default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-sample', type=int, default=0, help='0 = all samples')
    args = ap.parse_args()
    if args.stage == 1:
        ap.error('the Stage-1 (blue-scheme) tables are produced by '
                 'distill/sha3_stage1_distill.py (SHA3-512) and '
                 'distill/sha3_384_stage1_distill.py (SHA3-384), which fit the NN1 '
                 'soft labels; this script distils the Stage-2 joint table only. '
                 'Re-run with --stage 2.')
    cfg = VARIANTS[args.variant]
    alpha = cfg['alpha']

    print(f"[SHA3-{args.variant} stage{args.stage} distill] device={DEVICE}")
    data = np.load(cfg['data'], allow_pickle=True)
    X = data['X'].astype(np.float32)
    y_raw = data['y'].astype(float)
    print(f"  data: X{X.shape}, y[{y_raw.min()}..{y_raw.max()}], mean={y_raw.mean():.2f}")

    if args.n_sample and args.n_sample < len(y_raw):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_raw), args.n_sample, replace=False)
        X, y_raw = X[idx], y_raw[idx]

    model, scaler, y_mean, y_std = load_nn1(args.variant, cfg)
    print(f"  model: MLP_stageB, ckpt y_mean={y_mean:.2f}, y_std={y_std:.2f}")
    X_inf = scaler.transform(X).astype(np.float32) if scaler is not None else X
    print("  generating soft labels (CUDA)...")
    y_soft = predict_soft(model, X_inf, y_mean, y_std)

    idx_all = np.arange(len(y_raw))
    tr, te = train_test_split(idx_all, test_size=0.15, random_state=args.seed)
    tr, va = train_test_split(tr, test_size=0.15 / 0.85, random_state=args.seed)
    # NN Spearman reported on the test split (avoid training leakage)
    sp_model = float(spearmanr(y_soft[te], y_raw[te]).statistic)
    print(f"  NN1 Spearman(soft, raw) [test] = {sp_model:.4f}")

    cols = feature_slices(args.stage, args.variant)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[tr][:, cols])
    X_te_s = scaler.transform(X[te][:, cols])

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr_s, y_soft[tr])
    preds_te = ridge.predict(X_te_s)
    sp_soft = float(spearmanr(preds_te, y_soft[te]).statistic)
    sp_raw = float(spearmanr(preds_te, y_raw[te]).statistic)
    print(f"  [Ridge vs soft]  Spearman(vs soft)={sp_soft:.4f}  (distillation fidelity)")
    print(f"  [Ridge vs soft]  Spearman(vs raw)={sp_raw:.4f}  (proxy quality)")

    defs = agg_defs(args.stage, args.variant)
    F_te = compute_agg(X[te], defs)
    # Map coefficients back to full-feature coordinates
    coef_full = np.zeros(2 * 64 * VARIANTS[args.variant]['init_x'] + 3200)
    coef_full[cols] = ridge.coef_
    table = build_rule_table(coef_full, defs, F_te, top_n=15)
    print(f"  Top-{min(10, len(table))} rules:")
    for r in table[:10]:
        print(f"    {r['name']:22s} {r['direction']:8s} w={r['w_norm']:.6f} max={r['max_val']}")

    payload = {
        'primitive': f'sha3_{args.variant}', 'stage': args.stage,
        'ckpt': cfg['ckpt'], 'soft_label_source': 'MLP_stageB',
        'nn_spearman_vs_raw': sp_model,
        'ridge': {'spearman_vs_soft': sp_soft, 'spearman_vs_raw': sp_raw},
        'alpha': alpha, 'n_rules': len(table), 'rules': table,
    }
    out = os.path.join(OUT_DIR, f'rules_sha3_{args.variant}_stage{args.stage}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[done] rules -> {out}")


if __name__ == '__main__':
    main()
