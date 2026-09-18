# -*- coding: utf-8 -*-
"""
distill/proxy_metrics_sd.py -- scheme-disjoint proxy-metric evaluation (paper fix
for the in-sample evaluation-pool concern raised in the full review).

What this script does differently from proxy_metrics.py
-------------------------------------------------------
proxy_metrics.py ranked the *entire* evaluation pool (which overlaps the ridge
fitting samples and the QP training samples) and reported top-1%/top-5% N_mix
improvements.  Here the pool is split at the level of *blue schemes*:

  * every sample is assigned to the blue scheme that generated it
    (recovered from the data, see below);
  * schemes are partitioned into train / test (default 85:15, seedable);
  * the QP (NN) is re-trained on train-scheme samples only (internal val subset
    used for early stopping), never seeing test-scheme samples;
  * the ridge rule surrogate is fitted on train-scheme samples only;
  * baseline statistics, top-1% / top-5% statistics and Spearman correlations
    (fidelity = rules vs QP soft labels, proxy = rules vs raw labels,
    QP = QP vs raw labels) are all computed on the test-scheme samples only.

Blue-scheme recovery per target
-------------------------------
  sha3_512 : X[:, 256:512]   (initial blue mask, 256 bits; layout as in
             train/sha3_512/train_stageB_512.py)
  sha3_384 : X[:, 320:640]   (initial blue mask, 320 bits; layout as in
             train/sha3_384/train_stageB.py)
  ascon    : ascon_train_<n>.npz b[:, :160] (initial blue slice; must have
             the same row count as ascon_features_v2_full.npz)

Usage
-----
  python distill/proxy_metrics_sd.py --target all --protocol grouped \
         --data-dir <repo>/data
  python distill/proxy_metrics_sd.py --target sha3_384 --diagnose
  python distill/proxy_metrics_sd.py --target all --protocol grouped \
         --max-train-samples 200000   # fast iteration

Output: JSON written to --out (default <repo>/proxy_metrics_scheme_disjoint.json).
Console prints one table row per (target, protocol) for copy into the paper.
"""
import argparse
import glob
import copy
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# data loading and scheme recovery (no heavy deps)
# ---------------------------------------------------------------------------

def _row_keys(mat):
    return np.array([r.tobytes() for r in mat])


def load_target(name, data_dir):
    """Return dict with X_nn, X_rules (or None), y, scheme (keys or None)."""
    if name == 'sha3_512':
        p = os.path.join(data_dir, 'sha3_512_train_stageB_200000.npz')
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        d = np.load(p)
        X = d['X'].astype(np.float32)
        y = d['y'].astype(np.float32)
        keys = _row_keys(X[:, 256:512])
        return {'X_nn': X, 'X_rules': X, 'y': y, 'name': name,
                'scheme': keys, 'alpha': 10.0, 'rank_w': 0.3,
                'hidden': (512, 256, 128), 'dropout': 0.3, 'batch': 128}
    if name == 'sha3_384':
        p = os.path.join(data_dir, 'sha3_384_train_stageB_pi1.npz')
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        d = np.load(p)
        X = d['X'].astype(np.float32)
        y = d['y'].astype(np.float32)
        keys = _row_keys(X[:, 320:640])
        return {'X_nn': X, 'X_rules': X, 'y': y, 'name': name,
                'scheme': keys, 'alpha': 10.0, 'rank_w': 0.3,
                'hidden': (512, 256, 128), 'dropout': 0.3, 'batch': 128}
    if name == 'ascon':
        base = os.path.join(data_dir, 'ascon')
        pv = os.path.join(base, 'ascon_features_v2_full.npz')
        pm = os.path.join(base, 'ascon_features_milp_full.npz')
        if not os.path.exists(pv) or not os.path.exists(pm):
            raise FileNotFoundError(f'{pv} or {pm}')
        dv = np.load(pv, allow_pickle=True)
        dm = np.load(pm, allow_pickle=True)
        Xv = dv['features'].astype(np.float32)
        Xm = dm['features'].astype(np.float32)
        y = dv['targets'].astype(np.float32)
        scheme = None
        cands = sorted(glob.glob(os.path.join(base, 'ascon_train_*.npz')))
        rp = cands[-1] if cands else os.path.join(base, 'ascon_train_1000000.npz')
        if os.path.exists(rp):
            dt = np.load(rp)
            bb = dt['b']
            if len(bb) == len(y):
                scheme = _row_keys(np.ascontiguousarray(bb[:, :160]))
            else:
                print(f"  [ascon] WARNING raw rows {len(bb)} != feature rows "
                      f"{len(y)}; scheme recovery disabled")
        return {'X_nn': Xv, 'X_rules': Xm, 'y': y, 'name': name,
                'scheme': scheme, 'alpha': 5.0, 'rank_w': 0.3,
                'hidden': (256, 128, 64), 'dropout': 0.2, 'batch': 128}
    raise ValueError(name)


def scheme_stats(keys, tag):
    _, counts = np.unique(keys, return_counts=True)
    order = np.sort(counts)[::-1]
    print(f"  [{tag}] samples={len(keys)} schemes={len(counts)} "
          f"per-scheme: min={counts.min()} max={counts.max()} "
          f"median={int(np.median(counts))} "
          f"top={list(order[:6].astype(int))}")


def make_split(scheme, n, protocol, seed, test_frac):
    """Return (train_idx, test_idx) for the given protocol.

    random : sample-level 85:15 shuffle (same protocol as the old tables).
    grouped: schemes are split 85:15; samples of a scheme stay together.
    intra  : every scheme is split internally 85:15 (no scheme is held out).
    """
    rng = np.random.RandomState(seed)
    if protocol == 'intra':
        if scheme is None:
            raise RuntimeError('scheme recovery unavailable')
        uniq = np.unique(scheme)
        tr, te = [], []
        for u in uniq:
            idx = np.where(scheme == u)[0]
            rng.shuffle(idx)
            k = max(1, int(round(len(idx) * (1 - test_frac))))
            tr.append(idx[:k]); te.append(idx[k:])
        return np.concatenate(tr), np.concatenate(te)
    if protocol == 'grouped':
        if scheme is None:
            raise RuntimeError('scheme recovery unavailable for this target')
        uniq = np.unique(scheme)
        perm = rng.permutation(len(uniq))
        ntest_s = max(1, int(round(len(uniq) * test_frac)))
        test_s = set(uniq[perm[:ntest_s]].tolist())
        tr = np.array([i for i in range(n) if scheme[i] not in test_s])
        te = np.array([i for i in range(n) if scheme[i] in test_s])
        return tr, te
    # random (sample-level)
    perm = rng.permutation(n)
    ntest = max(1, int(round(n * test_frac)))
    return perm[ntest:], perm[:ntest]


# ---------------------------------------------------------------------------
# training + evaluation (torch / sklearn are imported lazily)
# ---------------------------------------------------------------------------

def train_qp(X_tr, y_tr, X_va, y_va, hidden, dropout, rank_w,
             epochs, patience, batch, lr=1e-3, seed=42):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _MLP(nn.Module):
        def __init__(self, input_dim, hidden=hidden, dropout=dropout):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                           nn.Dropout(dropout)]
                prev = h
            self.encoder = nn.Sequential(*layers)
            self.reg = nn.Sequential(nn.Linear(prev, 32), nn.ReLU(),
                                     nn.Linear(32, 1))
            self.rank = nn.Sequential(nn.Linear(prev, 32), nn.ReLU(),
                                      nn.Linear(32, 1))

        def forward(self, x, _=None):
            h = self.encoder(x)
            return {'reg': self.reg(h).squeeze(-1),
                    'rank': self.rank(h).squeeze(-1)}

    def rank_loss(pred, target, margin=2.0):
        n = pred.shape[0]
        half = max(1, n // 2)
        i1 = torch.randperm(n, device=pred.device)[:half]
        i2 = torch.randperm(n, device=pred.device)[:half]
        pi, pj = pred[i1], pred[i2]
        ti, tj = target[i1], target[i2]
        valid = ti < tj
        if valid.sum() == 0:
            return torch.zeros((), device=pred.device)
        return torch.clamp(margin - (pj - pi), min=0).mean()

    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_tr_n = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_va_n = ((y_va - y_mean) / y_std).astype(np.float32)

    model = _MLP(X_tr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    tr_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_n))
    va_ds = TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va_n))
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch * 2, shuffle=False)

    best_loss, best_state, counter = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out['reg'], yb) + rank_w * rank_loss(out['rank'], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        vloss, preds = 0.0, []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(device)
                o = model(xb)
                vloss += crit(o['reg'], yb.to(device)).item() * len(xb)
                preds.append(o['reg'].cpu().numpy())
        vloss /= len(va_ds)
        if vloss < best_loss - 1e-4:
            best_loss = vloss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break
        sched.step()
    model.load_state_dict(best_state)
    return model, scaler, y_mean, y_std, device


def _predict(model, scaler, X, y_mean, y_std, device, batch=4096):
    import torch
    Xs = scaler.transform(X).astype(np.float32)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch):
            xb = torch.FloatTensor(Xs[i:i + batch]).to(device)
            o = model(xb)
            out.append(o['reg'].cpu().numpy().ravel())
    return np.concatenate(out) * y_std + y_mean


def _topk(y, score, k):
    n = max(1, int(len(y) * k))
    idx = np.argsort(score)[:n]
    sel = y[idx]
    return {'mean': float(sel.mean()), 'std': float(sel.std()),
            'n': int(len(sel)), 'rel': float((sel.mean() - y.mean()) / y.mean())}


def evaluate_one(cfg, protocol, seed, test_frac, epochs, patience,
                 max_train_samples, out_devices=None):
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    name = cfg['name']
    X_nn, X_rules, y = cfg['X_nn'], cfg['X_rules'], cfg['y']
    scheme = cfg['scheme']
    n = len(y)
    if scheme is not None:
        scheme_stats(scheme, name)
    tr, te = make_split(scheme, n, protocol, seed, test_frac)
    if max_train_samples and len(tr) > max_train_samples:
        rng = np.random.RandomState(seed)
        tr = np.sort(rng.choice(tr, max_train_samples, replace=False))
    rng = np.random.RandomState(seed + 1)
    perm = rng.permutation(len(tr))
    nva = max(1, int(round(len(tr) * 0.1)))
    va_i, tr_i = perm[:nva], perm[nva:]
    tr_i = np.sort(tr[tr_i])
    va_i = np.sort(tr[va_i])
    print(f"  train samples={len(tr_i):,}  val={len(va_i):,}  "
          f"test={len(te):,}  (protocol={protocol})")
    if len(te) < 50:
        print("  !! test partition very small; results will be noisy")

    model, scaler, y_mean, y_std, device = train_qp(
        X_nn[tr_i], y[tr_i], X_nn[va_i], y[va_i],
        hidden=cfg['hidden'], dropout=cfg['dropout'], rank_w=cfg['rank_w'],
        epochs=epochs, patience=patience, batch=cfg['batch'])
    soft_te = _predict(model, scaler, X_nn[te], y_mean, y_std, device)
    qp_sp = float(spearmanr(soft_te, y[te]).statistic)

    soft_tr = _predict(model, scaler, X_nn[tr_i], y_mean, y_std, device)
    sc = StandardScaler()
    sc.fit(X_rules[tr_i])
    ridge = Ridge(alpha=cfg['alpha']).fit(sc.transform(X_rules[tr_i]),
                                          soft_tr)
    score_te = ridge.predict(sc.transform(X_rules[te]))
    fid_sp = float(spearmanr(score_te, soft_te).statistic)
    proxy_sp = float(spearmanr(score_te, y[te]).statistic)

    res = {
        'target': name, 'protocol': protocol, 'seed': seed,
        'n_schemes': int(len(np.unique(scheme))) if scheme is not None else None,
        'n_train': int(len(tr_i)), 'n_val': int(len(va_i)),
        'n_test': int(len(te)),
        'baseline': {'mean': float(y[te].mean()), 'std': float(y[te].std()),
                     'n': int(len(te))},
        'qp': {'spearman_raw': qp_sp},
        'rules': {'fidelity': fid_sp, 'proxy': proxy_sp},
        'top1pct': {'rules': _topk(y[te], score_te, 0.01),
                    'qp': _topk(y[te], soft_te, 0.01)},
        'top5pct': {'rules': _topk(y[te], score_te, 0.05),
                    'qp': _topk(y[te], soft_te, 0.05)},
    }
    print(f"  baseline={res['baseline']['mean']:.2f}+/-"
          f"{res['baseline']['std']:.2f} (n={res['baseline']['n']})")
    print(f"  fidelity={fid_sp:.4f} proxy={proxy_sp:.4f} QP={qp_sp:.4f}")
    for k, tag in ((0.01, 'top1%'), (0.05, 'top5%')):
        r = res['top1pct' if k == 0.01 else 'top5pct']
        print(f"  {tag}: rules {r['rules']['mean']:.2f} "
              f"({r['rules']['rel'] * 100:+.1f}%)  "
              f"QP {r['qp']['mean']:.2f} ({r['qp']['rel'] * 100:+.1f}%)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='all',
                    choices=['sha3_512', 'sha3_384', 'ascon', 'all'])
    ap.add_argument('--protocol', default='grouped',
                    choices=['grouped', 'intra', 'random'])
    ap.add_argument('--data-dir', default=os.path.join(REPO_ROOT, 'data'))
    ap.add_argument('--test-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--max-train-samples', type=int, default=0,
                    help='cap of training samples (0 = unlimited)')
    ap.add_argument('--diagnose', action='store_true',
                    help='print dataset/scheme diagnostics and exit')
    ap.add_argument('--out', default=os.path.join(
        REPO_ROOT, 'proxy_metrics_scheme_disjoint.json'))
    args = ap.parse_args()

    targets = ([args.target] if args.target != 'all'
               else ['sha3_512', 'sha3_384', 'ascon'])
    results = []
    for t in targets:
        print(f"\n===== {t} =====")
        try:
            cfg = load_target(t, args.data_dir)
        except Exception as exc:
            print(f"  load failed: {exc}")
            continue
        if cfg['scheme'] is not None:
            scheme_stats(cfg['scheme'], t)
        else:
            print(f"  [{t}] no scheme recovery available for this target")
        if args.diagnose:
            continue
        if args.protocol in ('grouped', 'intra') and cfg['scheme'] is None:
            print(f"  skipped ({args.protocol} needs scheme recovery)")
            continue
        t0 = time.time()
        try:
            res = evaluate_one(cfg, args.protocol, args.seed, args.test_frac,
                               args.epochs, args.patience,
                               args.max_train_samples)
        except Exception as exc:
            print(f"  evaluation failed: {exc}")
            import traceback
            traceback.print_exc()
            continue
        res['time_s'] = round(time.time() - t0, 1)
        results.append(res)
    if results:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print('\n[saved]', args.out)


if __name__ == '__main__':
    main()
