# -*- coding: utf-8 -*-
"""Type distribution statistics of stage-B samples after two rounds of
propagation (after chi2).

Types: u(1,0,0) lr(0,1,0) ur(1,1,0) lb(0,0,1) ub(1,0,1) lg(0,1,1) ug(1,1,1) zero(0,0,0)
Two accounting conventions: relabel (true bit labels, identical to the data y)
and dependency-channel (the accounting used by the attack-script MILP).
"""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, 'data')
OUT_DIR = os.path.join(REPO, 'data', 'validation')
N_Z, N_Y, N_X = 64, 5, 5
RHO_BOX = [[0, 1, 62, 28, 27], [36, 44, 6, 55, 20],
           [3, 10, 43, 25, 39], [41, 45, 15, 21, 8],
           [18, 2, 61, 56, 14]]
RHO_IDX = np.array([[(np.arange(N_Z) - RHO_BOX[y][x]) % N_Z for x in range(N_X)]
                    for y in range(N_Y)])
PI_MAP = np.array([[(x, (x + 3 * y) % 5) for x in range(N_X)] for y in range(N_Y)])

TYPES = ['u', 'lr', 'ur', 'lb', 'ub', 'lg', 'ug', 'zero']


def propagate_relabel(init_red, init_blue, n_rounds=2):
    'Function propagate relabel.'
    N = init_red.shape[0]
    n_x = init_red.shape[2]
    state = np.zeros((N, N_Z, N_Y, N_X, 3), dtype=np.int8)
    for y in (0, 1):
        state[:, :, y, :n_x, 1] = np.where(init_red == 1, 1, 0)
        state[:, :, y, :n_x, 2] = np.where(init_blue == 1, 1, 0)

    def theta(s):
        C = s.sum(axis=2, dtype=np.int32) & 1
        Cm1 = np.take(C, [(x - 1) % 5 for x in range(N_X)], axis=2)
        Cx1 = np.roll(C, 1, axis=1)
        Cx1 = np.take(Cx1, [(x + 1) % 5 for x in range(N_X)], axis=2)
        D = Cm1 ^ Cx1
        return (s ^ D[:, :, None, :, :].astype(np.int8)) & 1

    def rho(s):
        out = np.empty_like(s)
        for y in range(N_Y):
            for x in range(N_X):
                out[:, :, y, x, :] = s[:, RHO_IDX[y, x], y, x, :]
        return out

    def pi(s):
        out = np.empty_like(s)
        for y in range(N_Y):
            for x in range(N_X):
                sy, sx = PI_MAP[y, x]
                out[:, :, y, x, :] = s[:, :, sy, sx, :]
        return out

    def chi(s):
        s1 = np.roll(s, -1, axis=3)
        s2 = np.roll(s, -2, axis=3)
        b1_0, b1_1, b1_2 = s1[..., 0], s1[..., 1], s1[..., 2]
        b2_0, b2_1, b2_2 = s2[..., 0], s2[..., 1], s2[..., 2]
        r_and = b1_1 & b2_1
        b_and = b1_2 & b2_2
        cross_ul = (b1_1 & b2_2) | (b1_2 & b2_1)
        ul_and_any = (b1_0 | b2_0) & (b1_1 | b1_2 | b2_1 | b2_2)
        ul_and = cross_ul | ul_and_any
        out = np.empty_like(s)
        out[..., 0] = (s[..., 0] != ul_and).astype(np.int8)
        out[..., 1] = (s[..., 1] + r_and) & 1
        out[..., 2] = (s[..., 2] + b_and) & 1
        return out

    for _ in range(n_rounds):
        state = pi(rho(theta(state)))
        state = chi(state)
    return state


def type_counts(state):
    'Function type counts.'
    ul, r, b = state[..., 0], state[..., 1], state[..., 2]
    return np.stack([
        (ul == 1) & (r == 0) & (b == 0),
        (ul == 0) & (r == 1) & (b == 0),
        (ul == 1) & (r == 1) & (b == 0),
        (ul == 0) & (r == 0) & (b == 1),
        (ul == 1) & (r == 0) & (b == 1),
        (ul == 0) & (r == 1) & (b == 1),
        (ul == 1) & (r == 1) & (b == 1),
        (ul == 0) & (r == 0) & (b == 0),
    ], axis=-1).sum(axis=(1, 2, 3))


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from propagate_milp_style import propagate_dep, load_xand_lookup  # noqa

    d = np.load(os.path.join(DATA_DIR, 'sha3_512',
                             'sha3_512_train_stageB_200000.npz'))
    X, y = d['X'], d['y']
    n_sample = int(os.environ.get('N_SAMPLE', '4000'))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), n_sample, replace=False)
    init = X[idx, :2 * 64 * 4].astype(np.int8)
    init_red = init[:, :64 * 4].reshape(n_sample, 64, 4)
    init_blue = init[:, 64 * 4:2 * 64 * 4].reshape(n_sample, 64, 4)

    print(f'=== type distribution ({n_sample} samples, after 2-round propagation / chi2, 1600 bits total) ===')
    for label, fn in [('relabel (true bit labels)', propagate_relabel),
                      ('dependency-channel (attack-script accounting)', None)]:
        if fn is not None:
            st = fn(init_red, init_blue, 2)
        else:
            xand = load_xand_lookup()
            st = propagate_dep(init_red, init_blue, 2, xand)
        tc = type_counts(st)
        print(f'\n--- {label} ---')
        for i, t in enumerate(TYPES):
            v = tc[:, i]
            print(f'  {t:>5}: mean={v.mean():7.1f} std={v.std():6.1f} '
                  f'range=[{v.min()},{v.max()}] zero-fraction={(v == 0).mean() * 100:5.1f}%')

        if fn is not None:
            u2 = tc[:, 0]
            print(f'  [check] relabel u count vs data y: agreement {(u2 == y[idx]).mean() * 100:.2f}%')
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez(os.path.join(OUT_DIR, f'type_stats_{label.split()[0]}.npz'),
                 idx=idx, y=y[idx], counts=tc)


if __name__ == '__main__':
    main()
