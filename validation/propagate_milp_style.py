# -*- coding: utf-8 -*-
'Module object.'
import os
import time

import numpy as np
from scipy.stats import spearmanr

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

EQ1_OFF = [(0, 3), (3, 3), (2, 0), (0, 0)]
EQ2_OFF = [(1, 4), (4, 4), (3, 1), (1, 1)]
EQ1_ZOFF = [0, 0, -39, -39]
EQ2_ZOFF = [0, 0, -25, -25]



def xor_feasible(ins, out, d):
    'Function xor feasible.'
    uls = [i[0] for i in ins]
    rs = [i[1] for i in ins]
    bs = [i[2] for i in ins]
    ps = [1 if (i[0] == 1 and i[1] == 0 and i[2] == 0) else 0 for i in ins]
    ubs = [1 if (i[0] == 1 and i[2] == 1) else 0 for i in ins]
    ou, orr, ob = out
    # ul
    for i, (ul, _, _) in enumerate(ins):
        if ul == 0:
            continue
        if ou < ul - d:
            return False
    if ou > sum(uls):
        return False
    if ou > 1 - d + sum(ubs):
        return False
    for ub in ubs:
        if ou < ub:
            return False
    # r
    if d + orr > sum(rs):
        return False
    for i in range(len(ins)):
        if orr < rs[i] - d - (sum(ps) - ps[i]):
            return False
    for p in ps:
        if orr + p + d > 1:
            return False
    # b
    if ob > sum(bs):
        return False
    for i in range(len(ins)):
        if ob < bs[i] - (sum(ps) - ps[i]):
            return False
    for p in ps:
        if ob > 1 - p:
            return False
    return True


def verify_xor_rule(n_inputs):
    'Function verify xor rule.'
    import itertools
    bad = 0
    multi = 0
    for ins in itertools.product([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                                  (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
                                 repeat=n_inputs):

        if all(i == (0, 0, 0) for i in ins):
            rule_out = (0, 0, 0)
        else:
            uls = [i[0] for i in ins]
            ps = [1 if (i[0] == 1 and i[1] == 0 and i[2] == 0) else 0 for i in ins]
            ou = int(any(uls))
            if any(ps):
                rule_out = (ou, 0, 0)
            else:
                rule_out = (ou, int(any(i[1] for i in ins)), int(any(i[2] for i in ins)))

        best_d, outs = None, []
        for d in (0, 1):
            for out in itertools.product([0, 1], repeat=3):
                if xor_feasible(ins, out, d):
                    if best_d is None or d < best_d:
                        best_d = d
                        outs = [out]
                    elif d == best_d:
                        outs.append(out)
        if best_d is None:
            bad += 1
            continue
        if len(outs) > 1:
            multi += 1
        if rule_out not in outs or (best_d != 0 and len(outs) == 1):

            if not xor_feasible(ins, rule_out, 0):
                bad += 1
    total = 8 ** n_inputs
    print(f'XOR {n_inputs}-input: enumerated {total} cases, rule output in min-d feasible set: {total - bad}/{total}'
          f' ({100 * (total - bad) / total:.2f}%), min-d multi-output: {multi}')
    return bad == 0



def load_xand_lookup():
    d = np.load(os.path.join(DATA_DIR, 'validation', 'xand_lookup.npz'))
    enc = np.zeros(512, dtype=np.uint8)
    for row, o0 in zip(d['inputs'], d['out0']):
        idx = 0
        for j, v in enumerate(row):
            idx |= int(v) << j
        enc[idx] = o0[0] | (o0[1] << 1) | (o0[2] << 2)
    return enc


def propagate_dep(init_red, init_blue, n_rounds, xand_enc):
    'Function propagate dep.'
    N = init_red.shape[0]
    n_x = init_red.shape[2]
    state = np.zeros((N, N_Z, N_Y, N_X, 3), dtype=np.uint8)
    for y in (0, 1):
        state[:, :, y, :n_x, 1] = (init_red == 1)
        state[:, :, y, :n_x, 2] = (init_blue == 1)

    def xor_theta(s):
        'Function xor theta.'

        c_ul = s[..., 0].max(axis=2)
        any_pu = ((s[..., 0] == 1) & (s[..., 1] == 0) & (s[..., 2] == 0)).max(axis=2)
        c_r = np.where(any_pu, 0, s[..., 1].max(axis=2))
        c_b = np.where(any_pu, 0, s[..., 2].max(axis=2))
        C = np.stack([c_ul, c_r, c_b], axis=-1)          # (N,64,5,3)

        Cxm1 = np.take(C, [(x - 1) % 5 for x in range(N_X)], axis=2)
        Cxp1 = np.roll(C, -1, axis=1)
        Cxp1 = np.take(Cxp1, [(x + 1) % 5 for x in range(N_X)], axis=2)
        D = xor2(Cxm1, Cxp1)

        D_y = D[:, :, None, :, :]
        new = np.empty_like(s)
        new[..., 0] = s[..., 0] | D_y[..., 0]
        s_pu = ((s[..., 0] == 1) & (s[..., 1] == 0) & (s[..., 2] == 0))
        d_pu = ((D_y[..., 0] == 1) & (D_y[..., 1] == 0) & (D_y[..., 2] == 0))
        any_pu_new = s_pu | d_pu
        new[..., 1] = np.where(any_pu_new, 0, s[..., 1] | D_y[..., 1])
        new[..., 2] = np.where(any_pu_new, 0, s[..., 2] | D_y[..., 2])
        return new

    def xor2(a, b):
        'Function xor2.'
        out = np.empty_like(a)
        a_pu = (a[..., 0] == 1) & (a[..., 1] == 0) & (a[..., 2] == 0)
        b_pu = (b[..., 0] == 1) & (b[..., 1] == 0) & (b[..., 2] == 0)
        out[..., 0] = a[..., 0] | b[..., 0]
        any_pu = a_pu | b_pu
        out[..., 1] = np.where(any_pu, 0, a[..., 1] | b[..., 1])
        out[..., 2] = np.where(any_pu, 0, a[..., 2] | b[..., 2])
        return out

    def rho_pi(s):
        out = np.empty_like(s)
        for y in range(N_Y):
            for x in range(N_X):
                out[:, :, y, x, :] = s[:, RHO_IDX[y, x], PI_MAP[y, x][0], PI_MAP[y, x][1], :]
        return out

    def chi_lookup(s):
        'Function chi lookup.'
        x0 = s
        x1 = np.roll(s, -1, axis=3)
        x2 = np.roll(s, -2, axis=3)
        idx = (x0[..., 0].astype(np.uint32) |
               (x0[..., 1].astype(np.uint32) << 1) |
               (x0[..., 2].astype(np.uint32) << 2) |
               (x1[..., 0].astype(np.uint32) << 3) |
               (x1[..., 1].astype(np.uint32) << 4) |
               (x1[..., 2].astype(np.uint32) << 5) |
               (x2[..., 0].astype(np.uint32) << 6) |
               (x2[..., 1].astype(np.uint32) << 7) |
               (x2[..., 2].astype(np.uint32) << 8))
        e = xand_enc[idx]                        # (N,64,5,5)
        out = np.empty_like(s)
        out[..., 0] = (e & 1).astype(np.uint8)
        out[..., 1] = ((e >> 1) & 1).astype(np.uint8)
        out[..., 2] = ((e >> 2) & 1).astype(np.uint8)
        return out

    for rnd in range(n_rounds):
        if rnd == 0:
            s = rho_pi(state)
        else:
            s = rho_pi(xor_theta(state))
        state = chi_lookup(s)
    return state


def compute_m(state):
    'Function compute m.'
    m = np.zeros(len(state), dtype=np.int32)
    for z in range(N_Z):
        ok1 = np.ones(len(state), dtype=bool)
        ok2 = np.ones(len(state), dtype=bool)
        for (dy, dx), dz in zip(EQ1_OFF, EQ1_ZOFF):
            zz = (z + dz) % N_Z
            pure_u = (state[:, zz, dy, dx, 0] == 1) & (state[:, zz, dy, dx, 1] == 0) & (state[:, zz, dy, dx, 2] == 0)
            ok1 &= ~pure_u
        for (dy, dx), dz in zip(EQ2_OFF, EQ2_ZOFF):
            zz = (z + dz) % N_Z
            pure_u = (state[:, zz, dy, dx, 0] == 1) & (state[:, zz, dy, dx, 1] == 0) & (state[:, zz, dy, dx, 2] == 0)
            ok2 &= ~pure_u
        m += ok1.astype(np.int32) + ok2.astype(np.int32)
    return m


def main():
    ok23 = verify_xor_rule(2) and verify_xor_rule(3)
    print(f'XOR rule verification: {ok23}')

    xand_enc = load_xand_lookup()

    data_path = os.path.join(DATA_DIR, 'sha3_512', 'sha3_512_train_stageB_200000.npz')
    d = np.load(data_path)
    X, y = d['X'], d['y']
    n_red, n_blue = d['n_red'], d['n_blue']
    n_sample = int(os.environ.get('N_SAMPLE', '3000'))
    seed = int(os.environ.get('SEED', '42'))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_sample, replace=False)
    init = X[idx, :2 * 64 * 4].astype(np.uint8)
    init_red = init[:, :64 * 4].reshape(n_sample, 64, 4)
    init_blue = init[:, 64 * 4:2 * 64 * 4].reshape(n_sample, 64, 4)
    y_s, nr_s, nb_s = y[idx], n_red[idx], n_blue[idx]


    t0 = time.time()
    s2 = propagate_dep(init_red, init_blue, 2, xand_enc)
    u2 = ((s2[..., 0] == 1) & (s2[..., 1] == 0) & (s2[..., 2] == 0)).sum(axis=(1, 2, 3))
    print(f'\n[check] dependency-channel 2-round u2 vs data y ({time.time()-t0:.1f}s)')
    print(f'  u2: mean={u2.mean():.1f} std={u2.std():.1f}  y: mean={y_s.mean():.1f} std={y_s.std():.1f}')
    print(f'  agreement: {(u2 == y_s).mean() * 100:.2f}%, mean diff: {(u2 - y_s).mean():.2f}')


    t0 = time.time()
    s3 = propagate_dep(init_red, init_blue, 3, xand_enc)
    m = compute_m(s3)
    D = np.minimum(np.minimum(nr_s, nb_s), m)
    print(f'\n[3 rounds] ({time.time()-t0:.1f}s)')
    for name, arr in [('n_red', nr_s), ('n_blue', nb_s), ('m', m), ('D', D)]:
        print(f'  {name}: mean={arr.mean():.2f} std={arr.std():.2f} range=[{arr.min()},{arr.max()}] unique={len(np.unique(arr))}')
    print(f'  D==0 fraction: {(D == 0).mean() * 100:.1f}%')
    for name, arr in [('n_red', nr_s), ('n_blue', nb_s), ('m', m), ('D', D)]:
        rho, p = spearmanr(y_s, arr)
        print(f'  Spearman(N_mix, {name}) = {rho:.4f} (p={p:.2e})')

    rho, p = spearmanr(y_s, np.minimum(nr_s, nb_s))
    print(f'  Spearman(N_mix, min(n_red,n_blue)) = {rho:.4f}')

    np.savez(os.path.join(OUT_DIR, 'precheck_dep.npz'),
             y=y_s, n_red=nr_s, n_blue=nb_s, m=m, D=D)


if __name__ == '__main__':
    main()
