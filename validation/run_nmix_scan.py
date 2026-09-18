# -*- coding: utf-8 -*-
"""Scan the number of mixed bits N_mix -> best achievable attack degree D*.

Initial state (after the second chi): exactly N_mix mixed bits of type (1,0,0)
at free positions; the remaining bits follow the relabel type proportions
(lr/ur/lb/ub/lg/ug/zero) measured on stage-B samples. A third round is then
propagated with the MILP dependency-channel model plus the 128 matching
equations, maximizing the attack degree D* = min(d_R, d_B, m).

Expected behavior: small N_mix yields a large D*, and D* decreases as N_mix
grows, supporting the use of N_mix as the training target of the QP.

Usage: python run_nmix_scan.py --nmix 20,40,60,80,100,130,160,200,250 --tl 120
Repeated runs with the same --out file merge results and keep the best D*.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'better_attack'))

import gurobipy as gp
from gurobipy import GRB
from base_MILP.operation_MILP import Bit
from base_MILP.Keccak_MILP_64 import (
    create_theta_operation, create_chi_operation, rho, pi)

OUT_DIR = os.path.join(REPO, 'data', 'validation')
N_Z, N_Y, N_X = 64, 5, 5
N_TOTAL = 1600

# Relabel type proportions (means over 4000 stage-B samples, u excluded)
PROP = {'lr': 287, 'ur': 66, 'lb': 239, 'ub': 56, 'lg': 108, 'ug': 31, 'zero': 683}
PROP_SUM = sum(PROP.values())  # 1470
NON_U = list(PROP.keys())
TYPE_STATS_PATH = os.path.join(REPO, 'data', 'validation', 'type_stats_relabel.npz')


def counts_for(n_mix):
    'Function counts for.'
    other = N_TOTAL - n_mix
    cnt = {t: int(round(other * v / PROP_SUM)) for t, v in PROP.items()}
    diff = other - sum(cnt.values())
    rem = {t: other * PROP[t] / PROP_SUM - cnt[t] for t in PROP}
    order = sorted(PROP, key=lambda t: -rem[t])
    for i in range(abs(diff)):
        t = order[i % len(order)]
        cnt[t] += 1 if diff > 0 else -1
    cnt['u'] = n_mix
    assert sum(cnt.values()) == N_TOTAL, (cnt, sum(cnt.values()))
    return cnt


def counts_for_case(n_mix, non_u_props):
    """Build a count vector for one sampled non-u type-proportion case.

    non_u_props is a 7-vector over NON_U (lr, ur, lb, ub, lg, ug, zero) that
    sums to 1. The u count is fixed to n_mix and the remaining 1600-n_mix bits
    are allocated to the non-u types with the largest-remainder method.
    """
    other = N_TOTAL - n_mix
    raw = other * np.asarray(non_u_props, dtype=float)
    cnt = {t: int(np.floor(v)) for t, v in zip(NON_U, raw)}
    diff = int(other - sum(cnt.values()))
    rem = {t: raw[i] - cnt[t] for i, t in enumerate(NON_U)}
    order = sorted(NON_U, key=lambda t: -rem[t])
    for i in range(diff):
        cnt[order[i % len(order)]] += 1
    cnt['u'] = n_mix
    assert sum(cnt.values()) == N_TOTAL, (cnt, sum(cnt.values()))
    return cnt


def load_case_proportions():
    """Load empirical non-u type proportions from type_stats_relabel.npz if present."""
    if not os.path.exists(TYPE_STATS_PATH):
        return None
    d = np.load(TYPE_STATS_PATH)
    counts = d['counts'][:, 1:]          # drop u; columns follow NON_U
    s = counts.sum(axis=1, keepdims=True)
    return counts / s



def add_type_indicators(model, bits):
    'Function add type indicators.'
    z = {}
    for zz in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                b = bits[zz][y][x]
                zv = {}
                for t in ['u', 'lr', 'ur', 'lb', 'ub', 'lg', 'ug', 'zero']:
                    zv[t] = model.addVar(vtype=GRB.BINARY, name=f'z_{zz}_{y}_{x}_{t}')
                model.addConstr(sum(zv.values()) == 1)

                model.addConstr(zv['u'] <= b.ul)
                model.addConstr(zv['u'] <= 1 - b.r)
                model.addConstr(zv['u'] <= 1 - b.b)
                model.addConstr(zv['u'] >= b.ul - b.r - b.b)
                model.addConstr(zv['lr'] <= 1 - b.ul)
                model.addConstr(zv['lr'] <= b.r)
                model.addConstr(zv['lr'] <= 1 - b.b)
                model.addConstr(zv['lr'] >= b.r - b.ul - b.b)
                model.addConstr(zv['ur'] <= b.ul)
                model.addConstr(zv['ur'] <= b.r)
                model.addConstr(zv['ur'] <= 1 - b.b)
                model.addConstr(zv['ur'] >= b.ul + b.r - b.b - 1)
                model.addConstr(zv['lb'] <= 1 - b.ul)
                model.addConstr(zv['lb'] <= 1 - b.r)
                model.addConstr(zv['lb'] <= b.b)
                model.addConstr(zv['lb'] >= b.b - b.ul - b.r)
                model.addConstr(zv['ub'] <= b.ul)
                model.addConstr(zv['ub'] <= 1 - b.r)
                model.addConstr(zv['ub'] <= b.b)
                model.addConstr(zv['ub'] >= b.ul + b.b - b.r - 1)
                model.addConstr(zv['lg'] <= 1 - b.ul)
                model.addConstr(zv['lg'] <= b.r)
                model.addConstr(zv['lg'] <= b.b)
                model.addConstr(zv['lg'] >= b.r + b.b - b.ul - 1)
                model.addConstr(zv['ug'] <= b.ul)
                model.addConstr(zv['ug'] <= b.r)
                model.addConstr(zv['ug'] <= b.b)
                model.addConstr(zv['ug'] >= b.ul + b.r + b.b - 2)
                model.addConstr(zv['zero'] <= 1 - b.ul)
                model.addConstr(zv['zero'] <= 1 - b.r)
                model.addConstr(zv['zero'] <= 1 - b.b)
                model.addConstr(zv['zero'] >= 1 - b.ul - b.r - b.b)
                z[(zz, y, x)] = zv
    return z


def build_scan_model(cnt, model_name, time_limit):
    'Function build scan model.'
    model = gp.Model(model_name)
    model.setParam('MIPFocus', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Heuristics', 0.5)
    model.setParam('Threads', 2)
    model.setParam('OutputFlag', 0)

    state = [[[Bit(model, f's{zz}_{y}_{x}', ('*', '*', '*')) for x in range(N_X)]
              for y in range(N_Y)] for zz in range(N_Z)]
    z = add_type_indicators(model, state)
    for t, c in cnt.items():
        model.addConstr(gp.quicksum(z[(zz, y, x)][t] for zz in range(N_Z)
                                    for y in range(N_Y) for x in range(N_X)) == c,
                        name=f'cnt_{t}')


    r_init = cnt['lr'] + cnt['ur'] + cnt['lg'] + cnt['ug']
    b_init = cnt['lb'] + cnt['ub'] + cnt['lg'] + cnt['ug']

    theta_state, C, D, theta_vars = create_theta_operation(model, state, 'r3_theta')
    for x in range(5):
        for zz in range(64):
            model.addConstr(theta_vars[f"D_x{x}_z{zz}"]['delta_b'] == 0)
            for y in range(5):
                model.addConstr(theta_vars[f"new_z{zz}_y{y}_x{x}"]['delta_b'] == 0)
    rho_state = rho(theta_state)
    pi_state = pi(rho_state)
    chi_state, chi_vars = create_chi_operation(model, pi_state, 'r3_chi')

    delta_total_r = 0
    delta_total_b = 0
    for zz in range(64):
        for x in range(5):
            delta_total_r += theta_vars[f"C_x{x}_z{zz}"]['delta_r']
            delta_total_b += theta_vars[f"C_x{x}_z{zz}"]['delta_b']
            delta_total_r += theta_vars[f"D_x{x}_z{zz}"]['delta_r']
            delta_total_b += theta_vars[f"D_x{x}_z{zz}"]['delta_b']
            for y in range(5):
                delta_total_r += theta_vars[f"new_z{zz}_y{y}_x{x}"]['delta_r']
                delta_total_b += theta_vars[f"new_z{zz}_y{y}_x{x}"]['delta_b']
    for zz in range(64):
        for y in range(5):
            for x in range(5):
                delta_total_r += chi_vars[f"new_z{zz}_y{y}_x{x}"]['delta_r']
                delta_total_b += chi_vars[f"new_z{zz}_y{y}_x{x}"]['delta_b']

    hash_output_bits = []
    for zz in range(64):
        eq1 = model.addVar(vtype=GRB.BINARY, name=f'equation1_{zz}')
        model.addConstr(eq1 <= 1 - chi_state[zz][0][3].ul + chi_state[zz][0][3].r + chi_state[zz][0][3].b)
        model.addConstr(eq1 <= 1 - chi_state[zz][3][3].ul + chi_state[zz][3][3].r + chi_state[zz][3][3].b)
        model.addConstr(eq1 <= 1 - chi_state[(zz - 39) % 64][2][0].ul + chi_state[(zz - 39) % 64][2][0].r + chi_state[(zz - 39) % 64][2][0].b)
        model.addConstr(eq1 <= 1 - chi_state[(zz - 39) % 64][0][0].ul + chi_state[(zz - 39) % 64][0][0].r + chi_state[(zz - 39) % 64][0][0].b)
        hash_output_bits.append(eq1)
        eq2 = model.addVar(vtype=GRB.BINARY, name=f'equation2_{zz}')
        model.addConstr(eq2 <= 1 - chi_state[zz][1][4].ul + chi_state[zz][1][4].r + chi_state[zz][1][4].b)
        model.addConstr(eq2 <= 1 - chi_state[zz][4][4].ul + chi_state[zz][4][4].r + chi_state[zz][4][4].b)
        model.addConstr(eq2 <= 1 - chi_state[(zz - 25) % 64][3][1].ul + chi_state[(zz - 25) % 64][3][1].r + chi_state[(zz - 25) % 64][3][1].b)
        model.addConstr(eq2 <= 1 - chi_state[(zz - 25) % 64][1][1].ul + chi_state[(zz - 25) % 64][1][1].r + chi_state[(zz - 25) % 64][1][1].b)
        hash_output_bits.append(eq2)
    total_equations = gp.quicksum(hash_output_bits)

    temp_degree = model.addVar(vtype=GRB.INTEGER, lb=0, name='complexity')
    model.addConstr(temp_degree <= r_init - delta_total_r)
    model.addConstr(temp_degree <= b_init - delta_total_b)
    model.addConstr(temp_degree <= total_equations)
    model.setObjective(temp_degree, GRB.MAXIMIZE)

    return {
        'model': model, 'temp_degree': temp_degree,
        'r_init': r_init, 'b_init': b_init,
        'delta_total_r': delta_total_r, 'delta_total_b': delta_total_b,
        'total_equations': total_equations,
    }


def _gv(x):
    return x if isinstance(x, (int, float)) else x.getValue()


def _solve_one(args):
    nm, cnt, time_limit, seed, mip_gap = args
    res = build_scan_model(cnt, f'scan_{nm}_{seed}', time_limit)
    model = res['model']
    model.setParam('Seed', seed)
    # A zero MIP gap asks Gurobi to continue until optimality is proven (or the
    # time limit is reached), which reduces under-estimated D* values.
    model.setParam('MIPGap', mip_gap)
    model.setParam('MIPGapAbs', mip_gap)
    t0 = time.time()
    model.optimize()
    dt = time.time() - t0
    if model.SolCount > 0:
        td = int(round(res['temp_degree'].x))
        dr = res['r_init'] - int(round(_gv(res['delta_total_r'])))
        db = res['b_init'] - int(round(_gv(res['delta_total_b'])))
        m = int(round(_gv(res['total_equations'])))
        out = (nm, td, dr, db, m, model.status, dt, seed)
    else:
        out = (nm, -1, None, None, None, model.status, dt, seed)
    model.dispose()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nmix', type=str, default='20,40,60,80,100,130,160,200,250')
    ap.add_argument('--tl', type=float, default=120,
                    help='Gurobi time limit per (N_mix, seed) solve in seconds')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=3,
                    help='Gurobi random seeds per (N_mix, case); max D* is reported')
    ap.add_argument('--cases', type=int, default=1,
                    help='independent type-count cases per N_mix; use >=5 for a more robust curve')
    ap.add_argument('--mip-gap', type=float, default=0.0,
                    help='Gurobi relative/absolute MIP gap tolerance; 0 requests optimality proof')
    ap.add_argument('--out', type=str, default='nmix_scan_512.npz')
    ap.add_argument('--fresh', action='store_true',
                    help='ignore an existing --out file; otherwise merge and keep the best D* per N_mix')
    args = ap.parse_args()

    nmix_list = [int(v) for v in args.nmix.split(',')]

    # For --cases > 1, try to sample different empirical type-count cases from
    # type_stats_relabel.npz.  If that file is not available, fall back to the
    # mean proportions (the same case is then only diversified by Gurobi seeds).
    case_props = None
    if args.cases > 1:
        case_props = load_case_proportions()
        if case_props is None:
            print('WARNING: type_stats_relabel.npz not found; '
                  '--cases will repeat the mean-proportion case with different seeds.')

    tasks = []
    rng = np.random.RandomState(12345)
    for nm in nmix_list:
        if case_props is not None:
            pool_size = len(case_props)
            replace = pool_size < args.cases
            sampled_rows = rng.choice(pool_size, size=args.cases, replace=replace)
        else:
            sampled_rows = [0] * args.cases
        for case in range(args.cases):
            if case_props is None:
                cnt = counts_for(nm)
            else:
                cnt = counts_for_case(nm, case_props[sampled_rows[case]])
            for s in range(args.seeds):
                tasks.append((nm, cnt, args.tl, s, args.mip_gap))
    import multiprocessing as mp
    if args.procs > 1 and len(tasks) > 1:
        with mp.Pool(args.procs) as pool:
            raw = pool.map(_solve_one, tasks)
    else:
        raw = [_solve_one(t) for t in tasks]

    best = {}
    for r in raw:
        nm, td, dr, db, m, st, t, s = r
        if nm not in best or td > best[nm][1]:
            best[nm] = r

    # Merge with an earlier result file (unless --fresh): repeated runs with
    # more time/seeds/random starts accumulate the best-known D* instead of
    # overwriting a good previous solution with a worse one.
    out_path = os.path.join(OUT_DIR, args.out)
    if os.path.exists(out_path) and not args.fresh:
        old = np.load(out_path)
        for nm, D, dr, db, m in zip(old['n_mix'], old['D'],
                                    old['d_R'], old['d_B'], old['m']):
            nm = int(nm)
            D = int(D)
            dr = int(dr) if int(dr) >= 0 else None
            db = int(db) if int(db) >= 0 else None
            m = int(m) if int(m) >= 0 else None
            old_row = (nm, D, dr, db, m, 'merged', 0.0, -1)
            if nm not in best or D > best[nm][1]:
                best[nm] = old_row
        old.close()

    rows = [best[nm] for nm in sorted(best)]

    print('\n=== summary: N_mix -> D* (best known, max over cases/seeds/files) ===')
    print(f'{"N_mix":>6} {"D*":>6} {"d_R":>6} {"d_B":>6} {"m":>6} {"status":>8} {"time":>6}')
    for r in rows:
        nm, td, dr, db, m, st, t, _ = r
        print(f'{nm:>6} {td:>6} {str(dr):>6} {str(db):>6} {str(m):>6} {str(st):>8} {t:>6.1f}')
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(out_path,
             n_mix=np.array([r[0] for r in rows]),
             D=np.array([r[1] for r in rows], dtype=np.int32),
             d_R=np.array([r[2] if r[2] is not None else -1 for r in rows], dtype=np.int32),
             d_B=np.array([r[3] if r[3] is not None else -1 for r in rows], dtype=np.int32),
             m=np.array([r[4] if r[4] is not None else -1 for r in rows], dtype=np.int32),
             status=np.array([str(r[5]) for r in rows]))


if __name__ == '__main__':
    main()
