'Module object.'

import sys
import os
import time
import argparse
import glob
import numpy as np
import multiprocessing as mp

_current_dir = os.path.dirname(os.path.abspath(__file__))
_milp_dir = os.path.dirname(os.path.abspath(__file__))
if _milp_dir not in sys.path:
    sys.path.insert(0, _milp_dir)

from operation import Bit, add_xor_red_cancel_only, add_and_no_cond
import gurobipy as gp
from gurobipy import GRB


try:
    from ascon import (create_first_P_S_operation_first_one_constant_cond_padding_three_stage,
                       create_first_P_L_operation)
    _HAS_ASCON_MILP = True
except ImportError:
    _HAS_ASCON_MILP = False


SLICE_NUMBER = 64
NUM_COLS = 5
STATE_SIZE = SLICE_NUMBER * NUM_COLS  # 320
RATE_SLICES = SLICE_NUMBER - 1


PL_OFFSETS = np.array([
    [19, 28],   # x=0
    [61, 39],   # x=1
    [1, 6],     # x=2
    [10, 17],   # x=3
    [7, 41],    # x=4
], dtype=np.int32) % SLICE_NUMBER


# ============================================================

# ============================================================

def _xor_bits(bits_list):
    'Function xor bits.'
    r_acc = 0
    b_acc = 0
    w_acc = 0
    for r, b, w in bits_list:
        r_acc ^= r
        b_acc ^= b
        w_acc = int(w_acc or w or (r_acc and b_acc))
    return (r_acc & ~w_acc, b_acc & ~w_acc, w_acc)


def deterministic_PS(state_r, state_b, state_w):
    'Function deterministic PS.'
    new_r = np.zeros_like(state_r)
    new_b = np.zeros_like(state_b)
    new_w = np.zeros_like(state_w)

    for z in range(SLICE_NUMBER):
        s_r = state_r[z]
        s_b = state_b[z]
        s_w = state_w[z]

        # Step 1: XOR forward
        t1_r = np.zeros(5, dtype=np.int8)
        t1_b = np.zeros(5, dtype=np.int8)
        t1_w = np.zeros(5, dtype=np.int8)

        t1_r[0] = s_r[0] ^ s_r[4]; t1_b[0] = s_b[0] ^ s_b[4]; t1_w[0] = int(s_w[0] or s_w[4] or (s_r[0] and not s_b[0] and s_r[4] and not s_b[4]) or (s_b[0] and not s_r[0] and s_b[4] and not s_r[4]))
        t1_r[1] = s_r[1]; t1_b[1] = s_b[1]; t1_w[1] = s_w[1]
        t1_r[2] = s_r[1] ^ s_r[2]; t1_b[2] = s_b[1] ^ s_b[2]; t1_w[2] = int(s_w[1] or s_w[2])
        t1_r[3] = s_r[3]; t1_b[3] = s_b[3]; t1_w[3] = s_w[3]
        t1_r[4] = s_r[3] ^ s_r[4]; t1_b[4] = s_b[3] ^ s_b[4]; t1_w[4] = int(s_w[3] or s_w[4])

        for x in [0, 2, 4]:
            t1_w[x] = int(t1_w[x] or (t1_r[x] and t1_b[x]))

        # Step 2: AND + XOR
        t2_r = np.zeros(5, dtype=np.int8)
        t2_b = np.zeros(5, dtype=np.int8)
        t2_w = np.zeros(5, dtype=np.int8)

        for x in range(5):
            a_r, a_b, a_w = t1_r[(x+1)%5], t1_b[(x+1)%5], t1_w[(x+1)%5]
            b_r, b_b, b_w = t1_r[(x+2)%5], t1_b[(x+2)%5], t1_w[(x+2)%5]

            and_w = int(
                (a_r and b_b) or (a_b and b_r) or
                (a_r and b_r) or (a_b and b_b) or
                a_w or b_w or
                (a_r and a_b) or (b_r and b_b)
            )
            and_r = 0
            and_b = 0

            xor_r = t1_r[x] ^ and_r
            xor_b = t1_b[x] ^ and_b
            xor_w = int(t1_w[x] or and_w or (xor_r and xor_b))
            if xor_w:
                xor_r = 0
                xor_b = 0

            t2_r[x] = xor_r
            t2_b[x] = xor_b
            t2_w[x] = xor_w

        # Step 3: XOR backward
        out_r = np.zeros(5, dtype=np.int8)
        out_b = np.zeros(5, dtype=np.int8)
        out_w = np.zeros(5, dtype=np.int8)

        out_r[0] = t2_r[0] ^ t2_r[4]
        out_b[0] = t2_b[0] ^ t2_b[4]
        out_w[0] = int(t2_w[0] or t2_w[4] or (out_r[0] and out_b[0]))

        out_r[1] = t2_r[1] ^ t2_r[0]
        out_b[1] = t2_b[1] ^ t2_b[0]
        out_w[1] = int(t2_w[1] or t2_w[0] or (out_r[1] and out_b[1]))

        out_r[2] = t2_r[2]
        out_b[2] = t2_b[2]
        out_w[2] = t2_w[2]

        out_r[3] = t2_r[2] ^ t2_r[3]
        out_b[3] = t2_b[2] ^ t2_b[3]
        out_w[3] = int(t2_w[2] or t2_w[3] or (out_r[3] and out_b[3]))

        out_r[4] = t2_r[4]
        out_b[4] = t2_b[4]
        out_w[4] = t2_w[4]

        for x in range(5):
            if out_w[x]:
                out_r[x] = 0
                out_b[x] = 0

        new_r[z] = out_r
        new_b[z] = out_b
        new_w[z] = out_w

    return new_r, new_b, new_w


def deterministic_PL(state_r, state_b, state_w):
    'Function deterministic PL.'
    new_r = np.zeros_like(state_r)
    new_b = np.zeros_like(state_b)
    new_w = np.zeros_like(state_w)

    for x in range(NUM_COLS):
        off0, off1 = int(PL_OFFSETS[x, 0]), int(PL_OFFSETS[x, 1])

        col_r = state_r[:, x]
        col_b = state_b[:, x]
        col_w = state_w[:, x]

        xor_r = col_r.copy()
        xor_b = col_b.copy()
        xor_w = col_w.copy()

        xor_r ^= np.roll(col_r, -off0, axis=0)
        xor_b ^= np.roll(col_b, -off0, axis=0)
        xor_w = np.logical_or(xor_w, np.roll(col_w, -off0, axis=0)).astype(np.int8)

        xor_r ^= np.roll(col_r, -off1, axis=0)
        xor_b ^= np.roll(col_b, -off1, axis=0)
        xor_w = np.logical_or(xor_w, np.roll(col_w, -off1, axis=0)).astype(np.int8)

        xor_w = np.logical_or(xor_w, np.logical_and(xor_r, xor_b)).astype(np.int8)
        xor_r[xor_w > 0] = 0
        xor_b[xor_w > 0] = 0

        new_r[:, x] = xor_r
        new_b[:, x] = xor_b
        new_w[:, x] = xor_w

    return new_r, new_b, new_w


def deterministic_full_forward(init_red, init_blue):
    'Function deterministic full forward.'
    r0 = init_red.astype(np.int8).copy()
    b0 = init_blue.astype(np.int8).copy()
    w0 = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int8)

    r_ps0, b_ps0, w_ps0 = deterministic_PS(r0, b0, w0)
    r_pl0, b_pl0, w_pl0 = deterministic_PL(r_ps0, b_ps0, w_ps0)

    r_ps1, b_ps1, w_ps1 = deterministic_PS(r_pl0, b_pl0, w_pl0)
    r_pl1, b_pl1, w_pl1 = deterministic_PL(r_ps1, b_ps1, w_ps1)

    u2 = int(np.sum(w_pl1))

    r_feat = np.concatenate([
        r0.flatten().astype(np.float32),
        r_ps0.flatten().astype(np.float32),
        r_pl0.flatten().astype(np.float32),
        r_ps1.flatten().astype(np.float32),
    ])
    b_feat = np.concatenate([
        b0.flatten().astype(np.float32),
        b_ps0.flatten().astype(np.float32),
        b_pl0.flatten().astype(np.float32),
        b_ps1.flatten().astype(np.float32),
    ])

    feat = np.concatenate([r_feat, b_feat]).astype(np.float32)

    stats = {
        'n_red_init': int(np.sum(r0)),
        'n_blue_init': int(np.sum(b0)),
        'n_red_ps0': int(np.sum(r_ps0)),
        'n_blue_ps0': int(np.sum(b_ps0)),
        'n_red_pl0': int(np.sum(r_pl0)),
        'n_blue_pl0': int(np.sum(b_pl0)),
        'n_red_ps1': int(np.sum(r_ps1)),
        'n_blue_ps1': int(np.sum(b_ps1)),
        'n_w_pl0': int(np.sum(w_pl0)),
        'n_w_ps1': int(np.sum(w_ps1)),
    }

    return feat, u2, stats


# ============================================================

# ============================================================

def generate_blue_configs(n_configs=5000, n_blue_range=(12, 18)):
    'Function generate blue configs.'
    rate_slots = 31
    configs = []
    n_blues_list = []

    min_b, max_b = n_blue_range

    for i in range(n_configs):
        n_blue = np.random.randint(min_b, max_b + 1)
        positions = np.random.choice(rate_slots, size=n_blue, replace=False)

        blue_mask = np.zeros(rate_slots, dtype=bool)
        blue_mask[positions] = True

        configs.append(blue_mask)
        n_blues_list.append(n_blue)

    return np.array(configs), np.array(n_blues_list, dtype=np.int32)


def blue_mask_to_2d(blue_mask_1d):
    'Function blue mask to 2d.'
    blue_2d = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int8)
    for z in range(RATE_SLICES):
        if blue_mask_1d[z]:
            blue_2d[z, 0] = 1
    return blue_2d


# ============================================================

# ============================================================

def solve_one_red_simple(blue_mask_1d, n_blue, seed, maxR, tmax, tlimit, gap,
                         invert_objective=False, excluded_reds=None):
    'Function solve one red simple.'
    model = gp.Model("Ascon_Red_Simple")
    model.setParam('Seed', seed)
    model.setParam('MIPGap', gap)
    model.setParam('TimeLimit', tlimit)
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)

    initial_state = [[None for _ in range(NUM_COLS)] for _ in range(SLICE_NUMBER)]
    red_bits = []

    for z in range(SLICE_NUMBER):
        for x in range(NUM_COLS):
            if x == 0 and z < RATE_SLICES:
                if blue_mask_1d[z]:
                    bit = Bit(model, f"init_z{z}_x{x}", (0, 0, 1, 0, 0))
                else:
                    bit = Bit(model, f"init_z{z}_x{x}", (0, '*', 0, 0, 0))
                    model.addConstr(bit.b == 0)
                    red_bits.append(bit.r)
            elif x == 0 and z == SLICE_NUMBER - 1:
                bit = Bit(model, f"init_z{z}_x{x}", (0, 0, 0, 0, 0))
            else:
                bit = Bit(model, f"init_z{z}_x{x}", (0, 0, 0, 0, 0))
            initial_state[z][x] = bit

    blue_count = n_blue

    if len(red_bits) > 0:
        model.addConstr(gp.quicksum(red_bits) >= blue_count, name="red_ge_blue")
        model.addConstr(gp.quicksum(red_bits) <= maxR, name="red_le_maxR")
        model.addConstr(gp.quicksum(red_bits) + blue_count <= tmax, name="total_le_tmax")
    else:
        model.dispose()
        return None

    pl_state = [[None for _ in range(NUM_COLS)] for _ in range(SLICE_NUMBER)]

    for z in range(SLICE_NUMBER):
        for x in range(NUM_COLS):
            off0, off1 = int(PL_OFFSETS[x, 0]), int(PL_OFFSETS[x, 1])

            in_bits = [
                initial_state[z][x],
                initial_state[(z + off0) % SLICE_NUMBER][x],
                initial_state[(z + off1) % SLICE_NUMBER][x],
            ]

            pl_bit = Bit(model, f"pl_z{z}_x{x}", (0, '*', '*', 0, 0))

            xor_vars = add_xor_red_cancel_only(
                model, in_bits, pl_bit,
                name=f"pl_xor_z{z}_x{x}"
            )

            if isinstance(xor_vars, dict):
                if 'delta_r' in xor_vars:
                    delta_r = xor_vars['delta_r']
                    if not isinstance(delta_r, int):
                        model.addConstr(delta_r == 0, name=f"pl_no_delta_r_z{z}_x{x}")
                if 'delta_b' in xor_vars:
                    delta_b = xor_vars['delta_b']
                    if not isinstance(delta_b, int):
                        model.addConstr(delta_b == 0, name=f"pl_no_delta_b_z{z}_x{x}")

            pl_state[z][x] = pl_bit

    # Deterministic objective: minimise (or, with invert_objective, maximise) the
    # number of marked bits after the first round. Diversity of the returned red
    # configurations comes from the exclusion constraints added just below.
    spread_terms = []
    for z in range(SLICE_NUMBER):
        for x in range(NUM_COLS):
            bit = pl_state[z][x]
            spread_terms.append(bit.r)
            spread_terms.append(bit.b)

    spread = gp.quicksum(spread_terms)

    for prev_red in (excluded_reds or []):
        cut_vars = [initial_state[z][x].r for (z, x) in prev_red
                    if isinstance(initial_state[z][x].r, gp.Var)]
        if cut_vars:
            model.addConstr(gp.quicksum(cut_vars) <= len(cut_vars) - 1)

    if invert_objective:
        model.setObjective(spread, GRB.MAXIMIZE)
    else:
        model.setObjective(spread, GRB.MINIMIZE)

    model.optimize()

    if model.SolCount == 0:
        model.dispose()
        return None

    init_red_2d = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int8)
    init_blue_2d = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int8)

    for z in range(SLICE_NUMBER):
        for x in range(NUM_COLS):
            bit = initial_state[z][x]
            rv = int(bit.r.X) if isinstance(bit.r, gp.Var) else int(bit.r)
            bv = int(bit.b.X) if isinstance(bit.b, gp.Var) else int(bit.b)
            init_red_2d[z, x] = rv
            init_blue_2d[z, x] = bv

    model.dispose()

    n_red = int(np.sum(init_red_2d))
    if n_red < n_blue:
        return None

    return init_red_2d, init_blue_2d


# ============================================================

# ============================================================

def solve_one_red_enhanced(blue_mask_1d, n_blue, seed, maxR, tmax, tlimit, gap,
                           invert_objective=False, excluded_reds=None):
    'Function solve one red enhanced.'
    if not _HAS_ASCON_MILP:

        print(f"  [WARNING] ascon.py unavailable, falling back to simple mode")
        return solve_one_red_simple(blue_mask_1d, n_blue, seed, maxR, tmax, tlimit, gap,
                                   invert_objective, excluded_reds)

    model = gp.Model("Ascon_Red_Enhanced")
    model.setParam('Seed', seed)
    model.setParam('MIPGap', gap)
    model.setParam('TimeLimit', tlimit)
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)

    # ================================================================

    # ================================================================
    initial_state = [[Bit(model, f"init_z{z}_x{x}", (0, 0, 0, 0, 0))
                      for x in range(NUM_COLS)] for z in range(SLICE_NUMBER)]

    red_bits = []

    for z in range(SLICE_NUMBER):
        x = 0
        if z >= SLICE_NUMBER - 1:

            initial_state[z][x] = Bit(model, f"init_z{z}_x{x}", (0, 0, 0, 0, 0))
        elif blue_mask_1d[z]:

            initial_state[z][x] = Bit(model, f"init_z{z}_x{x}", (0, 0, 1, 0, 0))
        else:

            initial_state[z][x] = Bit(model, f"init_z{z}_x{x}", (0, '*', 0, 0, 0))
            model.addConstr(initial_state[z][x].b == 0, name=f"no_blue_z{z}")

            model.addConstr(initial_state[z][x].r + initial_state[z][x].b <= 1,
                           name=f"exclusive_z{z}")
            red_bits.append(initial_state[z][x].r)

    # ================================================================

    # ================================================================
    blue_count = n_blue

    if len(red_bits) > 0:
        model.addConstr(gp.quicksum(red_bits) >= blue_count, name="red_ge_blue")
        model.addConstr(gp.quicksum(red_bits) <= maxR, name="red_le_maxR")
        model.addConstr(gp.quicksum(red_bits) + blue_count <= tmax, name="total_le_tmax")
    else:
        model.dispose()
        return None

    # ================================================================

    # ================================================================
    ps0_state, ps0_vars = create_first_P_S_operation_first_one_constant_cond_padding_three_stage(
        model, initial_state, "round0_PS")
    pl0_state, pl0_vars = create_first_P_L_operation(
        model, ps0_state, "round0_PL")

    # ================================================================


    # ================================================================
    # Deterministic objective: minimise (or, with invert_objective, maximise) the
    # number of marked bits after the first round. Diversity of the returned red
    # configurations comes from the exclusion constraints added just below.
    spread_terms = []
    for z in range(SLICE_NUMBER):
        for x in range(NUM_COLS):
            bit = pl0_state[z][x]
            if isinstance(bit.r, gp.Var):
                spread_terms.append(bit.r)
            if isinstance(bit.b, gp.Var):
                spread_terms.append(bit.b)

    if len(spread_terms) == 0:
        model.dispose()
        return None

    spread = gp.quicksum(spread_terms)

    for prev_red in (excluded_reds or []):
        cut_vars = [initial_state[z][x].r for (z, x) in prev_red
                    if isinstance(initial_state[z][x].r, gp.Var)]
        if cut_vars:
            model.addConstr(gp.quicksum(cut_vars) <= len(cut_vars) - 1)

    if invert_objective:
        model.setObjective(spread, GRB.MAXIMIZE)
    else:
        model.setObjective(spread, GRB.MINIMIZE)

    # ================================================================

    # ================================================================
    model.optimize()

    if model.SolCount == 0:
        model.dispose()
        return None

    # ================================================================

    # ================================================================
    init_red_2d = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int8)
    init_blue_2d = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int8)

    for z in range(SLICE_NUMBER):
        for x in range(NUM_COLS):
            bit = initial_state[z][x]
            rv = int(bit.r.X) if isinstance(bit.r, gp.Var) else int(bit.r)
            bv = int(bit.b.X) if isinstance(bit.b, gp.Var) else int(bit.b)
            init_red_2d[z, x] = rv
            init_blue_2d[z, x] = bv

    model.dispose()

    n_red = int(np.sum(init_red_2d))
    if n_red < n_blue:
        return None

    return init_red_2d, init_blue_2d


# ============================================================

# ============================================================

def worker_phase_b(params):
    'Function worker phase b.'
    (tasks, maxR, tmax, tlimit, gap, worker_id, seed_base, mode) = params

    X_list, y_list, meta_list = [], [], []
    t0 = time.time()
    n_gen = 0
    n_skipped = 0


    if mode == 'enhanced':
        solve_fn = solve_one_red_enhanced
    else:
        solve_fn = solve_one_red_simple

    for task_idx, (blue_mask_1d, n_blue, n_variants) in enumerate(tasks):
        if n_blue == 0:
            continue

        if n_blue > tmax - 3:
            continue

        red_sets_seen = set()

        for variant in range(n_variants):
            seed = seed_base + task_idx * 10000 + variant * 100

            invert = (variant % 5 == 0)

            result = solve_fn(
                blue_mask_1d, n_blue, seed, maxR, tmax, tlimit, gap,
                invert_objective=invert,
                excluded_reds=list(red_sets_seen),
            )

            if result is None:
                n_skipped += 1
                continue

            init_red, init_blue = result
            n_red = int(np.sum(init_red))

            if n_red < n_blue:
                n_skipped += 1
                continue

            r_key = tuple(sorted((z, x) for z in range(SLICE_NUMBER) for x in range(NUM_COLS)
                                 if init_red[z, x]))
            if r_key in red_sets_seen:
                n_skipped += 1
                continue
            red_sets_seen.add(r_key)

            feat, u2, stats = deterministic_full_forward(init_red, init_blue)

            X_list.append(feat)
            y_list.append(u2)
            meta_list.append((stats['n_red_init'], stats['n_blue_init'],
                              stats['n_red_ps0'], stats['n_blue_ps0']))
            n_gen += 1

    elapsed = time.time() - t0

    if n_skipped > 0:
        total_attempts = sum(t[2] for t in tasks)
        skip_rate = n_skipped / max(1, total_attempts) * 100
        print(f"  [worker#{worker_id}] skipped {n_skipped} ({skip_rate:.0f}%), "
              f"generated {n_gen}", flush=True)

    if len(X_list) == 0:
        return (np.zeros((0, 1280), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=[('n_red', 'i4'), ('n_blue', 'i4'),
                                   ('ps0_r', 'i4'), ('ps0_b', 'i4')]),
                0, elapsed)

    X_arr = np.stack(X_list).astype(np.float32)
    y_arr = np.array(y_list, dtype=np.int32)
    meta_arr = np.array(meta_list, dtype=[('n_red', 'i4'), ('n_blue', 'i4'),
                                           ('ps0_r', 'i4'), ('ps0_b', 'i4')])
    return X_arr, y_arr, meta_arr, n_gen, elapsed


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Ascon Work 1 two-stage data generation v3')
    parser.add_argument('--n-samples', type=int, default=1_000_000,
                        help='target sample count (default: 1000000)')
    parser.add_argument('--n-procs', type=int, default=None,
                        help='number of parallel processes (default: CPU cores)')
    parser.add_argument('--mode', type=str, default='simple',
                        choices=['simple', 'enhanced'],
                        help='MILP mode: simple (fast) / enhanced (with P_S AND modeling)')
    parser.add_argument('--blue-configs', type=int, default=5000,
                        help='blue config library size (default: 5000)')
    parser.add_argument('--blue-range', type=str, default='12,18',
                        help='blue count range min,max (default: 12,18)')
    parser.add_argument('--variants-per-blue', type=int, default=None,
                        help='red variants per blue config (default: auto)')
    parser.add_argument('--max-red', type=int, default=35,
                        help='red count upper bound (default: 25)')
    parser.add_argument('--total-max', type=int, default=45,
                        help='red+blue total upper bound (default: 45)')
    parser.add_argument('--time-limit', type=int, default=5,
                        help='per-solve MILP time limit in seconds (default: 5)')
    parser.add_argument('--gap', type=float, default=0.95,
                        help='MIP gap (default: 0.95)')
    parser.add_argument('--output', type=str, default=None,
                        help='output file path')
    parser.add_argument('--checkpoint-every', type=int, default=50000,
                        help='checkpoint interval (default: 50000)')
    parser.add_argument('--bench', type=int, default=None,
                        help='benchmark mode: generate N samples, print stats, then exit')
    args = parser.parse_args()


    if args.mode == 'enhanced' and not _HAS_ASCON_MILP:
        print("Warning: ascon.py unavailable, enhanced mode unavailable, falling back to simple mode")
        args.mode = 'simple'


    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_current_dir))),
                              'data', 'ascon')
    os.makedirs(result_dir, exist_ok=True)


    if args.n_procs is None:
        args.n_procs = min(mp.cpu_count(), 16)


    blue_min, blue_max = map(int, args.blue_range.split(','))


    print("=" * 60)
    print(f"Ascon Work 1: two-stage training data generation v3 (mode: {args.mode})")
    print("=" * 60)
    print(f"Target samples: {args.n_samples:,}")
    print(f"Parallel processes: {args.n_procs}")

    blue_configs, n_blues = generate_blue_configs(
        n_configs=args.blue_configs,
        n_blue_range=(blue_min, blue_max),
    )
    n_unique = len(blue_configs)
    print(f"Blue config library: {n_unique} configs")
    for nb in sorted(set(n_blues)):
        print(f"  n_blue={nb}: {int(np.sum(n_blues == nb))}")


    if args.variants_per_blue is None:
        capacity_per_pass = n_unique
        args.variants_per_blue = max(1, args.n_samples // capacity_per_pass + 1)
        args.variants_per_blue = min(args.variants_per_blue, 50)

    capacity_per_pass = n_unique * args.variants_per_blue
    tile_factor = max(1, args.n_samples // capacity_per_pass + 1)
    if tile_factor > 1:
        print(f"Insufficient capacity ({capacity_per_pass:,} < {args.n_samples:,}), tile {tile_factor}x")
        blue_configs = np.tile(blue_configs, (tile_factor, 1))
        n_blues = np.tile(n_blues, tile_factor)
    total_blue_slots = len(blue_configs)

    total_attempts = total_blue_slots * args.variants_per_blue
    print(f"Plan: {total_blue_slots:,} blue slots x {args.variants_per_blue} variants "
          f"= {total_attempts:,} MILP solves")
    print(f"Params: maxR={args.max_red}, total<={args.total_max}")
    print(f"MILP: gap={args.gap}, tlimit={args.time_limit}s")


    if args.output is None:
        args.output = os.path.join(result_dir, f'ascon_train_{args.n_samples}.npz')
    print(f"Output: {args.output}")
    print()


    slots_per_worker = total_blue_slots // args.n_procs
    remainder = total_blue_slots % args.n_procs

    tasks_for_workers = []
    idx = 0
    for pid in range(args.n_procs):
        n_slots = slots_per_worker + (1 if pid < remainder else 0)
        worker_tasks = []
        for i in range(n_slots):
            if idx < total_blue_slots:
                worker_tasks.append(
                    (blue_configs[idx], int(n_blues[idx]), args.variants_per_blue))
                idx += 1
        tasks_for_workers.append(
            (worker_tasks, args.max_red, args.total_max,
             args.time_limit, args.gap, pid, 42 + pid * 10000, args.mode))

    print(f"Slots per process: ~{slots_per_worker}")
    print(f"Starting {args.n_procs} processes...")
    sys.stdout.flush()

    t_start = time.time()
    all_X, all_y = [], []
    all_n_red, all_n_blue = [], []
    total_gen = 0

    with mp.Pool(args.n_procs) as pool:
        for i, (X_chunk, y_chunk, meta_chunk, n_gen, elapsed) in enumerate(
                pool.imap_unordered(worker_phase_b, tasks_for_workers)):
            pid = i
            all_X.append(X_chunk)
            all_y.append(y_chunk)
            all_n_red.append(meta_chunk['n_red'])
            all_n_blue.append(meta_chunk['n_blue'])
            total_gen += n_gen

            tput = n_gen / elapsed if elapsed > 0 else 0
            pct = min(100, total_gen / args.n_samples * 100)
            elapsed_total = time.time() - t_start
            overall_tput = total_gen / elapsed_total if elapsed_total > 0 else 0
            eta_sec = (args.n_samples - total_gen) / overall_tput if overall_tput > 0 else float('inf')

            print(f"  worker #{pid}: {n_gen:,} samples, "
                  f"{elapsed:.0f}s, {tput:.1f}sol/s, "
                  f"total: {total_gen:,}/{args.n_samples:,} "
                  f"({pct:.1f}%), overall: {overall_tput:.1f}sol/s, "
                  f"ETA: {eta_sec/3600:.1f}h",
                  flush=True)


            if args.checkpoint_every > 0 and total_gen > 0:
                last_mark = (total_gen // args.checkpoint_every) * args.checkpoint_every
                prev_mark = ((total_gen - n_gen) // args.checkpoint_every) * args.checkpoint_every
                if last_mark > prev_mark and last_mark > 0:
                    ckpt_path = args.output.replace('.npz', f'_ckpt_{last_mark}.npz')
                    X_ckpt = np.concatenate(all_X).astype(np.float32)
                    y_ckpt = np.concatenate(all_y).astype(np.int32)
                    np.savez_compressed(ckpt_path, X=X_ckpt, y=y_ckpt)
                    print(f"  [checkpoint] {last_mark:,} samples -> {ckpt_path}", flush=True)

            if args.bench and total_gen >= args.bench:
                print(f"\n[benchmark] reached {args.bench} samples, exiting.")
                break

            if total_gen >= args.n_samples:
                break

    pool.terminate()
    pool.join()
    wall_time = time.time() - t_start

    if len(all_X) == 0:
        print("Error: no samples generated!")
        sys.exit(1)

    X_final = np.concatenate(all_X).astype(np.float32)
    y_final = np.concatenate(all_y).astype(np.int32)
    n_red_final = np.concatenate(all_n_red).astype(np.int32)
    n_blue_final = np.concatenate(all_n_blue).astype(np.int32)

    actual_n = min(args.n_samples, len(y_final))
    X_final = X_final[:actual_n]
    y_final = y_final[:actual_n]
    n_red_final = n_red_final[:actual_n]
    n_blue_final = n_blue_final[:actual_n]

    r_final = X_final[:, :4 * STATE_SIZE].astype(np.uint8)
    b_final = X_final[:, 4 * STATE_SIZE:8 * STATE_SIZE].astype(np.uint8)

    np.savez_compressed(args.output,
                        r=r_final, b=b_final, targets=y_final,
                        n_red=n_red_final, n_blue=n_blue_final)
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print(f"Total samples: {len(y_final):,}")
    print(f"Total time: {wall_time / 3600:.2f} h ({wall_time:.0f}s)")
    if wall_time > 0:
        print(f"Average throughput: {len(y_final) / wall_time:.1f} sol/s")
    print(f"Output file: {args.output}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"r shape: {r_final.shape}, b shape: {b_final.shape}, targets shape: {y_final.shape}")
    if len(y_final) > 0:
        print(f"y (u2) range: [{np.min(y_final)}, {np.max(y_final)}], "
              f"mu={np.mean(y_final):.1f}, sigma={np.std(y_final):.1f}")
        print(f"initial red: [{np.min(n_red_final)},{np.max(n_red_final)}], "
              f"mu={np.mean(n_red_final):.1f}")
        print(f"initial blue: [{np.min(n_blue_final)},{np.max(n_blue_final)}], "
              f"mu={np.mean(n_blue_final):.1f}")

    if len(y_final) < args.n_samples * 0.8:
        shortfall = args.n_samples - len(y_final)
        print(f"\nWARNING:  shortfall: {shortfall:,} samples "
              f"(generated {len(y_final)/args.n_samples*100:.1f}%)")
        print(f"Suggestions:")
        print(f"  1. increase --blue-configs (current: {args.blue_configs})")
        print(f"  2. increase --variants-per-blue (current: {args.variants_per_blue})")
        print(f"  3. increase --max-red (current: {args.max_red})")

    if args.bench:
        print(f"\n[benchmark] throughput: {len(y_final) / wall_time:.1f} sol/s")
        eta_100w = (1_000_000 / (len(y_final) / wall_time)) / 3600
        print(f"[benchmark] 1M ETA: {eta_100w:.1f} h")


if __name__ == '__main__':
    mp.freeze_support()
    main()
