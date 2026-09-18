'Module object.'
import sys, os, time, argparse, hashlib
import numpy as np
import multiprocessing as mp

_current_dir = os.path.dirname(os.path.abspath(__file__))
_milp_dir = os.path.dirname(os.path.abspath(__file__))
if _milp_dir not in sys.path:
    sys.path.insert(0, _milp_dir)

from Keccak import *
from operation import Bit
import gurobipy as gp
from gurobipy import GRB

# ============================================================

# ============================================================
N_Z, N_Y, N_X = 64, 5, 5
INIT_FEAT_DIM = N_Z * N_X * 2        # 640
PI1_FEAT_DIM  = N_Z * N_Y * N_X * 2  # 3200
TOTAL_FEAT_DIM = INIT_FEAT_DIM + PI1_FEAT_DIM  # 3840

Y2_VALID_COLS = {(z, x) for z in range(N_Z) for x in range(3)
                 if not (x == 2 and z >= 60)}

RHO_BOX = [[0, 1, 62, 28, 27], [36, 44, 6, 55, 20],
           [3, 10, 43, 25, 39], [41, 45, 15, 21, 8],
           [18, 2, 61, 56, 14]]

# ============================================================

# ============================================================

def deterministic_full_forward(init_col_red, init_col_blue, use_pi1=False):
    'Function deterministic full forward.'
    state = np.zeros((N_Z, N_Y, N_X, 3), dtype=np.int8)
    for z in range(N_Z):
        for x in range(N_X):
            is_red = init_col_red[z, x]
            is_blue = init_col_blue[z, x]
            if is_red and is_blue:
                is_red = 0
            if is_red:
                state[z, 0, x] = [0, 1, 0]; state[z, 1, x] = [0, 1, 0]
            elif is_blue:
                state[z, 0, x] = [0, 0, 1]; state[z, 1, x] = [0, 0, 1]
            if (z, x) in Y2_VALID_COLS:
                if is_red:
                    state[z, 2, x] = [0, 1, 0]
                elif is_blue:
                    state[z, 2, x] = [0, 0, 1]

    def _extract_r(s3d, c):
        return s3d[:, :, :, c].astype(np.float32).flatten()

    def _theta(state_in):
        C = np.zeros((N_Z, N_X, 3), dtype=np.int8)
        for z in range(N_Z):
            for x in range(N_X):
                for k in range(3):
                    C[z,x,k] = (int(state_in[z,0,x,k])+int(state_in[z,1,x,k])+
                                int(state_in[z,2,x,k])+int(state_in[z,3,x,k])+
                                int(state_in[z,4,x,k])) % 2
        D = np.zeros((N_Z, N_X, 3), dtype=np.int8)
        for z in range(N_Z):
            for x in range(N_X):
                for k in range(3):
                    D[z,x,k] = (int(C[z,(x-1)%5,k])+int(C[(z-1)%64,(x+1)%5,k])) % 2
        out = np.zeros_like(state_in)
        for z in range(N_Z):
            for y in range(N_Y):
                for x in range(N_X):
                    for k in range(3):
                        out[z,y,x,k] = (int(state_in[z,y,x,k])+int(D[z,x,k])) % 2
        return out

    def _rho(state_in):
        out = np.zeros_like(state_in)
        for z in range(N_Z):
            for y in range(N_Y):
                for x in range(N_X):
                    out[z,y,x] = state_in[(z-RHO_BOX[y][x])%N_Z, y, x]
        return out

    def _pi(state_in):
        out = np.zeros_like(state_in)
        for z in range(N_Z):
            for y in range(N_Y):
                for x in range(N_X):
                    out[z,y,x] = state_in[z, x, (x+3*y)%5]
        return out

    def _chi(state_in):
        out = np.zeros_like(state_in)
        for z in range(N_Z):
            for y in range(N_Y):
                for x in range(N_X):
                    x1, x2 = (x+1) % 5, (x+2) % 5
                    b1, b2 = state_in[z, y, x1], state_in[z, y, x2]
                    orig = state_in[z,y,x]
                    r_and = int(b1[1]) and int(b2[1])
                    b_and = int(b1[2]) and int(b2[2])
                    cross_ul = (int(b1[1]) and int(b2[2])) or (int(b1[2]) and int(b2[1]))
                    ul_and_any = (int(b1[0]) or int(b2[0])) and (int(b1[1]) or int(b1[2]) or int(b2[1]) or int(b2[2]))
                    ul_and = cross_ul or ul_and_any
                    out[z,y,x,0] = int(int(orig[0]) != ul_and)
                    out[z,y,x,1] = (int(orig[1])+r_and) % 2
                    out[z,y,x,2] = (int(orig[2])+b_and) % 2
        return out

    # Round 0
    t0 = _theta(state); r0 = _rho(t0); p0 = _pi(r0)
    pi0_r = _extract_r(p0, 1); pi0_b = _extract_r(p0, 2)
    c0 = _chi(p0)

    # Round 1
    t1 = _theta(c0); r1 = _rho(t1); p1 = _pi(r1); c1 = _chi(p1)

    if use_pi1:
        pi1_r = _extract_r(p1, 1); pi1_b = _extract_r(p1, 2)
    else:
        pi0_r = _extract_r(p0, 1); pi0_b = _extract_r(p0, 2)


    u2 = 0
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                ul, r, b = c1[z,y,x]
                if ul and not r and not b:
                    u2 += 1

    feat = np.concatenate([
        init_col_red.astype(np.float32).flatten(),
        init_col_blue.astype(np.float32).flatten(),
        pi1_r if use_pi1 else pi0_r,
        pi1_b if use_pi1 else pi0_b,
    ])
    return feat, int(u2), {'n_red': int(np.sum(init_col_red)), 'n_blue': int(np.sum(init_col_blue))}


# ============================================================

# ============================================================

def generate_neutral_blue_masks(n_masks, n_blue_range=(22, 26), seed=42):
    'Function generate neutral blue masks.'
    rng = np.random.RandomState(seed)
    masks = []
    for i in range(n_masks):
        n_blue = rng.randint(*n_blue_range)
        mask = np.zeros((N_Z, N_X), dtype=np.int8)
        row_x0 = {z: False for z in range(N_Z)}
        row_x1 = {z: False for z in range(N_Z)}
        selected = 0
        candidates = [(z, x) for z in range(N_Z) for x in range(N_X)]
        rng.shuffle(candidates)
        for z, x in candidates:
            if selected >= n_blue:
                break
            if x == 0 and row_x1[z]:
                continue
            if x == 1 and row_x0[z]:
                continue
            mask[z, x] = 1
            selected += 1
            if x == 0:
                row_x0[z] = True
            elif x == 1:
                row_x1[z] = True
        if selected == n_blue:
            masks.append(mask)
    return masks


def load_blue_from_file(filepath, max_count=None):
    'Function load blue from file.'
    if not os.path.exists(filepath):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("blue_data", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solutions = mod.all_solutions
    if max_count and len(solutions) > max_count:
        solutions = solutions[:max_count]
    masks = []
    for s in solutions:
        m = np.zeros((N_Z, N_X), dtype=np.int8)
        for z in range(min(N_Z, len(s))):
            for x in range(min(N_X, len(s[z]))):
                if s[z][x] > 0.5:
                    m[z, x] = 1
        masks.append(m)
    return masks


# ============================================================

# ============================================================

def worker_generate(params):
    'Function worker generate.'
    seed, blue_mask, n_target, time_limit, gap, max_red, total_max, worker_id, use_pi1 = params

    X_list, y_list = [], []
    excluded_reds = []
    t0 = time.time()
    n_gen = 0
    n_iter = 0
    n_err = 0
    n_blue = int(np.sum(blue_mask))

    while n_gen < n_target and n_iter < n_target * 3:
        try:
            model = gp.Model(f"StgB_w{worker_id}")
            model.setParam('Seed', seed + n_iter)
            model.setParam('MIPGap', gap)
            model.setParam('TimeLimit', time_limit)
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            model.setParam('MIPFocus', 1)

            init_state = [[[Bit(model, f"z{z}y{y}x{x}", "c")
                            for x in range(N_X)] for y in range(N_Y)]
                          for z in range(N_Z)]

            red_vars = []
            red_pos_lookup = {}  # (z,x) -> red var

            for z in range(N_Z):
                for x in range(N_X):
                    is_blue = bool(blue_mask[z, x])
                    if is_blue:
                        b0 = Bit(model, f"b{z}_0_{x}", (0, 0, 1))
                    else:
                        b0 = Bit(model, f"r{z}_0_{x}", (0, '*', 0))
                        red_vars.append(b0.r)
                        red_pos_lookup[(z, x)] = b0.r
                    init_state[z][0][x] = b0

                    b1 = Bit(model, f"cpy{z}_1_{x}", (0, '*', '*'))
                    model.addConstr(b1.r == b0.r); model.addConstr(b1.b == b0.b)
                    init_state[z][1][x] = b1

                    if (z, x) in Y2_VALID_COLS:
                        b2 = Bit(model, f"cpy{z}_2_{x}", (0, '*', '*'))
                        model.addConstr(b2.r == b0.r); model.addConstr(b2.b == b0.b)
                        init_state[z][2][x] = b2

            sum_red = gp.quicksum(red_vars) if red_vars else 0


            for prev_red in excluded_reds:
                model.addConstr(gp.quicksum(red_pos_lookup[p] for p in prev_red) <= len(prev_red) - 1)


            model.addConstr(sum_red >= n_blue)
            model.addConstr(sum_red <= max_red)
            model.addConstr(sum_red + n_blue <= total_max)


            theta0, C0, D0, tv0 = create_first_theta_operation(model, init_state, 't0')
            rho0 = rho(theta0)
            pi0 = pi(rho0)

            for z in range(N_Z):
                for y in range(N_Y):
                    for xi in range(N_X):
                        model.addConstr(pi0[z][y][xi].r + pi0[z][y][(xi+1) % N_X].r <= 1)


            # Deterministic objective: minimise the number of marked bits after the
            # first round. Diversity of the returned configurations comes from the
            # exclusion constraints on previously returned red patterns above, not
            # from a randomised objective.
            obj = gp.quicksum(
                pi0[z][y][xi].r + pi0[z][y][xi].b
                for z in range(N_Z) for y in range(N_Y) for xi in range(N_X))
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()

            if model.SolCount > 0:
                init_red = np.zeros((N_Z, N_X), dtype=np.int8)
                init_blue_arr = np.zeros((N_Z, N_X), dtype=np.int8)
                red_positions = []
                for z in range(N_Z):
                    for x in range(N_X):
                        bit = init_state[z][0][x]
                        rv = int(bit.r.X) if isinstance(bit.r, gp.Var) else int(bit.r)
                        bv = int(bit.b.X) if isinstance(bit.b, gp.Var) else int(bit.b)
                        init_red[z, x] = rv
                        init_blue_arr[z, x] = bv
                        if rv:
                            red_positions.append((z, x))

                feat, u2, stats = deterministic_full_forward(init_red, init_blue_arr, use_pi1)
                X_list.append(feat)
                y_list.append(u2)
                n_gen += 1

                if red_positions:
                    excluded_reds.append(red_positions)
            else:
                if model.status == GRB.INFEASIBLE:
                    break

        except gp.GurobiError as ge:
            n_err += 1
            if n_err <= 3:
                print(f"  [W{worker_id}] GurobiError #{n_err}: {ge}", file=sys.stderr, flush=True)
            n_iter += 1; continue
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"  [W{worker_id}] Exception #{n_err}: {e}", file=sys.stderr, flush=True)
            n_iter += 1; continue
        finally:
            try:
                if 'model' in locals(): model.dispose()
            except Exception: pass
        n_iter += 1

    elapsed = time.time() - t0
    if n_err > 0:
        print(f"  [W{worker_id}] {n_gen} samples, {n_err} errors, {elapsed:.0f}s", flush=True)
    if len(X_list) == 0:
        return np.zeros((0, TOTAL_FEAT_DIM), dtype=np.uint8), np.zeros(0, dtype=np.int16), 0, elapsed

    X_arr = np.stack(X_list).astype(np.uint8)
    y_arr = np.array(y_list, dtype=np.int16)
    return X_arr, y_arr, n_gen, elapsed


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SHA3-384 Stage B training data generation')
    parser.add_argument('--n-total', type=int, default=200000, help='target total number of samples')
    parser.add_argument('--n-procs', type=int, default=8, help='number of parallel processes')
    parser.add_argument('--n-per-blue', type=int, default=6, help='number of red variants per blue scheme')
    parser.add_argument('--time-limit', type=int, default=5, help='MILP time limit (seconds)')
    parser.add_argument('--gap', type=float, default=0.7, help='MIPGap')
    parser.add_argument('--max-red', type=int, default=45, help='max red bits (optimal for b15-20_t80)')
    parser.add_argument('--total-max', type=int, default=80, help='max red+blue total (optimal for b15-20_t80)')
    parser.add_argument('--blue-min', type=int, default=15, help='min blue bits')
    parser.add_argument('--blue-max', type=int, default=20, help='max blue bits')
    parser.add_argument('--blue-file', type=str, default=None, help='blue scheme .py file')
    parser.add_argument('--random-blue', action='store_true', help='use random neutral blue')
    parser.add_argument('--n-blue-masks', type=int, default=5000, help='number of random blue schemes')
    parser.add_argument('--output', type=str, default=None, help='output .npz path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='dry run: generate only 10 samples to verify the pipeline')
    parser.add_argument('--pi1-features', action='store_true', default=True,
                        help='output pi1 features (default; the paper reads the red/blue masks after the second pi layer)')
    parser.add_argument('--pi0-features', action='store_true',
                        help='output pi0 features (first pi layer) instead of pi1')
    args = parser.parse_args()

    use_pi1 = not args.pi0_features


    blue_masks = None
    if args.blue_file:
        blue_masks = load_blue_from_file(args.blue_file)
        if blue_masks:
            print(f"Loaded {len(blue_masks)} blue schemes from {args.blue_file}")
    if blue_masks is None or args.random_blue:
        n_bm = max(args.n_blue_masks, args.n_total // args.n_per_blue + 1)
        blue_masks = generate_neutral_blue_masks(n_bm, n_blue_range=(args.blue_min, args.blue_max),
                                                  seed=args.seed)
        print(f"Generated {len(blue_masks)} neutral random blue schemes (blue={args.blue_min}-{args.blue_max})")


    if args.dry_run:
        args.n_total = 10
        args.n_per_blue = 2
        args.n_procs = 1
        print(f" Dry run mode: {args.n_total} samples, 1 process")


    if args.output:
        out_path = args.output
    else:
        suffix = '_pi1' if use_pi1 else ''
        out_path = os.path.join(os.path.dirname(os.path.dirname(_current_dir)),
                                'data', 'sha3_384',
                                f'sha3_384_train_stageB{suffix}.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    n_blue = len(blue_masks)
    n_per_blue = args.n_per_blue
    n_workers = min(args.n_procs, n_blue)

    print(f"\n=== SHA3-384 Stage B Data Generation ===")
    print(f"  Blue schemes: {n_blue}, {n_per_blue} red variants each")
    print(f"  Target: {args.n_total} samples, {n_workers} processes")
    print(f"  MILP: gap={args.gap}, time_limit={args.time_limit}s")
    print(f"  Constraints: red >= blue, red <= {args.max_red}, red+blue <= {args.total_max}")

    print(f"  Feature type: {'pi1' if use_pi1 else 'pi0'} (default pi0)")


    rng = np.random.RandomState(args.seed)
    blue_indices = list(range(n_blue))
    rng.shuffle(blue_indices)
    chunks = np.array_split(blue_indices, n_workers)

    tasks = []
    for wid, chunk in enumerate(chunks):
        for bidx in chunk:
            tasks.append((args.seed + bidx * 1000 + wid, blue_masks[bidx],
                           n_per_blue, args.time_limit, args.gap,
                           args.max_red, args.total_max, wid, use_pi1))


    t_start = time.time()
    all_X, all_y = [], []
    total_gen = 0

    with mp.Pool(n_workers) as pool:
        for X_chunk, y_chunk, n_gen, elapsed in pool.imap_unordered(worker_generate, tasks):
            if n_gen > 0:
                all_X.append(X_chunk)
                all_y.append(y_chunk)
                total_gen += n_gen
            if total_gen % 500 == 0 and total_gen > 0:
                print(f"  Generated {total_gen} samples...")

    t_total = time.time() - t_start

    if len(all_X) == 0:
        print("x No samples generated"); return

    X_full = np.concatenate(all_X)[:args.n_total]
    y_full = np.concatenate(all_y)[:args.n_total]
    actual = len(X_full)

    print(f"\nOK Done: {actual} samples, took {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"  Rate: {actual/t_total:.0f} samples/s")
    print(f"  y(u2) range: [{y_full.min()}, {y_full.max()}], mean={y_full.mean():.1f}, std={y_full.std():.1f}")

    np.savez_compressed(out_path, X=X_full, y=y_full)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
