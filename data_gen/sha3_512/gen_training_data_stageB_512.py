'Module object.'
import sys, os, time, argparse, numpy as np, multiprocessing as mp

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


CONTROLLABLE_COLS = [(z, x) for z in range(N_Z) for x in range(4)
                     if not (x == 3 and z >= 28)]
N_CONTROLLABLE = len(CONTROLLABLE_COLS)  # 220

PADDING_COLS_Y0 = {(z, 3) for z in range(28, N_Z)}
PADDING_COLS_Y1 = {(z, 3) for z in range(28, N_Z)}


INIT_FEAT_DIM = N_Z * 4 * 2
PI1_FEAT_DIM  = N_Z * N_Y * N_X * 2
TOTAL_FEAT_DIM = INIT_FEAT_DIM + PI1_FEAT_DIM  # 3712

RHO_BOX = [[0, 1, 62, 28, 27], [36, 44, 6, 55, 20],
           [3, 10, 43, 25, 39], [41, 45, 15, 21, 8],
           [18, 2, 61, 56, 14]]


DEFAULT_MIN_BLUE, DEFAULT_MAX_BLUE = 20, 35
DEFAULT_TOTAL_MAX = 100
RED_BLUE_GAP = 5
DEFAULT_TIME_LIMIT, DEFAULT_N_PROCS, DEFAULT_GAP = 10, 16, 0.5


# ============================================================

# ============================================================

def deterministic_full_forward_stageB(init_col_red, init_col_blue):
    'Function deterministic full forward stageB.'
    state = np.zeros((N_Z, N_Y, N_X, 3), dtype=np.int8)
    for z in range(N_Z):
        for x in range(4):
            is_red, is_blue = init_col_red[z, x], init_col_blue[z, x]
            if is_red and is_blue:
                is_red = 0
            if (z, x) not in PADDING_COLS_Y0:
                if is_red:   state[z, 0, x] = [0, 1, 0]
                elif is_blue: state[z, 0, x] = [0, 0, 1]
            if (z, x) not in PADDING_COLS_Y1:
                if is_red:   state[z, 1, x] = [0, 1, 0]
                elif is_blue: state[z, 1, x] = [0, 0, 1]

    def _extract_r(s3d, c):
        return s3d[:, :, :, c].astype(np.float32).flatten()

    # Round 0: theta0 + rho + pi + chi
    C0 = np.zeros((N_Z, N_X, 3), dtype=np.int8)
    for z in range(N_Z):
        for x in range(N_X):
            for k in range(3):
                C0[z,x,k] = (int(state[z,0,x,k])+int(state[z,1,x,k])+
                             int(state[z,2,x,k])+int(state[z,3,x,k])+
                             int(state[z,4,x,k])) % 2
    D0 = np.zeros((N_Z, N_X, 3), dtype=np.int8)
    for z in range(N_Z):
        for x in range(N_X):
            for k in range(3):
                D0[z,x,k] = (int(C0[z,(x-1)%5,k])+int(C0[(z-1)%64,(x+1)%5,k])) % 2
    theta0 = np.zeros_like(state)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                for k in range(3):
                    theta0[z,y,x,k] = (int(state[z,y,x,k])+int(D0[z,x,k])) % 2

    rho0 = np.zeros_like(theta0)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                rho0[z,y,x] = theta0[(z-RHO_BOX[y][x])%N_Z, y, x]
    pi0 = np.zeros_like(rho0)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                pi0[z,y,x] = rho0[z, x, (x+3*y)%5]

    chi0 = np.zeros_like(pi0)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                x1, x2 = (x+1) % 5, (x+2) % 5
                b1, b2 = pi0[z, y, x1], pi0[z, y, x2]
                orig = pi0[z,y,x]
                r_and = int(b1[1]) and int(b2[1])
                b_and = int(b1[2]) and int(b2[2])
                cross_ul = (int(b1[1]) and int(b2[2])) or (int(b1[2]) and int(b2[1]))
                ul_and_any = (int(b1[0]) or int(b2[0])) and (int(b1[1]) or int(b1[2]) or int(b2[1]) or int(b2[2]))
                ul_and = cross_ul or ul_and_any
                chi0[z,y,x,0] = int(int(orig[0]) != ul_and)
                chi0[z,y,x,1] = (int(orig[1])+r_and) % 2
                chi0[z,y,x,2] = (int(orig[2])+b_and) % 2

    # Round 1: theta1 + rho + pi + chi
    C1 = np.zeros((N_Z, N_X, 3), dtype=np.int8)
    for z in range(N_Z):
        for x in range(N_X):
            for k in range(3):
                C1[z,x,k] = (int(chi0[z,0,x,k])+int(chi0[z,1,x,k])+
                             int(chi0[z,2,x,k])+int(chi0[z,3,x,k])+
                             int(chi0[z,4,x,k])) % 2
    D1 = np.zeros((N_Z, N_X, 3), dtype=np.int8)
    for z in range(N_Z):
        for x in range(N_X):
            for k in range(3):
                D1[z,x,k] = (int(C1[z,(x-1)%5,k])+int(C1[(z-1)%64,(x+1)%5,k])) % 2
    theta1 = np.zeros_like(chi0)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                for k in range(3):
                    theta1[z,y,x,k] = (int(chi0[z,y,x,k])+int(D1[z,x,k])) % 2

    rho1 = np.zeros_like(theta1)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                rho1[z,y,x] = theta1[(z-RHO_BOX[y][x])%N_Z, y, x]
    pi1 = np.zeros_like(rho1)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                pi1[z,y,x] = rho1[z, x, (x+3*y)%5]

    chi1 = np.zeros_like(pi1)
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                x1, x2 = (x+1) % 5, (x+2) % 5
                b1, b2 = pi1[z, y, x1], pi1[z, y, x2]
                orig = pi1[z,y,x]
                r_and = int(b1[1]) and int(b2[1])
                b_and = int(b1[2]) and int(b2[2])
                cross_ul = (int(b1[1]) and int(b2[2])) or (int(b1[2]) and int(b2[1]))
                ul_and_any = (int(b1[0]) or int(b2[0])) and (int(b1[1]) or int(b1[2]) or int(b2[1]) or int(b2[2]))
                ul_and = cross_ul or ul_and_any
                chi1[z,y,x,0] = int(int(orig[0]) != ul_and)
                chi1[z,y,x,1] = (int(orig[1])+r_and) % 2
                chi1[z,y,x,2] = (int(orig[2])+b_and) % 2

    # u2
    u2 = 0
    for z in range(N_Z):
        for y in range(N_Y):
            for x in range(N_X):
                ul, r, b = chi1[z,y,x]
                if ul and not r and not b:
                    u2 += 1


    feat = np.concatenate([
        init_col_red.astype(np.float32).flatten(),
        init_col_blue.astype(np.float32).flatten(),
        _extract_r(pi1, 1),
        _extract_r(pi1, 2),
    ])

    stats = {'n_red_init': int(np.sum(init_col_red)),
             'n_blue_init': int(np.sum(init_col_blue))}
    return feat, int(u2), stats


# ============================================================

# ============================================================

def random_blue_mask_stageB(seed, min_blue=DEFAULT_MIN_BLUE, max_blue=DEFAULT_MAX_BLUE):
    'Function random blue mask stageB.'
    rng = np.random.RandomState(seed)
    ALL_POS = [(z, x) for z in range(N_Z) for x in range(4)
               if (z, x) not in PADDING_COLS_Y0]
    for _ in range(100):
        n_blue = rng.randint(min_blue, max_blue + 1)
        indices = rng.choice(len(ALL_POS), size=n_blue, replace=False)
        mask = np.zeros((N_Z, 4), dtype=np.int8)
        blue_set = set()
        for idx in indices:
            z, x = ALL_POS[idx]
            mask[z, x] = 1
            blue_set.add((z, x))


        if _check_chi_feasible(blue_set):
            return mask, n_blue
    # fallback
    n_blue = rng.randint(min_blue, max_blue + 1)
    indices = rng.choice(len(ALL_POS), size=n_blue, replace=False)
    mask = np.zeros((N_Z, 4), dtype=np.int8)
    for idx in indices:
        z, x = ALL_POS[idx]
        mask[z, x] = 1
    return mask, n_blue


def _check_chi_feasible(blue_set):
    'Function check chi feasible.'
    state = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(64)]
    for z, x in blue_set:
        state[z][0][x] = 1
        if (z, x) not in PADDING_COLS_Y1:
            state[z][1][x] = 1

    # rho
    ns = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(64)]
    for z in range(64):
        for y in range(5):
            for x in range(5):
                ns[z][y][x] = state[(z-RHO_BOX[y][x])%64][y][x]
    # pi
    ns2 = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(64)]
    for z in range(64):
        for y in range(5):
            for x in range(5):
                ns2[z][y][x] = ns[z][x][(x+3*y)%5]
    # check
    for z in range(64):
        for y in range(5):
            if ns2[z][y][0] and ns2[z][y][1]:
                return False
    return True


# ============================================================

# ============================================================

def worker_generate_stageB(params):
    'Function worker generate stageB.'
    (seed, blue_mask, minR, maxR, total_max,
     n_target, time_limit, gap, worker_id) = params

    X_list, y_list, meta_list = [], [], []
    add_r = []
    t0 = time.time()
    n_gen, n_iter = 0, 0

    while n_gen < n_target:
        try:
            model = gp.Model(f"StageB512_w{worker_id}")
            model.setParam('Seed', seed + n_iter)
            model.setParam('MIPGap', gap)
            model.setParam('TimeLimit', time_limit)
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            model.setParam('MIPFocus', 1)


            initial_state = [[[Bit(model, f"sB_z{z}y{y}x{x}", "c")
                               for x in range(N_X)] for y in range(N_Y)]
                             for z in range(N_Z)]

            red_bits, blue_bits = [], []
            n_blue = 0
            for z in range(N_Z):
                for x in range(4):
                    is_blue = bool(blue_mask[z, x])
                    if (z, x) in PADDING_COLS_Y0:
                        initial_state[z][0][x] = Bit(model, f"sB_z{z}y0x{x}", "c")
                    elif is_blue:
                        initial_state[z][0][x] = Bit(model, f"sB_z{z}y0x{x}", (0, 0, 1))
                    else:
                        initial_state[z][0][x] = Bit(model, f"sB_z{z}y0x{x}", (0, '*', 0))

                    if (z, x) in PADDING_COLS_Y1:
                        initial_state[z][1][x] = Bit(model, f"sB_z{z}y1x{x}", "c")
                    else:
                        initial_state[z][1][x] = Bit(model, f"sB_z{z}y1x{x}", (0, '*', '*'))
                        model.addConstr(initial_state[z][1][x].r == initial_state[z][0][x].r)
                        model.addConstr(initial_state[z][1][x].b == initial_state[z][0][x].b)

                for x in range(4):
                    if (z, x) not in PADDING_COLS_Y0:
                        if blue_mask[z, x]:
                            blue_bits.append(initial_state[z][0][x].b)
                            n_blue += 1
                        else:
                            red_bits.append(initial_state[z][0][x].r)


            for one_r in add_r:
                model.addConstr(
                    gp.quicksum([initial_state[z][0][x].r for z, x in one_r])
                    <= len(one_r) - 1)


            sum_red = gp.quicksum(red_bits) if red_bits else 0
            sum_blue = gp.quicksum(blue_bits) if blue_bits else n_blue
            if isinstance(sum_blue, int):
                model.addConstr(sum_red >= max(minR, sum_blue + RED_BLUE_GAP))
            else:
                model.addConstr(sum_red >= sum_blue + RED_BLUE_GAP)
            model.addConstr(sum_red <= maxR)
            if isinstance(sum_blue, int):
                model.addConstr(sum_red + sum_blue <= total_max)
            else:
                model.addConstr(sum_red + sum_blue <= total_max)

            # ==== Round 0: theta0(identity)->rho0->pi0 ====
            theta0, _, _, _ = create_first_theta_operation(model, initial_state, 'sB_r0')
            rho0 = rho(theta0)
            pi0 = pi(rho0)


            for z in range(N_Z):
                for y in range(N_Y):
                    for xi in range(N_X):
                        model.addConstr(pi0[z][y][xi].r + pi0[z][y][(xi+1) % N_X].r <= 1)
                        model.addConstr(pi0[z][y][xi].b + pi0[z][y][(xi+1) % N_X].b <= 1)







            # Deterministic objective: minimise the number of marked bits after the
            # first round. Diversity of the returned configurations comes from the
            # exclusion constraints on previously returned red patterns above, not
            # from a randomised objective.
            obj = gp.quicksum(
                pi0[z][y][x].r + pi0[z][y][x].b
                for z in range(N_Z) for y in range(N_Y) for x in range(N_X))
            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            if model.SolCount > 0:
                init_red = np.zeros((N_Z, 4), dtype=np.int8)
                init_blue_arr = np.zeros((N_Z, 4), dtype=np.int8)
                red_positions = []
                for z in range(N_Z):
                    for x in range(4):
                        if (z, x) in PADDING_COLS_Y0:
                            continue
                        bit = initial_state[z][0][x]
                        rv = int(bit.r.X) if isinstance(bit.r, gp.Var) else int(bit.r)
                        bv = int(bit.b.X) if isinstance(bit.b, gp.Var) else int(bit.b)
                        init_red[z, x] = rv
                        init_blue_arr[z, x] = bv
                        if rv:
                            red_positions.append((z, x))

                feat, u2, stats = deterministic_full_forward_stageB(
                    init_red, init_blue_arr)
                X_list.append(feat)
                y_list.append(u2)
                meta_list.append((stats['n_red_init'], stats['n_blue_init']))
                n_gen += 1
                if red_positions:
                    add_r.append(red_positions)
            else:
                if model.status == GRB.INFEASIBLE:
                    break

            n_iter += 1
        except gp.GurobiError as ge:
            if n_gen < 3:
                print(f"  [W{worker_id}] GurobiError: {ge}", file=sys.stderr, flush=True)
            n_iter += 1
            continue
        except Exception as e:
            if n_gen < 3:
                print(f"  [W{worker_id}] Exception: {e}", file=sys.stderr, flush=True)
            n_iter += 1
            continue
        finally:
            try:
                if 'model' in locals():
                    model.dispose()
            except Exception:
                pass

    elapsed = time.time() - t0
    if not X_list:
        return (np.zeros((0, TOTAL_FEAT_DIM), dtype=np.uint8),
                np.zeros(0, dtype=np.int16),
                np.zeros(0, dtype=[('n_red', 'i2'), ('n_blue', 'i2')]),
                0, elapsed)
    X_arr = np.stack(X_list).astype(np.uint8)
    y_arr = np.array(y_list, dtype=np.int16)
    meta_arr = np.array(meta_list, dtype=[('n_red', 'i2'), ('n_blue', 'i2')])
    return X_arr, y_arr, meta_arr, n_gen, elapsed


# ============================================================

# ============================================================

def worker_blue_batch_stageB(params):
    (seed, blue_masks, minR, maxR, total_max,
     n_per_blue, time_limit, gap, worker_id) = params
    all_X, all_y, all_meta = [], [], []
    total_gen = 0
    t0 = time.time()
    for bi, blue_mask in enumerate(blue_masks):
        if total_gen >= n_per_blue * len(blue_masks):
            break
        local_seed = seed + bi * 10000
        Xc, yc, mc, ng, _ = worker_generate_stageB(
            (local_seed, blue_mask, minR, maxR, total_max,
             n_per_blue, time_limit, gap, worker_id))
        if ng > 0:
            all_X.append(Xc); all_y.append(yc); all_meta.append(mc)
            total_gen += ng
    elapsed = time.time() - t0
    if not all_X:
        return (np.zeros((0, TOTAL_FEAT_DIM), dtype=np.uint8),
                np.zeros(0, dtype=np.int16),
                np.zeros(0, dtype=[('n_red', 'i2'), ('n_blue', 'i2')]),
                0, elapsed)
    X_arr = np.concatenate(all_X).astype(np.uint8)
    y_arr = np.concatenate(all_y).astype(np.int16)
    meta_arr = np.concatenate(all_meta)
    return X_arr, y_arr, meta_arr, total_gen, elapsed


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SHA3-512 Stage B training data generation (pi1-layer features)')
    parser.add_argument('--n-samples', type=int, default=200_000)
    parser.add_argument('--n-procs', type=int, default=DEFAULT_N_PROCS)
    parser.add_argument('--min-blue', type=int, default=DEFAULT_MIN_BLUE)
    parser.add_argument('--max-blue', type=int, default=DEFAULT_MAX_BLUE)
    parser.add_argument('--min-red-gap', type=int, default=RED_BLUE_GAP)
    parser.add_argument('--max-red', type=int, default=160)
    parser.add_argument('--total-max', type=int, default=DEFAULT_TOTAL_MAX)
    parser.add_argument('--time-limit', type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument('--gap', type=float, default=DEFAULT_GAP)
    parser.add_argument('--n-per-blue', type=int, default=50)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--checkpoint-every', type=int, default=50000)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-samples', type=int, default=50)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(_current_dir)),
                              'data', 'sha3_512')
    os.makedirs(output_dir, exist_ok=True)
    if args.output is None:
        args.output = os.path.join(output_dir,
                                   f'sha3_512_train_stageB_{args.n_samples}.npz')

    print("=" * 60)
    print("SHA3-512 Stage B: pi1-Layer Training Data Generation")
    print("=" * 60)
    print(f"Target samples: {args.n_samples:,}")
    print(f"Parallel processes: {args.n_procs}")
    print(f"Params: B[{args.min_blue},{args.max_blue}] "
          f"R>=B+{args.min_red_gap} R+B<={args.total_max}")
    print(f"MILP: 1 round to pi0 (deterministic objective, exclusion cuts), "
          f"gap={args.gap}, tlimit={args.time_limit}s")
    print(f"Features: init(512) + pi1_r(1600) + pi1_b(1600) = {TOTAL_FEAT_DIM}")
    print(f"pi1 features from deterministic propagation (incl. chi nonlinearity)")
    print(f"Samples per blue scheme: {args.n_per_blue}")
    print()

    if args.dry_run:
        args.n_samples = args.max_samples
        args.checkpoint_every = 10


    n_blue_needed = max(100, args.n_samples // args.n_per_blue // args.n_procs * 2)
    print(f"Generating {n_blue_needed} neutral blue schemes...")
    blue_masks = []
    for i in range(n_blue_needed):
        mask, _ = random_blue_mask_stageB(10000 + i, args.min_blue, args.max_blue)
        blue_masks.append(mask)
    print(f"  Generated: {len(blue_masks)} schemes")


    n_blue = len(blue_masks)
    blues_per_proc = max(1, n_blue // args.n_procs)
    per_worker_target = args.n_samples // args.n_procs + 1

    tasks = []
    for pid in range(args.n_procs):
        start = pid * blues_per_proc
        end = min(start + blues_per_proc, n_blue)
        if start >= n_blue:
            break
        proc_blues = blue_masks[start:end]
        minR = args.min_blue + args.min_red_gap
        tasks.append((42 + pid * 1000,
                      proc_blues, minR, args.max_red, args.total_max,
                      max(1, per_worker_target // max(1, len(proc_blues))),
                      args.time_limit, args.gap, pid))


    X_mmap_path = args.output + '.X_tmp'
    y_mmap_path = args.output + '.y_tmp'
    nr_mmap_path = args.output + '.nr_tmp'
    nb_mmap_path = args.output + '.nb_tmp'

    X_mmap = np.memmap(X_mmap_path, dtype='uint8', mode='w+',
                       shape=(args.n_samples, TOTAL_FEAT_DIM))
    y_mmap = np.memmap(y_mmap_path, dtype='int16', mode='w+',
                       shape=(args.n_samples,))
    nr_mmap = np.memmap(nr_mmap_path, dtype='int16', mode='w+',
                        shape=(args.n_samples,))
    nb_mmap = np.memmap(nb_mmap_path, dtype='int16', mode='w+',
                        shape=(args.n_samples,))

    write_pos = 0
    last_ckpt_mark = 0
    t_start = time.time()

    with mp.Pool(args.n_procs) as pool:
        for i, (X_chunk, y_chunk, meta_chunk, n_gen, elapsed) in enumerate(
                pool.imap_unordered(worker_blue_batch_stageB, tasks)):
            if n_gen == 0:
                print(f"  Process #{i}: 0 samples (skipped)")
                continue
            room = args.n_samples - write_pos
            n_write = min(n_gen, room)
            end_pos = write_pos + n_write
            X_mmap[write_pos:end_pos] = X_chunk[:n_write]
            y_mmap[write_pos:end_pos] = y_chunk[:n_write]
            nr_mmap[write_pos:end_pos] = meta_chunk['n_red'][:n_write]
            nb_mmap[write_pos:end_pos] = meta_chunk['n_blue'][:n_write]
            del X_chunk, y_chunk, meta_chunk
            write_pos = end_pos
            total_gen = write_pos

            tput = n_gen / elapsed if elapsed > 0 else 0
            print(f"  Process #{i}: {n_gen:,} samples, {elapsed:.0f}s, "
                  f"{tput:.1f} sol/s, total: {total_gen:,}/{args.n_samples:,} "
                  f"({min(100,total_gen/args.n_samples*100):.1f}%)", flush=True)

            ckpt_mark = (total_gen // args.checkpoint_every) * args.checkpoint_every
            if ckpt_mark > last_ckpt_mark:
                X_mmap.flush(); y_mmap.flush()
                nr_mmap.flush(); nb_mmap.flush()
                ckpt_path = args.output.replace('.npz', f'_ckpt_{ckpt_mark}.npz')
                np.savez_compressed(ckpt_path,
                                    X=X_mmap[:ckpt_mark],
                                    y=y_mmap[:ckpt_mark],
                                    n_red=nr_mmap[:ckpt_mark],
                                    n_blue=nb_mmap[:ckpt_mark])
                print(f"  [CHECKPOINT] {ckpt_mark:,} -> {ckpt_path}", flush=True)
                last_ckpt_mark = ckpt_mark
            if write_pos >= args.n_samples:
                break

    pool.terminate(); pool.join()
    wall_time = time.time() - t_start

    final_n = write_pos
    if final_n == 0:
        print("ERROR: no samples were generated!")
        sys.exit(1)

    X_mmap.flush(); y_mmap.flush(); nr_mmap.flush(); nb_mmap.flush()
    try:
        np.savez_compressed(args.output,
                            X=X_mmap[:final_n], y=y_mmap[:final_n],
                            n_red=nr_mmap[:final_n], n_blue=nb_mmap[:final_n])
    except MemoryError:
        np.savez(args.output,
                 X=X_mmap[:final_n], y=y_mmap[:final_n],
                 n_red=nr_mmap[:final_n], n_blue=nb_mmap[:final_n])

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    y_view = y_mmap[:final_n]
    nr_view = nr_mmap[:final_n]; nb_view = nb_mmap[:final_n]

    print(f"\n{'='*60}")
    print("Stage B data generation complete!")
    print(f"{'='*60}")
    print(f"Total samples: {final_n:,}")
    print(f"Total time: {wall_time/3600:.2f}h ({wall_time:.0f}s)")
    print(f"Throughput: {final_n/wall_time:.1f} sol/s")
    print(f"Output: {args.output} ({file_size_mb:.1f} MB)")
    print(f"y(u2) range: [{np.min(y_view)},{np.max(y_view)}], "
          f"mu={np.mean(y_view.astype(np.float64)):.1f}, "
          f"sigma={np.std(y_view.astype(np.float64)):.1f}")
    print(f"init_red: [{np.min(nr_view)},{np.max(nr_view)}], "
          f"mu={np.mean(nr_view.astype(np.float64)):.1f}")
    print(f"init_blue: [{np.min(nb_view)},{np.max(nb_view)}], "
          f"mu={np.mean(nb_view.astype(np.float64)):.1f}")

    for p in [X_mmap_path, y_mmap_path, nr_mmap_path, nb_mmap_path]:
        try: os.remove(p)
        except Exception: pass


if __name__ == '__main__':
    main()
