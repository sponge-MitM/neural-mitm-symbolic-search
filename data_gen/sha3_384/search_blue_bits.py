'Module object.'
import sys
import os
import argparse
import hashlib
import time
import random


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Keccak import *
import gurobipy as gp
from gurobipy import GRB


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="random seed (Gurobi Seed)")
parser.add_argument("--time-limit", type=int, default=3000, help="solve time limit (seconds)")
parser.add_argument("--least-number", type=int, default=22, help="minimum number of bits in the blue neutral set")
parser.add_argument("--max-solutions", type=int, default=100000, help="maximum number of solutions to search")
parser.add_argument("--use-rules", action="store_true", help="enable explicit rules extracted by the neural network")
parser.add_argument("--rule-alpha", type=float, default=0.3, help="rule 1: blue_x012 penalty coefficient")
parser.add_argument("--rule-beta",  type=float, default=0.1, help="rule 2: pi1_total penalty coefficient")
parser.add_argument("--rule-gamma", type=float, default=0.15, help="rule 3: blue_x34 reward coefficient")
args, _ = parser.parse_known_args()

least_number = args.least_number
max_solutions = args.max_solutions
seed = args.seed
time_limit = args.time_limit

print(f"=== [SHA3-384] Searching for Blue Bit Count == {least_number} ===")
print(f"  Seed: {seed}, time limit: {time_limit}s, target solutions: {max_solutions}")


model = gp.Model("Keccak384_MILP_BlueSearch")
model.setParam("MIPGap", 0.65)
model.setParam("TimeLimit", time_limit)
model.setParam("Seed", seed)
model.setParam("MIPFocus", 1)
# model.setParam("Threads", 4)


initial_state = [[[Bit(model, "constant", "c") for x in range(5)] for y in range(5)] for z in range(64)]
blue_bits = []
blue_vars = {}




for z in range(64):
    for x in range(5):

        initial_state[z][0][x] = Bit(model, f"initial_state[{z}][0][{x}]", (0, 0, "*"))
        initial_state[z][1][x].b = initial_state[z][0][x].b
        blue_bits.append(initial_state[z][0][x].b)
        blue_vars[(z, x)] = initial_state[z][0][x].b


        if x <= 2:
            if x == 2 and z >= 60:
                continue
            initial_state[z][2][x].b = initial_state[z][0][x].b
            blue_bits.append(initial_state[z][2][x].b)

print(f"Number of blue bit variables: {len(blue_bits)}")
print(f"blue_vars dict size: {len(blue_vars)}")


theta_state_1, C_1, D_1, theta_vars1 = create_first_theta_operation(model, initial_state, "theta_1")


rho_state_1 = rho(theta_state_1)
pi_state_1 = pi(rho_state_1)


for z in range(64):
    for y in range(5):
        model.addConstr(pi_state_1[z][y][0].b + pi_state_1[z][y][1].b <= 1)


theta_state_2, C_2, D_2, theta_vars2 = create_second_theta_operation(model, pi_state_1, "theta_2")
rho_state_2 = rho(theta_state_2)
pi_state_2 = pi(rho_state_2)


model.addConstr(gp.quicksum(blue_bits) >= least_number, name="blue_sum_constraint")


diffusion_bit = []
next_chi_adjacent = []

for x in range(5):
    for z in range(64):
        model.addConstr(theta_vars2[f"C_x{x}_z{z}"]["delta_r"] == 0)
        model.addConstr(theta_vars2[f"C_x{x}_z{z}"]["delta_b"] == 0)
        model.addConstr(theta_vars2[f"D_x{x}_z{z}"]["delta_r"] == 0)
        model.addConstr(theta_vars2[f"D_x{x}_z{z}"]["delta_b"] == 0)

for x in range(5):
    for y in range(5):
        for z in range(64):
            model.addConstr(theta_vars2[f"new_z{z}_y{y}_x{x}"]["delta_r"] == 0)
            model.addConstr(theta_vars2[f"new_z{z}_y{y}_x{x}"]["delta_b"] == 0)

for x in range(5):
    for y in range(5):
        for z in range(64):
            diffusion_bit.append(theta_state_2[z][y][x].b)


            a = model.addVar(vtype=GRB.BINARY, name=f"next_chi_adj_z{z}_y{y}_x{x}")
            model.addConstr(a >= pi_state_2[z][y][x].b + pi_state_2[z][y][(x + 1) % 5].b - 1)
            model.addConstr(2 * a <= pi_state_2[z][y][x].b + pi_state_2[z][y][(x + 1) % 5].b)
            next_chi_adjacent.append(a)



    model.setObjective(gp.quicksum(diffusion_bit)-0.01*gp.quicksum(next_chi_adjacent),gp.GRB.MINIMIZE)





RHO_BOX = [[0, 1, 62, 28, 27], [36, 44, 6, 55, 20], [3, 10, 43, 25, 39], [41, 45, 15, 21, 8], [18, 2, 61, 56, 14]]


ALL_POSITIONS = [(z, x) for z in range(64) for x in range(5)]


Y2_VALID = {(z, x) for z in range(64) for x in range(3) if not (x == 2 and z >= 60)}


def build_state_gf2(blue_set):
    'Function build state gf2.'
    state = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(64)]
    for (z, x) in blue_set:
        state[z][0][x] = 1
        state[z][1][x] = 1
        if (z, x) in Y2_VALID:
            state[z][2][x] = 1
    return state


def rho_gf2(state):
    'Function rho gf2.'
    new_state = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(64)]
    for z in range(64):
        for y in range(5):
            for x in range(5):
                new_state[z][y][x] = state[(z - RHO_BOX[y][x]) % 64][y][x]
    return new_state


def pi_gf2(state):
    'Function pi gf2.'
    new_state = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(64)]
    for z in range(64):
        for y in range(5):
            for x in range(5):
                new_state[z][y][x] = state[z][x][(x + 3 * y) % 5]
    return new_state


def check_chi_adjacency(pi_state):
    'Function check chi adjacency.'
    for z in range(64):
        for y in range(5):
            if pi_state[z][y][0] and pi_state[z][y][1]:
                return False
    return True


def gf2_feasible(blue_set):
    'Function gf2 feasible.'
    state = build_state_gf2(blue_set)
    state = rho_gf2(state)
    state = pi_gf2(state)
    return check_chi_adjacency(state)



print(f"\n=== Phase 1: Gurobi Solution Pool Batch Search (run up to {time_limit}s) ===")
phase_start = time.time()

model.setParam("PoolSearchMode", 2)
model.setParam("PoolSolutions", max_solutions)
model.optimize()

pool_count = model.SolCount
phase1_elapsed = time.time() - phase_start
print(f"Gurobi solve finished: {pool_count} solutions (took {phase1_elapsed:.0f}s)")


all_solutions_map = {}  # frozenset(blue_positions) -> state_matrix
for sol_idx in range(pool_count):
    model.setParam("SolutionNumber", sol_idx)
    state_matrix = [[0 for x in range(5)] for z in range(64)]
    blue_set = set()
    for z in range(64):
        for x in range(5):
            var = blue_vars.get((z, x))
            if var is not None:
                try:
                    val = int(round(var.Xn))
                except Exception:
                    try:
                        val = int(round(var.X))
                    except Exception:
                        val = 0
                state_matrix[z][x] = val
                if val == 1:
                    blue_set.add((z, x))
    key = frozenset(blue_set)
    if key not in all_solutions_map:
        all_solutions_map[key] = state_matrix

unique_seeds = len(all_solutions_map)
all_solutions = list(all_solutions_map.values())
print(f"  Unique seeds after dedup: {unique_seeds}")

if not all_solutions:
    print("x No seed solutions found, cannot continue")
    sys.exit(1)


if len(all_solutions) < max_solutions:
    print(f"\n=== Phase 2: GF(2) Perturbation Generation (to reach {max_solutions} solutions) ===")
    phase2_start = time.time()
    needed = max_solutions - len(all_solutions)
    print(f"  Need {needed} additional solutions")

    K_SWAP_MIN = 1
    K_SWAP_MAX = 4
    PROGRESS_INTERVAL = max(10000, needed // 20)
    attempts = 0
    successes = 0

    while len(all_solutions) < max_solutions:
        elapsed = time.time() - phase_start
        if elapsed >= time_limit:
            print(f"\n  Time limit reached ({time_limit}s), stopping perturbation")
            break


        key = random.choice(list(all_solutions_map.keys()))
        blue_set = set(key)


        k = random.randint(K_SWAP_MIN, K_SWAP_MAX)
        blue_list = list(blue_set)
        non_blue = [p for p in ALL_POSITIONS if p not in blue_set]

        if not blue_list or not non_blue:
            continue

        to_remove = random.sample(blue_list, min(k, len(blue_list)))
        to_add = random.sample(non_blue, min(k, len(non_blue)))

        new_blue = (blue_set - set(to_remove)) | set(to_add)
        new_key = frozenset(new_blue)
        attempts += 1

        if new_key not in all_solutions_map:
            if gf2_feasible(new_blue):
                state_matrix = [[0 for x in range(5)] for z in range(64)]
                for (z, x) in new_blue:
                    state_matrix[z][x] = 1
                all_solutions_map[new_key] = state_matrix
                all_solutions.append(state_matrix)
                successes += 1

                if len(all_solutions) % PROGRESS_INTERVAL == 0:
                    now = time.time()
                    rate = successes / max(0.001, now - phase2_start)
                    print(f"  Progress: {len(all_solutions)}/{max_solutions} "
                          f"(success rate {100*successes/max(1,attempts):.0f}%, "
                          f"rate {rate:.0f} sol/s, elapsed {now - phase_start:.0f}s)")

    phase2_elapsed = time.time() - phase2_start
    print(f"  GF(2) perturbation: succeeded {successes}/{attempts} (took {phase2_elapsed:.0f}s)")

total_elapsed = time.time() - phase_start


print(f"\n{'='*50}")
if len(all_solutions) >= max_solutions:
    print(f"OK Found {len(all_solutions)} solutions! (total time {total_elapsed:.0f}s)")
elif len(all_solutions) > 0:
    print(f" Target {max_solutions} solutions, actually found {len(all_solutions)} (total time {total_elapsed:.0f}s)")
else:
    print(f"x No solutions found (total time {total_elapsed:.0f}s)")



if len(all_solutions) > 0:
    fname = "SHA3_384_all_blue_rules.py" if args.use_rules else "SHA3_384_all_blue.py"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'sha3_384')
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, fname)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"all_solutions = {all_solutions}\n")
    print(f"OK Saved {len(all_solutions)} solutions to: {output_file}")
    print(f"  Per-solution dimensions: 64x5 = {len(all_solutions[0])}x{len(all_solutions[0][0])} bits")
else:
    print("x No valid solutions found!")
    print("  Consider checking constraint settings or trying different solver parameters")

if len(all_solutions) < max_solutions:
    sys.exit(1)
