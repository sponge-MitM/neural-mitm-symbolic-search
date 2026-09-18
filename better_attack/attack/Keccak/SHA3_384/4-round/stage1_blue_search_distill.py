"""
SHA3-384 4-round meet-in-the-middle attack - Stage 1 blue search (pi1_b rule-embedded version + multi-solution search).

Based on stage1_blue_search.py, with the pi1 heuristic rule embedded in the objective function.

Strategy:
    Search 5 distinct blue distributions for each least_number (via exclusion constraints).

Rule source:
    fast_pi1_calibrate.py deterministic pi1 Ridge (2026-07-28)

SHA3-384 blue rule characteristics:
    - pi1_b is not globally harmful (unlike SHA3-512)
    - y=0 and y=3 are harmful for blue (MINIMIZE)
    - y=2 and y=4 are neutral for blue (not included in the rule)
"""
from base_MILP.Keccak_MILP_64 import *
from output.write_in_file_slice_64 import *

# ============================================================
# Rule weight definitions (source: distill/rules_sha3_384_stage1.json, retrained NN)
# ============================================================
RULES_DISTILL = [
    ("init_blue_x1", 1, 0.175344, 11.0),
    ("init_blue_total", 1, 0.139641, 19.0),
    ("init_blue_x4", -1, 0.118225, 9.0),
    ("init_blue_x0", 1, 0.109368, 9.0),
    ("pi1_blue_y1", 1, 0.094489, 178.0),
    ("init_blue_x3", -1, 0.082831, 9.0),
    ("pi1_blue_y0", -1, 0.049328, 177.0),
    ("blue_z_first_half", 1, 0.047665, 16.0),
    ("blue_z_second_half", 1, 0.045608, 17.0),
    ("pi1_blue_y4", 1, 0.039591, 177.0),
    ("pi1_blue_y3", 1, 0.036256, 179.0),
    ("init_blue_x2", 1, 0.034810, 8.0),
    ("pi1_blue_total", 1, 0.026843, 850.0),
]

_w_sum = sum(abs(r[2]) for r in RULES_DISTILL)
RULES_DISTILL = [(name, sign, w_raw / _w_sum, max_v)
              for name, sign, w_raw, max_v in RULES_DISTILL]

EPSILON = 0.02

print("=" * 60)
print("SHA3-384 Stage 1 Blue Search - pi1_b rule-embedded (5 solutions/least_number)")
print(f"Number of rules: {len(RULES_DISTILL)}, eps={EPSILON}")
print(f"Weight sum: {sum(abs(r[2]) for r in RULES_DISTILL):.6f}")
for name, sign, w, mv in RULES_DISTILL:
    direction = "MAXIMIZE" if sign < 0 else "MINIMIZE"
    print(f"  {name:16s} {direction:8s} w={w:+.6f} max={mv}")
print("=" * 60)

# Dictionary to store the initial state lists for each least_number
all_solutions = {}

for least_number in [17, 18]:
    add_constr = []          # List of constraints excluding already-found solutions
    solutions_list = []

    for search_number in range(15):
        print(f"\n=== Search DOF >= {least_number}, run {search_number+1}/5 ===")

        model = gp.Model("Keccak_MILP_384_Blue_Better")
        model.setParam('MIPGap', 0.0)
        model.setParam('MIPFocus', 2)
        model.setParam('TimeLimit', 2000)

        # Initial state: 64-slice (full state; blue search uses the full state space)
        initial_state = [[[Bit(model, 'constant', 'uc') for x in range(5)] for y in range(5)] for z in range(64)]

        blue_bits = []
        blue_vars = {}

        for z in range(64):
            for x in range(5):
                initial_state[z][0][x] = Bit(model, f'initial_state[{z}][0][{x}]', (0, 0, '*'))
                initial_state[z][1][x].b = initial_state[z][0][x].b
                blue_bits.append(initial_state[z][0][x].b)
                blue_vars[(z, x)] = initial_state[z][0][x].b
                if x <= 2:
                    if x == 2 and z >= 60:
                        continue
                    initial_state[z][2][x].b = initial_state[z][0][x].b
                    blue_bits.append(initial_state[z][2][x].b)

        # ---- Exclude previously found solutions ----
        for prev_one, prev_zero in add_constr:
            one_vars = [blue_vars[(z, x)] for z, x in prev_one]
            model.addConstr(gp.quicksum(one_vars) <= len(prev_one) - 1)

        model.addConstr(gp.quicksum(blue_bits) >= least_number)

        # ============================================================
        # Round 1: theta(identity) -> rho -> pi
        # ============================================================
        theta_state_1, C_1, D_1, theta_vars1 = create_first_theta_operation(model, initial_state, 'theta_1')

        # Round 1 has no diffusion; forbid blue bit cancellation
        for x in range(5):
            for z in range(64):
                model.addConstr(theta_vars1[f"D_x{x}_z{z}"]['delta_r'] == 0)
                model.addConstr(theta_vars1[f"D_x{x}_z{z}"]['delta_b'] == 0)
        for x in range(5):
            for y in range(5):
                for z in range(64):
                    model.addConstr(theta_vars1[f"new_z{z}_y{y}_x{x}"]['delta_r'] == 0)
                    model.addConstr(theta_vars1[f"new_z{z}_y{y}_x{x}"]['delta_b'] == 0)

        rho_state_1 = rho(theta_state_1)
        pi_state_1 = pi(rho_state_1)

        # chi adjacency constraint (adjacent x cannot both be blue, SHA3-384 full state)
        for z in range(64):
            for y in range(5):
                for x in range(5):
                    model.addConstr(pi_state_1[z][y][x].b + pi_state_1[z][y][(x + 1) % 5].b <= 1)

        # ============================================================
        # Round 2: theta(second, pass-through, forbid propagation)
        # ============================================================
        theta_state_2, C_2, D_2, theta_vars2 = create_theta_operation(model, pi_state_1, 'theta_2')

        # Cancellation constraint: forbid cancellation in round 2
        for x in range(5):
            for z in range(64):
                model.addConstr(theta_vars2[f"C_x{x}_z{z}"]['delta_r'] == 0)
                model.addConstr(theta_vars2[f"C_x{x}_z{z}"]['delta_b'] == 0)
                model.addConstr(theta_vars2[f"D_x{x}_z{z}"]['delta_r'] == 0)
                model.addConstr(theta_vars2[f"D_x{x}_z{z}"]['delta_b'] == 0)
        for x in range(5):
            for y in range(5):
                for z in range(64):
                    model.addConstr(theta_vars2[f"new_z{z}_y{y}_x{x}"]['delta_r'] == 0)
                    model.addConstr(theta_vars2[f"new_z{z}_y{y}_x{x}"]['delta_b'] == 0)

        rho_state_2 = rho(theta_state_2)
        pi_state_2 = pi(rho_state_2)

        # ============================================================
        # Feature computation
        # ============================================================
        # Diffusion bits (total blue bits in the theta2 layer)
        diffusion_bit = []
        adjacent_bit_list = []

        for x in range(5):
            for y in range(5):
                for z in range(64):
                    diffusion_bit.append(theta_state_2[z][y][x].b)
                    # pi2 adjacent blue pairs (original logic preserved)
                    a = model.addVar(vtype=GRB.BINARY, name=f"adj_z{z}_y{y}_x{x}")
                    model.addConstr(a >= pi_state_2[z][y][x].b + pi_state_2[z][y][(x + 1) % 5].b - 1)
                    model.addConstr(2 * a <= pi_state_2[z][y][x].b + pi_state_2[z][y][(x + 1) % 5].b)
                    adjacent_bit_list.append(a)

        # Distilled rule features: round-2 pi (pi_state_2) blue total/y rows + init y=0 layer blue total/x columns
        pi1_blue_total = gp.quicksum(pi_state_2[z][y][x].b for z in range(64) for y in range(5) for x in range(5))
        pi1_blue_y = [gp.quicksum(pi_state_2[z][y][x].b for z in range(64) for x in range(5)) for y in range(5)]
        init_blue_total = gp.quicksum(initial_state[z][0][x].b for z in range(64) for x in range(5))
        init_blue_x = [gp.quicksum(initial_state[z][0][x].b for z in range(64)) for x in range(5)]

        blue_z_first_half = gp.quicksum(initial_state[z][0][x].b for z in range(32) for x in range(5))
        blue_z_second_half = gp.quicksum(initial_state[z][0][x].b for z in range(32, 64) for x in range(5))

        feature_map = {"pi1_blue_total": pi1_blue_total, "init_blue_total": init_blue_total,
                       "blue_z_first_half": blue_z_first_half, "blue_z_second_half": blue_z_second_half}
        for _y in range(5):
            feature_map[f"pi1_blue_y{_y}"] = pi1_blue_y[_y]
        for _x in range(5):
            feature_map[f"init_blue_x{_x}"] = init_blue_x[_x]

        heuristic_for_min = 0
        for name, sign, w_norm, max_val in RULES_DISTILL:
            feat = feature_map[name]
            if sign > 0:        # MINIMIZE feature (distillation convention)
                heuristic_for_min += w_norm * (feat / max_val)
            else:               # MAXIMIZE feature
                heuristic_for_min -= w_norm * (feat / max_val)
        # ============================================================
        # Objective: minimize diffusion + maximize adjacent blue pairs + pi1 rule term
        # ============================================================
        model.setObjective(
            gp.quicksum(diffusion_bit)                       # Minimize diffusion
            - 0.01 * gp.quicksum(adjacent_bit_list)           # Maximize pi2 adjacent blue pairs
            + EPSILON * heuristic_for_min,                    # pi1 rule term
            GRB.MINIMIZE)

        model.optimize()

        # ---- Extract current solution ----
        state_matrix = [[[0 for x in range(5)] for y in range(5)] for z in range(64)]
        for z in range(64):
            for x in range(5):
                bv = initial_state[z][0][x].b
                b_value = int(bv.X) if isinstance(bv, gp.Var) else int(bv)
                state_matrix[z][0][x] = b_value
                state_matrix[z][1][x] = b_value
                if x <= 2 and not (x == 2 and z >= 60):
                    state_matrix[z][2][x] = b_value

        # ---- Build exclusion constraints for the next search ----
        temp_one = []
        temp_zero = []
        for z in range(64):
            for x in range(5):
                bv = initial_state[z][0][x].b
                b_value = int(bv.X) if isinstance(bv, gp.Var) else int(bv)
                if b_value > 0.5:
                    temp_one.append((z, x))
                else:
                    temp_zero.append((z, x))

        add_constr.append((temp_one, temp_zero))
        solutions_list.append(state_matrix)
        print(f"  Run {search_number+1}: blue bits (y=0) = {len(temp_one)}")

    all_solutions[least_number] = solutions_list
    print(f"\nleast_number={least_number}: found {len(solutions_list)} solutions")

f = open("../blue_result/blue_scheme_distill.py", 'w')
f.write(f"all_solutions = {all_solutions}\n")
f.write(f"# Rules: pi1_b Ridge (2026-07-28), epsilon={EPSILON}\n")
f.close()
print(f"\nResults saved to ../blue_result/blue_scheme_distill.py")
