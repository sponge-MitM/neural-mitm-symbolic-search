"""
SHA3-512 3-round meet-in-the-middle attack - Stage 1 blue search (neural-network rule-embedded version + multi-solution search).

Based on stage1_blue_search.py, with the Ridge-extracted heuristic rules embedded in the objective function.

Strategy:
    Search 5 distinct blue distributions for each least_number (via exclusion constraints).

How the heuristic conditions are embedded (per the paper's formula):
    max  F(X) + epsilon . H(X)
    H(X) = Sigma w_i . h_i(X)

Each rule h_i is normalized to [0,1], and the weights w_i satisfy Sigma|w_i| = 1.
epsilon is chosen so that |epsilon . H(X)| < 0.5 (the primary objective's minimum span is 1), so the secondary objective does not flip the primary objective's ordering.

Rule source:
    distill/rules_sha3_512_stage1.json - Ridge fit to the NN1 (blue-scheme
    predictor) soft labels over the 31-dim blue features of the init/pi1 layers.
    The active_z rules (blue_z_active, pi1_blue_active_z) are excluded from the
    embedded table (OR-variable linearization), the remaining 13 rules are
    renormalized to sum |w| = 1 (see the README embedding notes).
    Rules regenerated from the rerun dataset (2026-08-26, corrected chi
    propagation); NN1 test Spearman = 0.8015.
"""

import os

from base_MILP.Keccak_MILP_64 import *
from output.write_in_file_slice_64 import *

# ============================================================
# Rule weight definitions (source: distill/rules_sha3_512_stage1.json, NN1 BlueMLP soft labels)
# active_z features (blue_z_active, pi1_blue_active_z) are excluded: they require OR
# linearization in the MILP embedding. Weights renormalized to sum |w| = 1.
# ============================================================
RULES_DISTILL = [
    ("init_blue_total", 1, 0.219956, 33.0),
    ("init_blue_x1", 1, 0.164394, 11.0),
    ("blue_z_second_half", 1, 0.151784, 20.0),
    ("blue_z_first_half", 1, 0.138943, 22.0),
    ("init_blue_x2", 1, 0.108725, 11.0),
    ("pi1_blue_y4", 1, 0.044614, 97.0),
    ("init_blue_x0", 1, 0.044567, 12.0),
    ("pi1_blue_y0", 1, 0.040421, 100.0),
    ("pi1_blue_total", 1, 0.024414, 496.0),
    ("pi1_blue_y1", 1, 0.023838, 103.0),
    ("init_blue_x3", 1, 0.021665, 9.0),
    ("pi1_blue_y2", 1, 0.012240, 100.0),
    ("pi1_blue_y3", -1, 0.004439, 100.0),
]

# Normalize weights: Sigma|w_i| = 1
_w_sum = sum(abs(r[2]) for r in RULES_DISTILL)
RULES_DISTILL = [(name, sign, w_raw / _w_sum, max_v)
              for name, sign, w_raw, max_v in RULES_DISTILL]

EPSILON = 0.02  # epsilon in [0.01, 0.02], see paper

print("=" * 60)
print("SHA3-512 Stage 1 Blue Search - rule-embedded (5 solutions/least_number)")
print(f"Number of rules: {len(RULES_DISTILL)}, eps={EPSILON}")
print(f"Weight sum: {sum(abs(r[2]) for r in RULES_DISTILL):.6f}")
for name, sign, w, mv in RULES_DISTILL:
    direction = "MAXIMIZE" if sign < 0 else "MINIMIZE"
    print(f"  {name:16s} {direction:8s} w={w:+.6f} max={mv}")
print("=" * 60)

# Dictionary to store the initial state lists for each least_number
all_solutions = {}

for least_number in [12, 13]:
    add_constr = []          # List of constraints excluding already-found solutions
    solutions_list = []

    for search_number in range(5):
        print(f"\n=== Search DOF >= {least_number}, run {search_number+1}/5 ===")

        model = gp.Model("Keccak_MILP_Blue_Better")
        model.setParam('MIPGap', 0.0)
        model.setParam('MIPFocus', 2)
        model.setParam('TimeLimit', 2000)

        # Initial state
        initial_state = [[[Bit(model, 'constant', 'uc') for x in range(5)] for y in range(5)] for z in range(64)]

        blue_bits = []
        blue_vars = {}

        for z in range(64):
            for x in range(4):
                if x == 3 and z >= 60:
                    continue
                initial_state[z][0][x] = Bit(model, f'initial_state[{z}][0][{x}]', (0, 0, '*'))
                initial_state[z][1][x] = Bit(model, f'initial_state[{z}][1][{x}]', (0, 0, 0))
                initial_state[z][1][x].b = initial_state[z][0][x].b
                blue_bits.append(initial_state[z][0][x].b)
                blue_vars[(z, x)] = initial_state[z][0][x].b

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

        # Constraint: in each pi1 row, x=0 and x=1 cannot both be blue
        for z in range(64):
            for y in range(5):
                model.addConstr(pi_state_1[z][y][0].b + pi_state_1[z][y][1].b <= 1)

        # ============================================================
        # Round 2: theta(second, linear) -> rho -> pi
        # ============================================================
        theta_state_2, C_2, D_2, theta_vars2 = create_second_theta_operation(model, pi_state_1, 'theta_2')
        rho_state_2 = rho(theta_state_2)
        pi_state_2 = pi(rho_state_2)

        # Cancellation constraint: forbid cancellation in round 2
        for x in range(5):
            for z in range(64):
                model.addConstr(theta_vars2[f"C_x{x}_z{z}"]['delta_r'] == 0)
                model.addConstr(theta_vars2[f"C_x{x}_z{z}"]['delta_b'] == 0)
                model.addConstr(theta_vars2[f"D_x{x}_z{z}"]['delta_r'] == 0)
                model.addConstr(theta_vars2[f"D_x{x}_z{z}"]['delta_b'] == 0)

        # ============================================================
        # Build original objective terms + heuristic rule features (2026-07-28 update: blue all-harmful)
        # ============================================================
        diffusion_bit = []

        for x in range(5):
            for y in range(5):
                for z in range(64):
                    model.addConstr(theta_vars2[f"new_z{z}_y{y}_x{x}"]['delta_r'] == 0)
                    model.addConstr(theta_vars2[f"new_z{z}_y{y}_x{x}"]['delta_b'] == 0)
                    diffusion_bit.append(theta_state_2[z][y][x].b)

        # Distilled rule features: round-2 pi (pi_state_2) blue total/y rows + init y=0 layer blue total/x columns
        pi1_blue_total = gp.quicksum(pi_state_2[z][y][x].b for z in range(64) for y in range(5) for x in range(5))
        pi1_blue_y = [gp.quicksum(pi_state_2[z][y][x].b for z in range(64) for x in range(5)) for y in range(5)]
        init_blue_total = gp.quicksum(initial_state[z][0][x].b for z in range(64) for x in range(4))
        init_blue_x = [gp.quicksum(initial_state[z][0][x].b for z in range(64)) for x in range(4)]
        blue_z_first_half = gp.quicksum(initial_state[z][0][x].b for z in range(64) for x in range(4))
        blue_z_second_half = gp.quicksum(initial_state[z][0][x].b for z in range(32, 64) for x in range(4))

        feature_map = {"pi1_blue_total": pi1_blue_total, "init_blue_total": init_blue_total,
                       "blue_z_first_half": blue_z_first_half, "blue_z_second_half": blue_z_second_half}
        for _y in range(5):
            feature_map[f"pi1_blue_y{_y}"] = pi1_blue_y[_y]
        for _x in range(4):
            feature_map[f"init_blue_x{_x}"] = init_blue_x[_x]

        heuristic_for_min = 0
        for name, sign, w_norm, max_val in RULES_DISTILL:
            feat = feature_map[name]
            if sign > 0:        # MINIMIZE feature (distillation convention)
                heuristic_for_min += w_norm * (feat / max_val)
            else:               # MAXIMIZE feature
                heuristic_for_min -= w_norm * (feat / max_val)
        # Objective function
        model.setObjective(
            gp.quicksum(diffusion_bit)                     # Minimize diffusion
            + EPSILON * heuristic_for_min,                   # rule term
            GRB.MINIMIZE)

        model.optimize()

        # ---- Extract current solution ----
        state_matrix = [[[0 for x in range(5)] for y in range(5)] for z in range(64)]
        for z in range(64):
            for x in range(4):
                if isinstance(initial_state[z][0][x].b, gp.Var):
                    state_matrix[z][0][x] = int(initial_state[z][0][x].b.X)
                else:
                    state_matrix[z][0][x] = int(initial_state[z][0][x].b)
                state_matrix[z][1][x] = state_matrix[z][0][x]

        # ---- Build exclusion constraints for the next search ----
        temp_one = []
        temp_zero = []
        for z in range(64):
            for x in range(4):
                if x == 3 and z >= 60:
                    continue
                bv = initial_state[z][0][x].b
                b_value = int(bv.X) if isinstance(bv, gp.Var) else int(bv)
                if b_value > 0.5:
                    temp_one.append((z, x))
                else:
                    temp_zero.append((z, x))

        add_constr.append((temp_one, temp_zero))
        solutions_list.append(state_matrix)
        print(f"  Run {search_number+1}: blue bits = {len(temp_one)}")

    all_solutions[least_number] = solutions_list
    print(f"\nleast_number={least_number}: found {len(solutions_list)} solutions")

os.makedirs("../blue_result", exist_ok=True)
f = open("../blue_result/SHA3_512_all_blue_distill.py", 'w')
f.write(f"all_solutions = {all_solutions}\n")
f.write(f"# Rules: pi1_b separated (2026-07-28, blue-all-harmful), epsilon={EPSILON}\n")
