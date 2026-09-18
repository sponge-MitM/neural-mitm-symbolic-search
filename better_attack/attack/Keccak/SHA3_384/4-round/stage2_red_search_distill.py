"""
SHA3-384 4-round meet-in-the-middle attack - Stage 2 red search
(distilled-rule-embedded version + two-phase strategy).

Model: the full 64-slice Keccak-f[1600] state (base_MILP.Keccak_MILP_64), matching
the 64-slice Stage-1 blue library written by
attack/Keccak/SHA3_384/4-round/stage1_blue_search_distill.py and the 1600-dimensional
pi1 masks used for distillation.

Rule source: distill/rules/rules_sha3_384_stage2.json (Ridge fit to the MLP_stageB
soft labels)
  Distillation fidelity Spearman(rules, soft labels) = 0.8970
  Proxy quality   Spearman(rules, raw labels)        = 0.5157
  QP (MLP_stageB) Spearman vs raw labels             = 0.6369
The embedded table is the JSON table renormalized so that sum |w| = 1.

Strategy:
    Phase 1: search each blue scheme for 2000s, recording temp_degree
    Phase 2: for each least_number pick the blue scheme with the largest
             temp_degree and deep-search for 30000s

SHA3-384 vs SHA3-512 rule differences:
    - Red minefield: y=2 (384) vs y=3 (512) - different rate/capacity boundaries
    - Blue has "good" lanes: y=2,4 neutral (384) vs all-harmful (512)
    - x=3 significant: 9/10 in Top-10 (384) vs 3/10 (512)
"""
import os

from gurobipy import GRB
from base_MILP.Keccak_MILP_64 import *
from output.write_in_file_slice_64 import *
from attack.Keccak.SHA3_384.blue_result.blue_scheme_distill import all_solutions

# ============================================================
# Rule weight definitions (source: distill/rules_sha3_384_stage2.json, retrained NN)
# ============================================================
RULES_DISTILL = [
    ("pi1_red_total", 1, 0.258640, 236.0),
    ("pi1_blue_total", 1, 0.229405, 241.0),
    ("pi1_red_y0", 1, 0.060463, 54.0),
    ("pi1_red_y3", 1, 0.059896, 48.0),
    ("pi1_blue_y0", 1, 0.051894, 55.0),
    ("pi1_red_y2", 1, 0.051612, 47.0),
    ("pi1_blue_y3", 1, 0.049042, 52.0),
    ("pi1_red_y1", 1, 0.047191, 49.0),
    ("pi1_blue_y1", 1, 0.044647, 46.0),
    ("pi1_blue_y4", 1, 0.042138, 44.0),
    ("pi1_blue_y2", 1, 0.041685, 49.0),
    ("pi1_red_y4", 1, 0.039477, 47.0),
    ("init_red_x3", -1, 0.008851, 19.0),
    ("init_red_x1", 1, 0.007533, 12.0),
    ("init_blue_x1", 1, 0.007525, 12.0),
]

# Normalize weights: Sigma|w_i| = 1
_w_sum = sum(abs(r[2]) for r in RULES_DISTILL)
RULES_NORM = [(name, sign, w_raw / _w_sum, max_v)
              for name, sign, w_raw, max_v in RULES_DISTILL]

EPSILON = 0.02  # epsilon in [0.01, 0.02], see paper
NUM_ROUNDS = 3

print("=" * 60)
print("SHA3-384 Stage 2 Red Search - distilled-rule embedded (Route A, based on better_new)")
print(f"Number of rules: {len(RULES_NORM)}, eps={EPSILON}")
print(f"Weight sum: {sum(abs(r[2]) for r in RULES_NORM):.6f}")
for name, sign, w, mv in RULES_NORM:
    direction = "MAXIMIZE" if sign < 0 else "MINIMIZE"
    print(f"  {name:16s} {direction:8s} w={w:+.6f} max={mv}")
print("=" * 60)


def _build_model(blue_scheme_new, least_number, time_limit, model_name):
    """Build and solve the MILP model, returning a results dictionary."""
    model = gp.Model(model_name)
    model.setParam('MIPFocus', 1)
    model.setParam('TimeLimit', time_limit)
    num_rounds = NUM_ROUNDS

    # 1. Initialize state (64-slice, SHA3-384 attack space)
    print("Initializing Keccak state...")
    initial_state = [[[Bit(model, 'constant', 'uc') for x in range(5)] for y in range(5)] for z in range(64)]

    for z in range(64):
        for x in range(5):
            if blue_scheme_new[z][0][x] < 0.5:
                initial_state[z][0][x].b = 0
                initial_state[z][0][x].r = model.addVar(vtype=GRB.BINARY, name=f"initial_{z}_{0}_{x}_r")
                initial_state[z][1][x].b = initial_state[z][0][x].b
                initial_state[z][1][x].r = initial_state[z][0][x].r
            else:
                initial_state[z][0][x].r = 0
                initial_state[z][0][x].b = 1
                initial_state[z][1][x].b = initial_state[z][0][x].b
                initial_state[z][1][x].r = initial_state[z][0][x].r
            if x <= 2:
                if x == 2 and z >= 60:
                    continue
                initial_state[z][2][x].b = initial_state[z][0][x].b
                initial_state[z][2][x].r = initial_state[z][0][x].r

    # 2. Apply round functions
    print("Applying round functions...")
    intermediate_states = []
    current_state = initial_state

    for round_num in range(num_rounds):
        print(f"Applying round {round_num + 1}")
        if round_num == 0:
            theta_state, C, D, theta_vars = create_first_theta_operation(model, current_state, f"round{round_num}_theta")
        elif round_num == 1:
            theta_state, C, D, theta_vars = create_second_theta_operation(model, current_state, f"round{round_num}_theta")
        else:
            theta_state, C, D, theta_vars = create_theta_operation(model, current_state, f"round{round_num}_theta")

        # Forbid blue bit cancellation
        for x in range(5):
            for z in range(64):
                model.addConstr(theta_vars[f"D_x{x}_z{z}"]['delta_b'] == 0)
                for y in range(5):
                    model.addConstr(theta_vars[f"new_z{z}_y{y}_x{x}"]['delta_b'] == 0)

        rho_state = rho(theta_state)
        pi_state = pi(rho_state)

        if round_num == 0:
            chi_state, chi_vars = create_first_chi_operation_384(model, pi_state, f"round{round_num}_chi")
        elif round_num == 1:
            chi_state, chi_vars = create_second_chi_operation(model, pi_state, f"round{round_num}_chi")
        else:
            chi_state, chi_vars = create_chi_operation(model, pi_state, f"round{round_num}_chi")

        intermediate_states.append({
            'theta': theta_state, 'C': C, 'D': D, 'theta_var': theta_vars,
            'rho': rho_state, 'pi': pi_state,
            'chi': chi_state, 'chi_var': chi_vars
        })
        current_state = chi_state

    final_state = current_state
    print(f"Completed {num_rounds} rounds application")

    # ============================================================
    # 3. Build heuristic rule H(X) (pi1 layer, 64-slice)
    # ============================================================
    pi1 = intermediate_states[1]['pi']  # Round-2 pi (pi1)

    # Distilled rule features: pi1 layer red/blue (total / y rows) + init layer red/blue x columns (y=0 layer)
    pi1_red_total = gp.quicksum(pi1[z][y][x].r for z in range(64) for y in range(5) for x in range(5))
    pi1_blue_total = gp.quicksum(pi1[z][y][x].b for z in range(64) for y in range(5) for x in range(5))
    pi1_red_y = [gp.quicksum(pi1[z][y][x].r for z in range(64) for x in range(5)) for y in range(5)]
    pi1_blue_y = [gp.quicksum(pi1[z][y][x].b for z in range(64) for x in range(5)) for y in range(5)]
    init_red_x = [gp.quicksum(initial_state[z][0][x].r for z in range(64)) for x in range(5)]
    init_blue_x = [gp.quicksum(initial_state[z][0][x].b for z in range(64)) for x in range(5)]

    feature_map = {
        "pi1_red_total": pi1_red_total,
        "pi1_blue_total": pi1_blue_total,
    }
    for _y in range(5):
        feature_map[f"pi1_red_y{_y}"] = pi1_red_y[_y]
        feature_map[f"pi1_blue_y{_y}"] = pi1_blue_y[_y]
    for _x in range(5):
        feature_map[f"init_red_x{_x}"] = init_red_x[_x]
        feature_map[f"init_blue_x{_x}"] = init_blue_x[_x]
    heuristic_for_max = 0
    for name, sign, w_norm, max_v in RULES_NORM:
        feat = feature_map[name]
        if sign < 0:            # MAXIMIZE feature (distillation convention)
            heuristic_for_max += w_norm * (feat / max_v)
        else:                   # MINIMIZE feature
            heuristic_for_max -= w_norm * (feat / max_v)

    # ============================================================
    # 4. Equation count
    # ============================================================
    print("Calculating equation count...")
    red_vars_count = gp.quicksum(initial_state[z][y][x].r for z in range(64) for y in range(5) for x in range(5))
    blue_vars_count = gp.quicksum(initial_state[z][y][x].b for z in range(64) for y in range(5) for x in range(5))

    delta_total_r = 0
    delta_total_b = 0
    for round_state in intermediate_states:
        theta_vars = round_state['theta_var']
        for z in range(64):
            for x in range(5):
                delta_total_r += theta_vars[f"C_x{x}_z{z}"]['delta_r']
                delta_total_b += theta_vars[f"C_x{x}_z{z}"]['delta_b']
                delta_total_r += theta_vars[f"D_x{x}_z{z}"]['delta_r']
                delta_total_b += theta_vars[f"D_x{x}_z{z}"]['delta_b']
                for y in range(5):
                    delta_total_r += theta_vars[f"new_z{z}_y{y}_x{x}"]['delta_r']
                    delta_total_b += theta_vars[f"new_z{z}_y{y}_x{x}"]['delta_b']
        chi_vars = round_state['chi_var']
        for z in range(64):
            for y in range(5):
                for x in range(5):
                    delta_total_r += chi_vars[f"new_z{z}_y{y}_x{x}"]['delta_r']
                    delta_total_b += chi_vars[f"new_z{z}_y{y}_x{x}"]['delta_b']
    model.addConstr(red_vars_count<=15)
    # SHA3-384 hash output equations (equation3 only)
    hash_output_bits = []
    cut_bits = []
    no_confusion = []
    for z in range(64):
        for x, y in [(3, 0), (3, 3), (0, 2), (0, 0)]:
            confusion = model.addVar(vtype=GRB.BINARY)  # Only upper-bound constraints + positive coefficients; equivalent under continuous relaxation
            model.addConstr(confusion <= 1 - final_state[z][y][x].ul + final_state[z][y][x].r + final_state[z][y][x].b)
            no_confusion.append(confusion)
        equation1 = model.addVar(vtype=GRB.BINARY, name=f'equation1_{z}')
        model.addConstr(equation1 <= 1 - final_state[z][0][3].ul + final_state[z][0][3].r + final_state[z][0][3].b)
        model.addConstr(equation1 <= 1 - final_state[z][3][3].ul + final_state[z][3][3].r + final_state[z][3][3].b)
        model.addConstr(equation1 <= 1 - final_state[(z - 39) % 64][2][0].ul + final_state[(z - 39) % 64][2][0].r + final_state[(z - 39) % 64][2][0].b)
        model.addConstr(equation1 <= 1 - final_state[(z - 39) % 64][0][0].ul + final_state[(z - 39) % 64][0][0].r + final_state[(z - 39) % 64][0][0].b)
        hash_output_bits.append(equation1)

    total_equations = gp.quicksum(hash_output_bits)
    print("Equation calculation completed")

    # ============================================================
    # 5. Objective: temp_degree + rule term (MAXIMIZE)
    # ============================================================
    temp_degree = model.addVar(vtype=GRB.INTEGER, lb=0, name='complexity')
    model.addConstr(temp_degree <= red_vars_count - delta_total_r - gp.quicksum(cut_bits))
    model.addConstr(temp_degree <= blue_vars_count - delta_total_b - gp.quicksum(cut_bits))
    model.addConstr(temp_degree <= total_equations)

    model.setObjective(temp_degree + EPSILON * heuristic_for_max, GRB.MAXIMIZE)
    print("Constraints and objective function set (with pi1 rules)")

    # 6. Solve the model
    print("Starting model solution...")
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("WARNING: Model is INFEASIBLE!")

    return {
        'model': model,
        'initial_state': initial_state,
        'intermediate_states': intermediate_states,
        'final_state': final_state,
        'temp_degree': temp_degree,
        'red_vars_count': red_vars_count,
        'blue_vars_count': blue_vars_count,
        'delta_total_r': delta_total_r,
        'delta_total_b': delta_total_b,
        'total_equations': total_equations,
    }


# ============================================================
# Phase 1: Quick evaluation (2000s per scheme)
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: Quick evaluation (2000s per blue scheme)")
print("=" * 60)

phase1_results = []  # (key, blue_number, blue_scheme_new, temp_degree, obj_val)

for key in all_solutions.keys():
    for blue_number, blue_scheme_new in enumerate(all_solutions[key]):
        print(f"\nPhase 1: least_number={key}, blue#{blue_number}")

        res = _build_model(blue_scheme_new, key, 5000, f"Keccak_P1_k{key}_b{blue_number}")
        model = res['model']

        has_sol = model.status != GRB.INFEASIBLE and model.SolCount > 0
        td = res['temp_degree'].x if has_sol else -1
        obj_val = model.ObjVal if has_sol else -float('inf')
        phase1_results.append((key, blue_number, blue_scheme_new, td, obj_val))
        print(f"  -> obj_val={obj_val:.4f}, temp_degree={td}")

        model.dispose()

# ============================================================
# Select the blue scheme with the largest obj_val for each least_number
# ============================================================
print("\n" + "=" * 60)
print("Selecting best blue scheme per least_number...")
print("=" * 60)

best_per_key = {}
for key, bn, scheme, td, obj_val in phase1_results:
    if key not in best_per_key or obj_val > best_per_key[key][3]:
        best_per_key[key] = (bn, scheme, td, obj_val)

for key, (best_bn, _, best_td, best_obj) in sorted(best_per_key.items()):
    print(f"  least_number={key}: best is blue#{best_bn}, phase1 obj_val={best_obj:.4f}, temp_degree={best_td}")

# ============================================================
# Phase 2: Deep search (30000s per best scheme)
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Deep search for best per least_number (30000s)")
print("=" * 60)

for key, (best_bn, best_scheme, phase1_td, phase1_obj) in sorted(best_per_key.items()):
    print(f"\nleast_number={key}: best is blue#{best_bn} (phase1 obj_val={phase1_obj:.4f}, temp_degree={phase1_td})")

    res = _build_model(best_scheme, key, 30000, f"Keccak_P2_k{key}_b{best_bn}")
    model = res['model']
    if model.SolCount == 0:
        print(f"  WARNING: no feasible solution for least_number={key}, blue#{best_bn}; skipping output")
        model.dispose()
        continue
    initial_state = res['initial_state']
    intermediate_states = res['intermediate_states']
    temp_degree = res['temp_degree']
    red_vars_count = res['red_vars_count']
    blue_vars_count = res['blue_vars_count']
    delta_total_r = res['delta_total_r']
    delta_total_b = res['delta_total_b']
    total_equations = res['total_equations']
    num_rounds = NUM_ROUNDS

    # Output detailed results (including the temp_degree value)
    os.makedirs("../final_result", exist_ok=True)
    f = open(f"../final_result/SHA3_384_round_{num_rounds + 1}_preimage{key}_{best_bn}_distill.py", 'w')
    f.write(f"Red_variables={red_vars_count.getValue() - delta_total_r.getValue()}\n")
    f.write(f"Blue_variables={blue_vars_count.getValue() - delta_total_b.getValue()}\n")
    f.write(f"Total_equations={total_equations.getValue()}\n")
    f.write(f"temp_degree={temp_degree.x}\n")
    f.write(f"phase1_obj_val={phase1_obj}\n")
    f.write(f"phase1_temp_degree={phase1_td}\n")
    f.write(f"# Rules: pi1 Ridge red/blue separation (2026-07-28), eps={EPSILON}\n")

    print(f"Keccak MILP automation modeling completed (temp_degree={temp_degree.x})")

    row_num = 0
    if_blue = write_row(initial_state, row_num, '$A$')
    f.write(f"initial_state_output = {if_blue}\n")
    intermediate_states_output = []

    index = 1
    for round_state in intermediate_states:
        round_state_output = dict()
        theta_vars = round_state['theta_var']
        chi_vars = round_state['chi_var']
        row_num += 1
        C = round_state['C']
        if_blue = write_row_C(C, row_num, theta_vars, f'$C_{index}$')
        round_state_output['C'] = if_blue
        row_num += 0.4
        D = round_state['D']
        if_blue = write_row_D(D, row_num, theta_vars, f'$D_{index}$')
        round_state_output['D'] = if_blue
        row_num += 0.4
        theta = round_state['theta']
        if_blue = write_row_theta(theta, row_num, theta_vars, f'$\\theta_{index}$')
        round_state_output['theta'] = if_blue
        row_num += 1
        rho_state = round_state['rho']
        if_blue = write_row(rho_state, row_num, f'$\\rho_{index}$')
        round_state_output['rho'] = if_blue
        row_num += 1
        pi_state = round_state['pi']
        if_blue = write_row(pi_state, row_num, f'$\\pi_{index}$')
        round_state_output['pi'] = if_blue
        row_num += 1
        chi = round_state['chi']
        if_blue = write_row_chi(chi, row_num, chi_vars, f'$\\chi_{index}$')
        round_state_output['chi'] = if_blue
        index += 1
        intermediate_states_output.append(round_state_output)

    f.write(f"intermediate_states_output={intermediate_states_output}")
    f.close()
    model.dispose()

print("\n" + "=" * 60)
print("All done!")
print("=" * 60)
