"""
# Distilled rules (Route A: Ridge fit to AsconMLP soft labels)
#   Spearman(rules, soft labels)=0.9335  Spearman(rules, raw labels)=0.8942  baseline=0.8940
Ascon-XOF 3-round preimage attack - milp_linear K=15 Ridge rule-embedded version (Phase 1/2)
=====================================================================
Based on the 15 Ridge rules from distill/rules_ascon.json (Spearman=0.9335).

Strategy:
    Phase 1: 5 groups of random weight perturbations (+/-30%), each quick-searched for 3000s, recording obj_val and temp_degree
    Phase 2: pick the best perturbed weights from Phase 1, deep-search for 30000s and output

Constraints:
    blue_vars_count <= red_vars_count (blue <= red), no blue upper bound
"""

import random
import gurobipy as gp
from gurobipy import GRB
from base_MILP.Ascon_MILP_64 import *
from output.write_in_file_slice_64 import *

NUM_PERTURB = 5          # Number of perturbation groups in Phase 1
PHASE1_TIME = 3000       # Phase 1 quick-search time limit (seconds)
PHASE2_TIME = 30000      # Phase 2 deep-search time limit (seconds)
RIDGE_SCALE = 0.001      # epsilon = 0.001, see paper
NUM_ROUNDS = 2           # 3 rounds = num_rounds-1

# ---- milp_linear K=15 base weights (source: distill/rules_ascon.json, retrained NN) ----
# Signs: MAXIMIZE features add positively to the MAXIMIZE objective; MINIMIZE features subtract.
DISTILL_WEIGHTS = {
    "pl0_to_ps1_b_delta": 0.144764,
    "pl0_to_ps1_r_delta": 0.086112,
    "pl0_b_total": 0.081020,
    "ps0_to_pl0_b_delta": 0.076785,
    "ps1_r_total": 0.074902,
    "pl0_b_active_z_count": 0.061603,
    "pl0_b_x_adj_pairs": 0.049072,
    "ps1_r_active_z_count": 0.041461,
    "ps1_b_active_z_count": 0.034795,
    "ps1_b_total": 0.028502,
    "init_rb_diff": 0.026691,
    "pl0_r_x_adj_pairs": 0.023793,
    "ps1_r_z_adj_pairs": 0.022040,
    "pl0_to_ps1_b_overlap": 0.017992,
    "pl0_r_total": 0.016899,
}


def _perturb_weights(seed):
    """Generate the perturbed weight dictionary (+/-30%)."""
    rng = random.Random(seed)
    return {k: v * rng.uniform(0.7, 1.3) for k, v in DISTILL_WEIGHTS.items()}


# ---- MILP helper functions (bound to the model) ----

def _adj_pairs(model, state, prefix, direction, channel='b'):
    """Adjacent-pair indicators matching the distilled feature extractor
    (ascon_propagator.extract_milp_linear_features): x direction uses pairs
    (x, x+1) for x=0..3; z direction uses NON-circular pairs (z, z+1) for
    z=0..slice_number-2 (col[:-1] & col[1:])."""
    terms = []
    z_range = range(slice_number - 1) if direction == 'z' else range(slice_number)
    for z in z_range:
        for x in range(5):
            if direction != 'z' and x == 4:
                continue  # x direction has no 5th adjacent-bit pair; skip early to avoid creating isolated variables
            p = model.addVar(vtype=GRB.BINARY, name=f"{prefix}_{direction}adj_{z}_{x}")
            if direction == 'z':
                a = state[z][x].r if channel == 'r' else state[z][x].b
                b = state[z + 1][x].r if channel == 'r' else state[z + 1][x].b
            else:
                a = state[z][x].r if channel == 'r' else state[z][x].b
                b = state[z][x + 1].r if channel == 'r' else state[z][x + 1].b
            model.addConstr(p <= a)
            model.addConstr(p <= b)
            model.addConstr(p >= a + b - 1)
            terms.append(p)
    return terms


def _active_z(model, state, prefix, channel='b'):
    terms = []
    for z in range(slice_number):
        az = model.addVar(vtype=GRB.BINARY, name=f"{prefix}_az_{z}")
        z_sum = gp.quicksum(state[z][x].r if channel == 'r' else state[z][x].b for x in range(5))
        model.addConstr(az <= z_sum)
        model.addConstr(5 * az >= z_sum)
        terms.append(az)
    return terms


def _z_max(model, state, prefix, channel='b'):
    """Return constant 5 as the z_max heuristic term.

    The original z_max >= z_sum only has a lower-bound constraint; with positive coefficients
    and MAXIMIZE the solver pushes z_max to its upper bound 5, so the "variable" is just the
    constant 5 (2 variables + 64 constraints for nothing). We return the constant directly; the
    constant term (RIDGE_SCALE * weight * 5) stays unchanged, so obj_val comparability across
    Phase 1 seeds is unaffected. For true max_z z_sum semantics, implement the equality-binding encoding before using it.
    """
    return 5


def _overlap(model, sa, sb, prefix, channel='r'):
    terms = []
    for z in range(slice_number):
        for x in range(5):
            ov = model.addVar(vtype=GRB.BINARY, name=f"{prefix}_ov_{z}_{x}")
            a = sa[z][x].r if channel == 'r' else sa[z][x].b
            b = sb[z][x].r if channel == 'r' else sb[z][x].b
            model.addConstr(ov <= a)
            model.addConstr(ov <= b)
            model.addConstr(ov >= a + b - 1)
            terms.append(ov)
    return terms


def _build_model(pert_weights, time_limit, model_name):
    """Build and solve the MILP model, returning a results dictionary."""
    model = gp.Model(model_name)
    model.setParam("MIPFocus", 1)
    model.setParam('TimeLimit', time_limit)

    # 1. Initialize Ascon-XOF preimage attack state
    initial_state = [[Bit(model, f"init_z{z}_x{x}", (0, 0, 0)) for x in range(5)] for z in range(slice_number)]

    for z in range(slice_number):
        x = 0
        if z >= slice_number - 2:
            initial_state[z][x] = Bit(model, f"init_z{z}_x{x}", (0, 0, 0))
        else:
            initial_state[z][x] = Bit(model, f"init_z{z}_x{x}", (0, '*', '*'))
            model.addConstr(initial_state[z][x].r + initial_state[z][x].b <= 1,
                            f"rate_r_b_exclusive_z{z}_x{x}")

    # === blue <= red constraint ===
    red_vars_count = gp.quicksum(initial_state[z][0].r for z in range(slice_number))
    blue_vars_count = gp.quicksum(initial_state[z][0].b for z in range(slice_number))
    model.addConstr(blue_vars_count <= red_vars_count, name="blue_le_red")
    model.addConstr(blue_vars_count <= 9)

    # 2. Apply Ascon round functions
    intermediate_states = []
    current_state = initial_state

    for round_num in range(NUM_ROUNDS):
        if round_num == 0:
            ps_state, ps_vars = create_first_P_S_operation_first_one_padding_three_stage(
                model, current_state, f"round{round_num}_PS")
            temp_state_1 = None
            temp_state_2 = None
        elif round_num == 1:
            temp_state_1, temp_state_2, ps_state, ps_vars = create_second_P_S_operation(
                model, current_state, f"round{round_num}_PS")
        else:
            temp_state_1, temp_state_2, ps_state, ps_vars = create_P_S_operation(
                model, current_state, f"round{round_num}_PS")

        if round_num == 0:
            pl_state, pl_vars = create_first_P_L_operation(model, ps_state, f"round{round_num}_PL")
        else:
            pl_state, pl_vars = create_P_L_operation(model, ps_state, f"round{round_num}_PL")

        intermediate_states.append({
            'temp_state_1': temp_state_1, 'temp_state_2': temp_state_2,
            'ps_state': ps_state, 'ps_vars': ps_vars,
            'pl_state': pl_state, 'pl_vars': pl_vars,
            'round_num': round_num,
        })
        current_state = pl_state

    final_state = current_state

    # 3. Variable statistics
    delta_total_r = 0
    delta_total_b = 0
    capacity_flags = 0

    for round_state in intermediate_states:
        ps_vars = round_state['ps_vars']
        if round_state['round_num'] == 0:
            for z in range(slice_number):
                capacity_flags += ps_vars[f'{z}_vars'][0]
                capacity_flags += ps_vars[f'{z}_vars'][1]
                capacity_flags += ps_vars[f'{z}_vars'][2]
        else:
            for z in range(slice_number):
                for x in range(5):
                    for key in ['temp1', 'temp2', 'new']:
                        k = f"{key}_z{z}_x{x}"
                        if k in ps_vars:
                            delta_total_r += ps_vars[k]['delta_r']
                            delta_total_b += ps_vars[k]['delta_b']
        pl_vars = round_state['pl_vars']
        for z in range(slice_number):
            for x in range(5):
                k = f"new_z{z}_x{x}"
                if k in pl_vars:
                    delta_total_r += pl_vars[k]['delta_r']
                    delta_total_b += pl_vars[k]['delta_b']

    # 4. Hash output bits
    hash_output_bits = []
    cut_bit = []
    sum_of_a0a2a4 = []
    for z in range(slice_number):
        result = Bit(model, f"sum_a0a2a4_z{z}", ('*', '*', '*'))
        v = xor_with_ul_input_no_delta_b(model, [final_state[z][0], final_state[z][2], final_state[z][4]], result)
        sum_of_a0a2a4.append(result)
        delta_total_r += v['delta_r']
    no_confusion = []
    for z in range(slice_number):
        # good_slice: only upper-bound constraints and positive objective coefficient, so after continuous relaxation the optimum still takes the upper bound (equivalent)
        temp = model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name='good_slice')
        for x in range(5):
            model.addConstr(temp <= 1 - final_state[z][x].ul + final_state[z][x].r + final_state[z][x].b)
            if x in (0, 2, 4):
                # The XOR of sum_of_a0a2a4 has already cached a pure-u indicator p for final_state[z][x];
                # no-confusion reward = 1-p (in the optimum, confusion always takes the upper bound 1-ul+b+r = 1-p, equivalent)
                p = get_pure_u_indicator(model, final_state[z][x], f"confusion_pureu_z{z}_x{x}")
                no_confusion.append(1 - p)
            else:
                # x=1,3: no cached indicator, keep the variable but relax to continuous (only upper-bound constraints + positive coefficient)
                confusion = model.addVar(vtype=GRB.INTEGER, lb=0, ub=1)
                model.addConstr(confusion <= 1 - final_state[z][x].ul + final_state[z][x].b + final_state[z][x].r)
                no_confusion.append(confusion)
        model.addConstr(temp <= 2 - final_state[z][1].r - sum_of_a0a2a4[z].b)
        model.addConstr(temp <= 2 - final_state[z][1].b - sum_of_a0a2a4[z].r)
        hash_output_bits.append(temp)


    # 5. Ridge heuristic (distilled rules, Route A)
    ps0_state = intermediate_states[0]['ps_state']
    pl0_state = intermediate_states[0]['pl_state']
    ps1_state = intermediate_states[1]['ps_state']

    ps0_r_total = gp.quicksum(ps0_state[z][x].r for z in range(slice_number) for x in range(5))
    ps0_b_total = gp.quicksum(ps0_state[z][x].b for z in range(slice_number) for x in range(5))
    pl0_r_total = gp.quicksum(pl0_state[z][x].r for z in range(slice_number) for x in range(5))
    pl0_b_total = gp.quicksum(pl0_state[z][x].b for z in range(slice_number) for x in range(5))
    ps1_r_total = gp.quicksum(ps1_state[z][x].r for z in range(slice_number) for x in range(5))
    ps1_b_total = gp.quicksum(ps1_state[z][x].b for z in range(slice_number) for x in range(5))

    ps1_r_x_adj = _adj_pairs(model, ps1_state, 'ps1_r', 'x', 'r')
    ps1_b_x_adj = _adj_pairs(model, ps1_state, 'ps1_b', 'x', 'b')
    ps1_r_z_adj = _adj_pairs(model, ps1_state, 'ps1_r', 'z', 'r')
    pl0_b_x_adj = _adj_pairs(model, pl0_state, 'pl0_b', 'x', 'b')
    pl0_r_x_adj = _adj_pairs(model, pl0_state, 'pl0_r', 'x', 'r')
    ps1_b_az = _active_z(model, ps1_state, 'ps1_b')
    ps1_r_az = _active_z(model, ps1_state, 'ps1_r')
    pl0_b_az = _active_z(model, pl0_state, 'pl0_b')
    pl0_ps1_b_ov = _overlap(model, pl0_state, ps1_state, 'pl0_ps1_b', 'b')

    w = pert_weights
    heuristic_bonus = (
        # MAXIMIZE features (negative distilled coefficient -> add)
        + RIDGE_SCALE * w["pl0_to_ps1_b_delta"]     * (ps1_b_total - pl0_b_total)
        + RIDGE_SCALE * w["pl0_to_ps1_r_delta"]     * (ps1_r_total - pl0_r_total)
        + RIDGE_SCALE * w["pl0_b_total"]            * pl0_b_total
        + RIDGE_SCALE * w["ps1_r_total"]            * ps1_r_total
        + RIDGE_SCALE * w["pl0_b_active_z_count"]   * gp.quicksum(pl0_b_az)
        + RIDGE_SCALE * w["pl0_b_x_adj_pairs"]      * gp.quicksum(pl0_b_x_adj)
        + RIDGE_SCALE * w["ps1_b_total"]            * ps1_b_total
        + RIDGE_SCALE * w["pl0_r_x_adj_pairs"]      * gp.quicksum(pl0_r_x_adj)
        + RIDGE_SCALE * w["ps1_r_z_adj_pairs"]      * gp.quicksum(ps1_r_z_adj)
        + RIDGE_SCALE * w["pl0_r_total"]            * pl0_r_total
        # MINIMIZE features (positive distilled coefficient -> subtract)
        - RIDGE_SCALE * w["ps0_to_pl0_b_delta"]     * (pl0_b_total - ps0_b_total)
        - RIDGE_SCALE * w["ps1_r_active_z_count"]   * gp.quicksum(ps1_r_az)
        - RIDGE_SCALE * w["ps1_b_active_z_count"]   * gp.quicksum(ps1_b_az)
        - RIDGE_SCALE * w["init_rb_diff"]           * (red_vars_count - 2 * blue_vars_count)
        - RIDGE_SCALE * w["pl0_to_ps1_b_overlap"]   * gp.quicksum(pl0_ps1_b_ov)
    )

    # 6. Constraints and objective
    temp_degree = model.addVar(vtype=GRB.INTEGER, name='complexity')
    model.addConstr(temp_degree <= red_vars_count - delta_total_r - gp.quicksum(cut_bit))
    model.addConstr(temp_degree <= blue_vars_count - delta_total_b - gp.quicksum(cut_bit))
    model.addConstr(temp_degree <= gp.quicksum(hash_output_bits))
    degree_from_c = model.addVar(vtype=GRB.INTEGER, name='degree_from_c')
    model.addConstr(capacity_flags + degree_from_c <= 128 * slice_number / 64 - 1 - temp_degree)
    model.addConstr(degree_from_c - 1 >= slice_number)
    model.setObjective(temp_degree + heuristic_bonus + 0.001*gp.quicksum(no_confusion), GRB.MAXIMIZE)

    model.optimize()

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
        'hash_output_bits': hash_output_bits,
        'cut_bit': cut_bit,
        'capacity_flags': capacity_flags,
    }


# ============================================================
# Phase 1: Quick evaluation (3000s x 5 perturbation groups)
# ============================================================
print("=" * 60)
print("Ascon-XOF 3R Better - Phase 1: Quick evaluation")
print(f"  {NUM_PERTURB} perturb groups x {PHASE1_TIME}s each")
print("=" * 60)

phase1_results = []

for seed in range(NUM_PERTURB):
    pert = _perturb_weights(seed)
    name = f"XOF3R_P1_s{seed}"
    print(f"\nPhase 1: seed={seed}")

    res = _build_model(pert, PHASE1_TIME, name)
    m = res['model']
    has_sol = m.status != GRB.INFEASIBLE and m.SolCount > 0
    td = res['temp_degree'].x if has_sol else -1
    obj = m.ObjVal if has_sol else -float('inf')
    phase1_results.append((seed, pert, td, obj))
    print(f"  -> obj_val={obj:.4f}, temp_degree={td}")
    m.dispose()

# Pick the best
best_seed, best_pert, _, best_obj = max(phase1_results, key=lambda x: x[3])
print(f"\nBest seed={best_seed}, phase1 obj_val={best_obj:.4f}")

# ============================================================
# Phase 2: Deep search (30000s)
# ============================================================
print("\n" + "=" * 60)
print(f"Ascon-XOF 3R Better - Phase 2: Deep search ({PHASE2_TIME}s)")
print(f"  Using perturb group seed={best_seed}")
print("=" * 60)

res = _build_model(best_pert, PHASE2_TIME, "XOF3R_P2")
model = res['model']
initial_state = res['initial_state']
intermediate_states = res['intermediate_states']
temp_degree = res['temp_degree']
red_vars_count = res['red_vars_count']
blue_vars_count = res['blue_vars_count']
delta_total_r = res['delta_total_r']
delta_total_b = res['delta_total_b']

# Output
_output_name = f"Ascon_XOF_round_{NUM_ROUNDS + 1}_preimage_better"
output_file = open(f"{_output_name}_new.py", 'w')
output_file.write(f"Red_variables={red_vars_count.getValue() - delta_total_r.getValue()}\n")
# output_file.write(f"Blue_variables={blue_vars_count.getValue() - delta_total_b.getValue()}\n")
output_file.write(f"temp_degree={temp_degree.x}\n")
output_file.write(f"phase1_obj_val={best_obj:.4f}\n")
output_file.write(f"phase1_seed={best_seed}\n")
output_file.write(f"# Rules: milp_linear K=15 Ridge (Spearman=0.9335), RIDGE_SCALE={RIDGE_SCALE}\n")

row_num = 0
initial_state_latex = write_Ascon_initial(initial_state, slice_number, row_num, dict(), '$A$')
output_file.write(f"initial_state_output = {initial_state_latex}\n")

intermediate_states_output = []
state_index = 1

for round_state in intermediate_states:
    round_state_output = dict()
    round_state_output['round_num'] = round_state['round_num']
    ps_vars = round_state['ps_vars']
    pl_vars = round_state['pl_vars']

    if round_state['round_num'] > 0:
        row_num += 0.4
        temp1_latex = write_Ascon_temp_s1(round_state['temp_state_1'], slice_number, row_num, ps_vars, f'$t_1{state_index}$')
        round_state_output['temp_state_1'] = temp1_latex
        row_num += 0.4
        temp2_latex = write_Ascon_temp_s2(round_state['temp_state_2'], slice_number, row_num, ps_vars, f'$t_2{state_index}$')
        round_state_output['temp_state_2'] = temp2_latex
        row_num += 0.4
        ps_latex = write_Ascon_P(round_state['ps_state'], slice_number, row_num, ps_vars, f'$P_S{state_index}$')
        round_state_output['ps_state'] = ps_latex
    else:
        row_num += 0.4
        ps_latex = write_Ascon_initial(round_state['ps_state'], slice_number, row_num, ps_vars, f'$P_S{state_index}$')
        round_state_output['ps_state'] = ps_latex

    row_num += 0.4
    pl_latex = write_Ascon_P(round_state['pl_state'], slice_number, row_num, pl_vars, f'$P_L{state_index}$')
    round_state_output['pl_state'] = pl_latex
    intermediate_states_output.append(round_state_output)
    state_index += 1

output_file.write(f"intermediate_states_output={intermediate_states_output}")
output_file.close()
model.dispose()

print("=" * 60)
