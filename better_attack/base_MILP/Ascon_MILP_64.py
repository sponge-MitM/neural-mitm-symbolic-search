from base_MILP.operation_MILP import *
from gurobipy import GRB

slice_number = 64  # Number of slices


def create_P_L_operation(model, old_state, operation_name="P_L"):
    """
    MILP modeling for P_L function

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - new_state: New state after P_L operation
    - P_L_vars: Variables related to P_L operation
    """

    # Initialize new state
    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_L_vars = {}

    # Define index offsets for each column
    offsets = [
        [19, 28],  # x=0
        [61, 39],  # x=1
        [1, 6],    # x=2
        [10, 17],  # x=3
        [7, 41]    # x=4
    ]

    # Calculate new state
    for z in range(slice_number):
        for x in range(5):
            # Create new state bit
            new_bit = Bit(model, f"{operation_name}_new_z{z}_x{x}", ('*', '*', '*'))

            # Get input bits
            input_bits = [
                old_state[z][x],  # old_state[z][x]
                old_state[(z + offsets[x][0]) % slice_number][x],  # first offset
                old_state[(z + offsets[x][1]) % slice_number][x]   # second offset
            ]

            # Create XOR operation
            xor_vars = xor_with_ul_input_no_delta_b(model, input_bits, new_bit, f"{operation_name}_new_z{z}_x{x}")

            # Store new state and variables
            new_state[z][x] = new_bit
            P_L_vars[f"new_z{z}_x{x}"] = xor_vars


    return new_state, P_L_vars


def create_first_P_L_operation(model, old_state, operation_name="P_L"):
    """
    MILP modeling for first round P_L function (special handling)

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - new_state: New state after P_L operation
    - P_L_vars: Variables related to P_L operation
    """

    # Initialize new state
    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_L_vars = {}

    # Define index offsets for each column
    offsets = [
        [19, 28],  # x=0
        [61, 39],  # x=1
        [1, 6],    # x=2
        [10, 17],  # x=3
        [7, 41]    # x=4
    ]

    # Calculate new state
    for z in range(slice_number):
        for x in range(5):
            # Get input bits
            input_bits = [
                old_state[z][x],  # old_state[z][x]
                old_state[(z + offsets[x][0]) % slice_number][x],  # first offset
                old_state[(z + offsets[x][1]) % slice_number][x]   # second offset
            ]

            # Create new state bit (special bit type)
            # When all inputs are constant zero, fold to a constant-zero Bit early:
            # XOR would also fold the output internally, but by then the r/b
            # variables are already created and orphaned (entering no constraint or
            # objective); here we directly use a constant-zero Bit, saving 128
            # orphaned variables for the whole x=2 column (semantics unchanged).
            if all(is_const_zero_bit(b) for b in input_bits):
                new_bit = Bit(model, f"{operation_name}_new_z{z}_x{x}", (0, 0, 0))
            else:
                new_bit = Bit(model, f"{operation_name}_new_z{z}_x{x}", (0, '*', '*'))

            # Create XOR operation (input state is (0,*,*); use the special ul0 version)
            xor_vars = xor_with_ul0_input_no_delta_b(model, input_bits, new_bit, f"{operation_name}_new_z{z}_x{x}")

            # Store new state and variables
            new_state[z][x] = new_bit
            P_L_vars[f"new_z{z}_x{x}"] = xor_vars


    return new_state, P_L_vars


def create_P_S_operation(model, old_state, operation_name="P_S"):
    """
    MILP modeling for P_S function

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - temp_state_1: Intermediate state after XOR operation in P_S
    - temp_state_2: Intermediate state after chi operation in P_S
    - new_state: New state output by P_S
    - P_S_vars: Variables related to P_S operation
    """

    # Initialize intermediate states and new state
    temp_state_1 = [[None for _ in range(5)] for _ in range(slice_number)]
    temp_state_2 = [[None for _ in range(5)] for _ in range(slice_number)]
    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_S_vars = {}

    # Step 1: Calculate temp_state_1
    for z in range(slice_number):
        # x=0: temp_state_1[z][0] = old_state[z][0] + old_state[z][4]
        bit0 = Bit(model, f"{operation_name}_temp1_z{z}_x0", ('*', '*', '*'))
        xor_vars0 = xor_with_ul_input_no_delta_b(model, [old_state[z][0], old_state[z][4]], bit0, f"{operation_name}_temp1_z{z}_x0")
        temp_state_1[z][0] = bit0
        P_S_vars[f"temp1_z{z}_x0"] = xor_vars0

        # x=1: temp_state_1[z][1] = old_state[z][1] (direct copy)
        temp_state_1[z][1] = old_state[z][1]
        P_S_vars[f"temp1_z{z}_x1"] = {'delta_r': 0, 'delta_b': 0}

        # x=2: temp_state_1[z][2] = old_state[z][1] + old_state[z][2]
        bit2 = Bit(model, f"{operation_name}_temp1_z{z}_x2", ('*', '*', '*'))
        xor_vars2 = xor_with_ul_input_no_delta_b(model, [old_state[z][1], old_state[z][2]], bit2, f"{operation_name}_temp1_z{z}_x2")
        temp_state_1[z][2] = bit2
        P_S_vars[f"temp1_z{z}_x2"] = xor_vars2

        # x=3: temp_state_1[z][3] = old_state[z][3] (direct copy)
        temp_state_1[z][3] = old_state[z][3]
        P_S_vars[f"temp1_z{z}_x3"] = {'delta_r': 0, 'delta_b': 0}

        # x=4: temp_state_1[z][4] = old_state[z][3] + old_state[z][4]
        bit4 = Bit(model, f"{operation_name}_temp1_z{z}_x4", ('*', '*', '*'))
        xor_vars4 = xor_with_ul_input_no_delta_b(model, [old_state[z][3], old_state[z][4]], bit4, f"{operation_name}_temp1_z{z}_x4")
        temp_state_1[z][4] = bit4
        P_S_vars[f"temp1_z{z}_x4"] = xor_vars4

    # Step 2: Calculate temp_state_2 (XAND unified encoding: temp2[x] = temp1[x]  xor  (temp1[x+1] and temp1[x+2]))
    for z in range(slice_number):
        for x in range(5):
            # Create XOR (XAND) operation bit
            xor_bit_name = f"{operation_name}_xor_z{z}_x{x}"
            xor_bit = Bit(model, xor_bit_name, ('*', '*', '*'))

            # XAND: y = x0  xor  (x1 and x2), eliminating the intermediate and_bit
            xand_vars = xand_operation(
                model, temp_state_1[z][x],
                temp_state_1[z][(x + 1) % 5], temp_state_1[z][(x + 2) % 5],
                xor_bit, operation_name=xor_bit_name)
            temp_state_2[z][x] = xor_bit
            P_S_vars[f"temp2_z{z}_x{x}"] = xand_vars
            # Backward-compatible interface: keep the and_ entry (attack scripts only perform an in check)
            P_S_vars[f"and_z{z}_x{x}"] = {'quad': 0}

    # Step 3: Calculate new_state
    for z in range(slice_number):
        # x=0: new_state[z][0] = temp_state_2[z][0] + temp_state_2[z][4]
        new_bit0 = Bit(model, f"{operation_name}_new_z{z}_x0", ('*', '*', '*'))
        xor_vars_new0 = xor_with_ul_input_no_delta_b(model, [temp_state_2[z][0], temp_state_2[z][4]], new_bit0, f"{operation_name}_new_z{z}_x0")
        new_state[z][0] = new_bit0
        P_S_vars[f"new_z{z}_x0"] = xor_vars_new0

        # x=1: new_state[z][1] = temp_state_2[z][1] + temp_state_2[z][0]
        new_bit1 = Bit(model, f"{operation_name}_new_z{z}_x1", ('*', '*', '*'))
        xor_vars_new1 = xor_with_ul_input_no_delta_b(model, [temp_state_2[z][1], temp_state_2[z][0]], new_bit1, f"{operation_name}_new_z{z}_x1")
        new_state[z][1] = new_bit1
        P_S_vars[f"new_z{z}_x1"] = xor_vars_new1

        # x=2: new_state[z][2] = temp_state_2[z][2] + 1 (direct copy)
        new_state[z][2] = temp_state_2[z][2]
        P_S_vars[f"new_z{z}_x2"] = {'delta_r': 0, 'delta_b': 0}

        # x=3: new_state[z][3] = temp_state_2[z][2] + temp_state_2[z][3]
        new_bit3 = Bit(model, f"{operation_name}_new_z{z}_x3", ('*', '*', '*'))
        xor_vars_new3 = xor_with_ul_input_no_delta_b(model, [temp_state_2[z][2], temp_state_2[z][3]], new_bit3, f"{operation_name}_new_z{z}_x3")
        new_state[z][3] = new_bit3
        P_S_vars[f"new_z{z}_x3"] = xor_vars_new3

        # x=4: new_state[z][4] = temp_state_2[z][4] (direct copy)
        new_state[z][4] = temp_state_2[z][4]
        P_S_vars[f"new_z{z}_x4"] = {'delta_r': 0, 'delta_b': 0}


    return temp_state_1, temp_state_2, new_state, P_S_vars


def create_second_P_S_operation(model, old_state, operation_name="P_S"):
    """
    MILP modeling for P_S function

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - temp_state_1: Intermediate state after XOR operation in P_S
    - temp_state_2: Intermediate state after chi operation in P_S
    - new_state: New state output by P_S
    - P_S_vars: Variables related to P_S operation
    """

    # Initialize intermediate states and new state
    temp_state_1 = [[None for _ in range(5)] for _ in range(slice_number)]
    temp_state_2 = [[None for _ in range(5)] for _ in range(slice_number)]
    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_S_vars = {}

    # Step 1: Calculate temp_state_1
    # Note: old_state is the output of the first P_L, so every bit has ul constant
    # 0; therefore temp1's XOR uses the ul0 special version and the output bit is
    # also declared as (0,*,*) (ul constant 0 folds all downstream pure-u / ul and b
    # indicators to 0).
    for z in range(slice_number):
        # x=0: temp_state_1[z][0] = old_state[z][0] + old_state[z][4]
        bit0 = Bit(model, f"{operation_name}_temp1_z{z}_x0", (0, '*', '*'))
        xor_vars0 = xor_with_ul0_input_no_delta_b(model, [old_state[z][0], old_state[z][4]], bit0, f"{operation_name}_temp1_z{z}_x0")
        temp_state_1[z][0] = bit0
        P_S_vars[f"temp1_z{z}_x0"] = xor_vars0

        # x=1: temp_state_1[z][1] = old_state[z][1] (direct copy)
        temp_state_1[z][1] = old_state[z][1]
        P_S_vars[f"temp1_z{z}_x1"] = {'delta_r': 0, 'delta_b': 0}

        # x=2: temp_state_1[z][2] = old_state[z][1] + old_state[z][2]
        bit2 = Bit(model, f"{operation_name}_temp1_z{z}_x2", (0, '*', '*'))
        xor_vars2 = xor_with_ul0_input_no_delta_b(model, [old_state[z][1], old_state[z][2]], bit2, f"{operation_name}_temp1_z{z}_x2")
        temp_state_1[z][2] = bit2
        P_S_vars[f"temp1_z{z}_x2"] = xor_vars2

        # x=3: temp_state_1[z][3] = old_state[z][3] (direct copy)
        temp_state_1[z][3] = old_state[z][3]
        P_S_vars[f"temp1_z{z}_x3"] = {'delta_r': 0, 'delta_b': 0}

        # x=4: temp_state_1[z][4] = old_state[z][3] + old_state[z][4]
        bit4 = Bit(model, f"{operation_name}_temp1_z{z}_x4", (0, '*', '*'))
        xor_vars4 = xor_with_ul0_input_no_delta_b(model, [old_state[z][3], old_state[z][4]], bit4, f"{operation_name}_temp1_z{z}_x4")
        temp_state_1[z][4] = bit4
        P_S_vars[f"temp1_z{z}_x4"] = xor_vars4

    # Step 2: Calculate temp_state_2 (XAND ul0 unified encoding; input temp1 bits are of type (0,*,*))
    for z in range(slice_number):
        for x in range(5):
            # Create XOR (XAND) operation bit
            xor_bit_name = f"{operation_name}_xor_z{z}_x{x}"
            xor_bit = Bit(model, xor_bit_name, ('*', '*', '*'))

            # x=0 cross-operation special case: temp1[z][2] is produced by a
            # single-input XOR (old[z][2] is a filtered constant zero), guaranteeing
            # b2==b1 and r2<=r1, so use the compressed encoding (12 constraints,
            # see xand_operation_ul0_x0).
            if x == 0 and is_const_zero_bit(old_state[z][2]):
                xand_vars = xand_operation_ul0_x0(
                    model, temp_state_1[z][x], temp_state_1[z][(x + 1) % 5],
                    temp_state_1[z][(x + 2) % 5], xor_bit, operation_name=xor_bit_name)
            else:
                xand_vars = xand_operation_ul0(
                    model, temp_state_1[z][x],
                    temp_state_1[z][(x + 1) % 5], temp_state_1[z][(x + 2) % 5],
                    xor_bit, operation_name=xor_bit_name)
            temp_state_2[z][x] = xor_bit
            P_S_vars[f"temp2_z{z}_x{x}"] = xand_vars
            # Backward-compatible interface: keep the and_ entry (attack scripts only perform an in check)
            P_S_vars[f"and_z{z}_x{x}"] = {'quad': 0}

    # Step 3: Calculate new_state
    for z in range(slice_number):
        # x=0: new_state[z][0] = temp_state_2[z][0] + temp_state_2[z][4]
        new_bit0 = Bit(model, f"{operation_name}_new_z{z}_x0", ('*', '*', '*'))
        xor_vars_new0 = xor_with_ul_input_no_delta_b(model, [temp_state_2[z][0], temp_state_2[z][4]], new_bit0, f"{operation_name}_new_z{z}_x0")
        new_state[z][0] = new_bit0
        P_S_vars[f"new_z{z}_x0"] = xor_vars_new0

        # x=1: new_state[z][1] = temp_state_2[z][1] + temp_state_2[z][0]
        new_bit1 = Bit(model, f"{operation_name}_new_z{z}_x1", ('*', '*', '*'))
        xor_vars_new1 = xor_with_ul_input_no_delta_b(model, [temp_state_2[z][1], temp_state_2[z][0]], new_bit1, f"{operation_name}_new_z{z}_x1")
        new_state[z][1] = new_bit1
        P_S_vars[f"new_z{z}_x1"] = xor_vars_new1

        # x=2: new_state[z][2] = temp_state_2[z][2] + 1 (direct copy)
        new_state[z][2] = temp_state_2[z][2]
        P_S_vars[f"new_z{z}_x2"] = {'delta_r': 0, 'delta_b': 0}

        # x=3: new_state[z][3] = temp_state_2[z][2] + temp_state_2[z][3]
        new_bit3 = Bit(model, f"{operation_name}_new_z{z}_x3", ('*', '*', '*'))
        xor_vars_new3 = xor_with_ul_input_no_delta_b(model, [temp_state_2[z][2], temp_state_2[z][3]], new_bit3, f"{operation_name}_new_z{z}_x3")
        new_state[z][3] = new_bit3
        P_S_vars[f"new_z{z}_x3"] = xor_vars_new3

        # x=4: new_state[z][4] = temp_state_2[z][4] (direct copy)
        new_state[z][4] = temp_state_2[z][4]
        P_S_vars[f"new_z{z}_x4"] = {'delta_r': 0, 'delta_b': 0}


    return temp_state_1, temp_state_2, new_state, P_S_vars



def create_first_P_S_operation_first_one(model, old_state, operation_name="P_S"):
    """
    MILP modeling for first round P_S function first bit (special handling)

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - new_state: New state after P_S operation
    - P_S_vars: Variables related to P_S operation
    """
    new_state = [[Bit(model, bit_type=(0, '*', '*'))] +
                 [Bit(model, bit_type=(0, 0, 0)) for _ in range(2)] +
                 [Bit(model, bit_type=(0, '*', 0))] +
                 [Bit(model, bit_type=(0, '*', '*'))]
                 for _ in range(slice_number)]
    P_S_vars = dict()
    for z in range(slice_number):
        aux_1 = model.addVar(vtype=GRB.BINARY, name=f"aux_1_{z}")
        aux_34 = model.addVar(vtype=GRB.BINARY, name=f"aux_34_{z}")

        # Add constraints
        model.addConstr((1 - new_state[z][0].b) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - aux_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - aux_1) + (1 - new_state[z][0].r) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][0].r) >= 1)
        model.addConstr(aux_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - aux_1) + aux_34 + new_state[z][3].r >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][0].b) >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][4].r) >= 1)
        model.addConstr(old_state[z][0].r + (1 - aux_34) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(aux_1 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + aux_1 >= 1)
        model.addConstr(aux_34 + new_state[z][3].r + (1 - new_state[z][4].r) >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (aux_1, aux_34)
    return new_state, P_S_vars


def create_first_P_S_operation_first_one_three_stage(model, old_state, operation_name="P_S"):
    """
    MILP modeling for first round P_S function first bit (special handling)
    with three-stage auxiliary first-round flag.

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - new_state: New state after P_S operation
    - P_S_vars: Variables related to P_S operation
    """
    new_state = [[Bit(model, bit_type=(0, '*', '*'))] +
                 [Bit(model, bit_type=(0, 0, 0)) for _ in range(2)] +
                 [Bit(model, bit_type=(0, '*', 0))] +
                 [Bit(model, bit_type=(0, '*', '*'))]
                 for _ in range(slice_number)]
    P_S_vars = dict()
    for z in range(slice_number):
        aux_1_0 = model.addVar(vtype=GRB.BINARY, name=f"aux_1_0_{z}")
        aux_1_1 = model.addVar(vtype=GRB.BINARY, name=f"aux_1_1_{z}")
        aux_34 = model.addVar(vtype=GRB.BINARY, name=f"aux_34_{z}")

        # Add constraints
        model.addConstr((1 - aux_34) + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - aux_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - aux_1_0) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - aux_1_1) + (1 - new_state[z][0].b) >= 1)
        model.addConstr((1 - aux_1_0) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - aux_1_1) + (1 - new_state[z][0].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - new_state[z][0].r) + (1 - new_state[z][0].b) >= 1)
        model.addConstr(aux_1_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(aux_1_1 + new_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].r + old_state[z][0].b + (1 - aux_34) >= 1)
        model.addConstr((1 - aux_1_0) + aux_34 + new_state[z][3].r + new_state[z][3].b >= 1)
        model.addConstr((1 - new_state[z][4].r) + (1 - new_state[z][4].b) >= 1)
        model.addConstr(aux_34 + new_state[z][3].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(aux_34 + new_state[z][3].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(aux_1_0 + (1 - new_state[z][0].b) + new_state[z][4].b >= 1)
        model.addConstr(aux_1_0 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - aux_1_1) + new_state[z][4].r + new_state[z][4].b >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (aux_1_0, aux_1_1, aux_34)
        for x in range(5):
            pass
    return new_state, P_S_vars


def create_first_P_S_operation_first_one_padding(model, old_state, operation_name="P_S"):
    """
    MILP modeling for first round P_S function first bit (special handling)
    with padding considered for auxiliary first-round flag.

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - new_state: New state after P_S operation
    - P_S_vars: Variables related to P_S operation
    """
    # Constant-zero input folding: for a slice whose old_state[z][0] is constant
    # zero (e.g. the padding region z>=slice_number-2 or z=63), the output is
    # uniquely all-zero on the projection and aux has no external reference, so
    # fold directly to a constant without creating variables/constraints. Taking
    # the minimal aux=0 matches the original model's optimal-solution behavior
    # (a tighter capacity only makes the objective worse).
    folded = [is_const_zero_bit(old_state[z][0]) or z == slice_number - 1
              for z in range(slice_number)]
    new_state = [
        [Bit(model, bit_type=(0, 0, 0)) for _ in range(5)]
        if folded[z] else
        [Bit(model, bit_type=(0, '*', '*'))] +
        [Bit(model, bit_type=(0, 0, 0)) for _ in range(2)] +
        [Bit(model, bit_type=(0, '*', '*'))] +
        [Bit(model, bit_type=(0, '*', '*'))]
        for z in range(slice_number)
    ]
    P_S_vars = dict()
    for z in range(slice_number):
        if folded[z]:
            P_S_vars[f'{z}_vars'] = (0, 0, 0)
            for x in range(5):
                pass
            continue
        aux_1 = model.addVar(vtype=GRB.BINARY, name=f"aux_1_{z}")
        aux_34 = model.addVar(vtype=GRB.BINARY, name=f"aux_34_{z}")

        # Add constraints
        model.addConstr((1 - aux_34) + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - aux_1) + (1 - new_state[z][0].b) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - aux_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - aux_1) + (1 - new_state[z][0].r) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + (1 - new_state[z][0].b) >= 1)
        model.addConstr(aux_34 + (1 - new_state[z][0].r) + new_state[z][3].r >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr(aux_1 + new_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - old_state[z][0].b) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(aux_1 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - aux_1) + aux_34 + new_state[z][3].r + new_state[z][3].b >= 1)
        model.addConstr(old_state[z][0].r + old_state[z][0].b + (1 - aux_34) >= 1)
        model.addConstr(aux_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(aux_1 + (1 - new_state[z][0].b) + new_state[z][4].b >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr(aux_34 + (1 - new_state[z][0].b) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][0].r) >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (aux_1, aux_34)
        for x in range(5):
            pass
    z = slice_number - 1
    P_S_vars[f'{z}_vars'] = (0, 0, 0)
    for x in range(5):
        pass
    return new_state, P_S_vars


def create_first_P_S_operation_first_one_padding_three_stage(model, old_state, operation_name="P_S"):
    """
    MILP modeling for first round P_S function first bit (special handling)
    with padding and three-stage auxiliary first-round flag.

    Parameters:
        pass
    - model: Gurobi model object
    - old_state: 64x5 2D state array [z][x]
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - new_state: New state after P_S operation
    - P_S_vars: Variables related to P_S operation
    """
    # Constant-zero input folding: for a slice whose old_state[z][0] is constant
    # zero (e.g. the padding region z>=slice_number-2 or z=63), the output is
    # uniquely all-zero on the projection and aux has no external reference, so
    # fold directly to a constant without creating variables/constraints. Taking
    # the minimal aux=0 matches the original model's optimal-solution behavior
    # (a tighter capacity only makes the objective worse).
    folded = [is_const_zero_bit(old_state[z][0]) or z == slice_number - 1
              for z in range(slice_number)]
    new_state = [
        [Bit(model, bit_type=(0, 0, 0)) for _ in range(5)]
        if folded[z] else
        [Bit(model, bit_type=(0, '*', '*'))] +
        [Bit(model, bit_type=(0, 0, 0)) for _ in range(2)] +
        [Bit(model, bit_type=(0, '*', '*'))] +
        [Bit(model, bit_type=(0, '*', '*'))]
        for z in range(slice_number)
    ]
    P_S_vars = dict()
    for z in range(slice_number):
        if folded[z]:
            P_S_vars[f'{z}_vars'] = (0, 0, 0)
            for x in range(5):
                pass
            continue
        aux_1_0 = model.addVar(vtype=GRB.BINARY, name=f"aux_1_0_{z}")
        aux_1_1 = model.addVar(vtype=GRB.BINARY, name=f"aux_1_1_{z}")
        aux_34 = model.addVar(vtype=GRB.BINARY, name=f"aux_34_{z}")

        # Add constraints
        model.addConstr((1 - aux_34) + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - aux_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - aux_1_0) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - aux_1_1) + (1 - new_state[z][0].b) >= 1)
        model.addConstr((1 - aux_1_0) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - aux_1_1) + (1 - new_state[z][0].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - new_state[z][0].r) + (1 - new_state[z][0].b) >= 1)
        model.addConstr(aux_1_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(aux_1_1 + new_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].r + old_state[z][0].b + (1 - aux_34) >= 1)
        model.addConstr((1 - aux_1_0) + aux_34 + new_state[z][3].r + new_state[z][3].b >= 1)
        model.addConstr((1 - new_state[z][4].r) + (1 - new_state[z][4].b) >= 1)
        model.addConstr(aux_34 + new_state[z][3].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(aux_34 + new_state[z][3].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(aux_1_0 + (1 - new_state[z][0].b) + new_state[z][4].b >= 1)
        model.addConstr(aux_1_0 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - aux_1_1) + new_state[z][4].r + new_state[z][4].b >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (aux_1_0, aux_1_1, aux_34)
        for x in range(5):
            pass
    z = slice_number - 1
    P_S_vars[f'{z}_vars'] = (0, 0, 0)
    return new_state, P_S_vars


def create_first_P_S_operation_first_one_no_initial(model, operation_name="P_S"):
    """
    MILP modeling for first round P_S function without initial state.

    Parameters:
        pass
    - model: Gurobi model object
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - second_initial_state: Initial state
    - P_S_vars: Variables related to P_S operation
    """
    initial_state = [[Bit(model, bit_type=(0, '*', '*')) for _ in range(2)] +
                     [Bit(model, bit_type=(0, 0, 0))] +
                     [Bit(model, bit_type=(0, '*', 0))] +
                     [Bit(model, bit_type=(0, '*', '*'))]
                     for _ in range(slice_number)]
    P_S_vars = dict()
    for z in range(slice_number - 1):
        # Different auxiliary flags
        aux_1 = model.addVar(vtype=GRB.BINARY, name=f"aux_1_{z}")
        aux_34 = model.addVar(vtype=GRB.BINARY, name=f"aux_34_{z}")

        # Add constraints
        model.addConstr((1 - aux_1) + (1 - initial_state[z][0].b) + (1 - initial_state[z][4].b) >= 1)
        model.addConstr((1 - aux_34) + (1 - initial_state[z][3].r) >= 1)
        model.addConstr((1 - aux_1) + (1 - initial_state[z][0].r) + (1 - initial_state[z][4].r) >= 1)
        model.addConstr((1 - initial_state[z][1].r) + (1 - initial_state[z][1].b) >= 1)
        model.addConstr((1 - initial_state[z][0].b) + initial_state[z][1].b >= 1)
        model.addConstr((1 - aux_1) + aux_34 + initial_state[z][3].r >= 1)
        model.addConstr((1 - initial_state[z][0].r) + initial_state[z][1].r >= 1)
        model.addConstr(aux_1 + initial_state[z][0].r + (1 - initial_state[z][4].r) >= 1)
        model.addConstr(aux_1 + initial_state[z][0].b + (1 - initial_state[z][4].b) >= 1)
        model.addConstr(initial_state[z][1].r + (1 - initial_state[z][3].r) >= 1)
        model.addConstr(initial_state[z][1].b + (1 - initial_state[z][4].b) >= 1)
        model.addConstr((1 - aux_1) + initial_state[z][0].r + initial_state[z][4].r + initial_state[z][0].b + initial_state[z][4].b >= 1)
        model.addConstr(aux_1 + (1 - initial_state[z][1].r) + initial_state[z][4].r >= 1)
        model.addConstr(aux_1 + (1 - initial_state[z][1].b) + initial_state[z][4].b >= 1)
        model.addConstr((1 - aux_34) + initial_state[z][1].r + initial_state[z][1].b >= 1)
        model.addConstr((1 - initial_state[z][4].r) + (1 - initial_state[z][1].b) >= 1)
        model.addConstr(aux_34 + (1 - initial_state[z][4].b) >= 1)
        model.addConstr(aux_34 + initial_state[z][3].r + (1 - initial_state[z][4].r) >= 1)

        P_S_vars[f'{z}_vars'] = (aux_1, aux_34)
    P_S_vars[f'{slice_number - 1}_vars'] = (0, 0)
    return initial_state, P_S_vars


def last_Hash_collision(model, state, operation_name="P_S"):
    """
    MILP modeling for final hash collision detection.

    Parameters:
        pass
    - model: Gurobi model object
    - state: State array
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - P_Svars: Variables related to collision detection
    """
    P_Svars = dict()
    for z in range(slice_number):
        # Different schemes
        c0 = model.addVar(vtype=GRB.BINARY, name=f"c_0_{operation_name}")
        c1 = model.addVar(vtype=GRB.BINARY, name=f"c_1_{operation_name}")
        c2 = model.addVar(vtype=GRB.BINARY, name=f"c_2_{operation_name}")
        c3 = model.addVar(vtype=GRB.BINARY, name=f"c_3_{operation_name}")
        c4 = model.addVar(vtype=GRB.BINARY, name=f"c_4_{operation_name}")

        # Whether each bit is not u
        u = [model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{operation_name}") for i in range(5)]

        # Constraints for u variables
        for i in range(5):
            model.addConstr(u[i] <= 1 - state[z][i].ul + state[z][i].r + state[z][i].b)

        # Collision detection constraints
        model.addConstr(u[0] + u[4] + (1 - c2) >= 1)
        model.addConstr(u[1] + u[3] + (1 - c2) >= 1)
        model.addConstr(u[2] + u[4] + (1 - c2) >= 1)
        model.addConstr(u[0] + u[3] + (1 - c3) >= 1)
        model.addConstr(u[1] + u[4] + (1 - c3) >= 1)
        model.addConstr(u[0] + u[2] + u[4] + (1 - c3) >= 1)
        model.addConstr(u[1] + u[2] + (1 - c3) >= 1)
        model.addConstr(u[3] + u[4] + (1 - c3) >= 1)
        model.addConstr(u[0] + u[1] + u[2] + u[3] + u[4] >= 1)
        model.addConstr((1 - c1) + (1 - c4) >= 1)
        model.addConstr(u[4] + (1 - c4) >= 1)
        model.addConstr(u[4] + (1 - c1) >= 1)
        model.addConstr(u[3] + (1 - c4) >= 1)
        model.addConstr(u[3] + (1 - c1) >= 1)
        model.addConstr((1 - u[2]) + (1 - u[3]) + (1 - u[4]) + (1 - c2) >= 1)
        model.addConstr((1 - u[1]) + (1 - u[3]) + (1 - c2) >= 1)
        model.addConstr((1 - c0) + (1 - c3) >= 1)
        model.addConstr(u[2] + (1 - c4) >= 1)
        model.addConstr(u[2] + (1 - c1) >= 1)
        model.addConstr((1 - c0) + (1 - c2) >= 1)
        model.addConstr(u[0] + (1 - c0) >= 1)
        model.addConstr((1 - u[0]) + (1 - u[2]) + (1 - u[3]) + c0 + c1 + c2 >= 1)
        model.addConstr((1 - u[1]) + (1 - u[2]) + (1 - u[3]) + (1 - u[4]) + (1 - c0) >= 1)
        model.addConstr(u[1] + (1 - c4) >= 1)
        model.addConstr((1 - u[0]) + u[3] + (1 - u[4]) + (1 - c2) >= 1)
        model.addConstr(u[0] + (1 - u[1]) + (1 - u[2]) + (1 - u[4]) + c2 + c4 >= 1)
        model.addConstr(u[1] + (1 - c1) >= 1)
        model.addConstr(u[3] + u[4] + (1 - c0) >= 1)
        model.addConstr((1 - u[0]) + (1 - u[1]) + (1 - u[2]) + c0 + c1 + c2 >= 1)
        model.addConstr((1 - u[0]) + (1 - u[3]) + (1 - u[4]) + (1 - c3) >= 1)
        model.addConstr(u[0] + u[2] + (1 - c2) >= 1)
        model.addConstr((1 - u[0]) + u[1] + u[2] + (1 - u[3]) + (1 - u[4]) + c2 >= 1)
        model.addConstr(u[1] + u[4] + (1 - c0) >= 1)
        model.addConstr((1 - u[0]) + (1 - u[1]) + (1 - u[4]) + c0 + c1 + c3 >= 1)
        model.addConstr((1 - u[0]) + u[1] + (1 - u[2]) + u[3] + (1 - u[4]) + c3 >= 1)
        model.addConstr(u[2] + u[3] + (1 - c0) >= 1)
        model.addConstr((1 - u[0]) + (1 - u[1]) + u[2] + (1 - u[3]) + u[4] + c3 >= 1)
        model.addConstr(u[0] + (1 - u[1]) + u[2] + (1 - u[3]) + (1 - u[4]) + c3 >= 1)
        model.addConstr(u[0] + u[1] + (1 - u[2]) + (1 - u[3]) + (1 - u[4]) + c3 >= 1)
        model.addConstr(u[0] + (1 - u[1]) + (1 - u[2]) + (1 - u[3]) + u[4] + c3 >= 1)
        model.addConstr((1 - c3) + (1 - c4) >= 1)

        P_Svars[f'{z}_vars'] = (c0, c1, c2, c3, c4)

    return P_Svars


def new_Hash_collision(model, state, operation_name="P_S"):
    """
    MILP modeling for final hash collision detection.

    Parameters:
        pass
    - model: Gurobi model object
    - state: State array
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - P_Svars: Variables related to collision detection
    """
    P_Svars = dict()
    for z in range(slice_number):
        # Different schemes for various bit counts
        c1 = model.addVar(vtype=GRB.BINARY, name=f"c_1_{operation_name}_{z}")
        c2 = model.addVar(vtype=GRB.BINARY, name=f"c_2_{operation_name}_{z}")
        c3 = model.addVar(vtype=GRB.BINARY, name=f"c_3_{operation_name}_{z}")
        c4 = model.addVar(vtype=GRB.BINARY, name=f"c_4_{operation_name}_{z}")
        c5 = model.addVar(vtype=GRB.BINARY, name=f"c_5_{operation_name}_{z}")
        special_c4 = model.addVar(vtype=GRB.BINARY, name=f"special_c4_{operation_name}")

        # Whether each bit is not u
        u = [model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{operation_name}") for i in range(5)]

        # Constraints for u variables
        for i in range(5):
            model.addConstr(u[i] <= 1 - state[z][i].ul + state[z][i].r + state[z][i].b)

        # Collision detection constraints
        model.addConstr(c1 <= u[0] + u[1] + u[2] + u[3] + u[4])
        model.addConstr(2 * c2 <= u[0] + u[1] + u[2] + u[3] + u[4])
        model.addConstr(3 * c3 <= u[0] + u[1] + u[2] + u[3] + u[4])
        model.addConstr(4 * c4 <= u[0] + u[1] + u[2] + u[3] + u[4])
        model.addConstr(5 * c5 <= u[0] + u[1] + u[2] + u[3] + u[4])
        model.addConstr(4 * special_c4 <= u[1] + u[2] + u[3] + u[4])
        model.addConstr(c1 + c2 + c3 + c4 + c5 + special_c4 <= 1)
        P_Svars[f'{z}_vars'] = (c1, c2, c3, c4, c5, special_c4)

    return P_Svars


def new_simple_Hash_collision(model, state, operation_name="P_S"):
    """
    MILP modeling for final hash collision detection (simplified).

    Parameters:
        pass
    - model: Gurobi model object
    - state: State array
    - operation_name: Operation name for variable naming

    Returns:
        pass
    - P_Svars: Variables related to collision detection
    """
    P_Svars = dict()
    for z in range(slice_number):
        # Different schemes for specific bit counts
        c4 = model.addVar(vtype=GRB.BINARY, name=f"c_4_{operation_name}")
        c5 = model.addVar(vtype=GRB.BINARY, name=f"c_5_{operation_name}")
        special_c4 = model.addVar(vtype=GRB.BINARY, name=f"special_c4_{operation_name}")

        # Whether each bit is not u
        u = [model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{operation_name}") for i in range(5)]

        # Constraints for u variables
        for i in range(5):
            model.addConstr(u[i] <= 1 - state[z][i].ul + state[z][i].r + state[z][i].b)

        # Collision detection constraints
        model.addConstr(4 * c4 <= u[0] + u[1] + u[2] + u[3] + u[4])
        model.addConstr(5 * c5 <= u[0] + u[1] + u[2] + u[3] + u[4])
        model.addConstr(4 * special_c4 <= u[1] + u[2] + u[3] + u[4])
        model.addConstr(c4 + c5 + special_c4 <= 1)
        P_Svars[f'{z}_vars'] = (c4, c5, special_c4)

    return P_Svars
