from operation import *
from gurobipy import GRB

slice_number = 64


def create_P_L_operation(model, old_state, operation_name="P_L"):
    'Function create P L operation.'


    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_L_vars = {}


    offsets = [
        [19, 28],  # x=0
        [61, 39],  # x=1
        [1, 6],  # x=2
        [10, 17],  # x=3
        [7, 41]  # x=4
    ]


    for z in range(slice_number):
        for x in range(5):

            new_bit = Bit(model, f"{operation_name}_new_z{z}_x{x}",('*','*','*',0,'*'))


            input_bits = [
                old_state[z][x],  # old_state[z][x]
                old_state[(z + offsets[x][0]) % slice_number][x],
                old_state[(z + offsets[x][1]) % slice_number][x]
            ]


            xor_vars = add_xor_red_cancel_only(model, input_bits, new_bit, name = f"{operation_name}_new_z{z}_x{x}")


            new_state[z][x] = new_bit
            P_L_vars[f"new_z{z}_x{x}"] = xor_vars


    # _add_P_L_condition_constraints(model, P_L_vars, old_state, new_state, offsets, operation_name)

    return new_state, P_L_vars

def create_last_P_L_operation(model, old_state, operation_name="P_L"):
    'Function create last P L operation.'


    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_L_vars = {}


    offsets = [
        [19, 28],  # x=0
        [61, 39],  # x=1
        [1, 6],  # x=2
        [10, 17],  # x=3
        [7, 41]  # x=4
    ]


    for z in range(slice_number):
        for x in range(5):

            new_bit = Bit(model, f"{operation_name}_new_z{z}_x{x}",('*','*','*',0,'*'))


            input_bits = [
                old_state[z][x],  # old_state[z][x]
                old_state[(z + offsets[x][0]) % slice_number][x],
                old_state[(z + offsets[x][1]) % slice_number][x]
            ]


            xor_vars = add_xor_red_cancel_only(model, input_bits, new_bit,0, name = f"{operation_name}_new_z{z}_x{x}")


            new_state[z][x] = new_bit
            P_L_vars[f"new_z{z}_x{x}"] = xor_vars


    # _add_P_L_condition_constraints(model, P_L_vars, old_state, new_state, offsets, operation_name)

    return new_state, P_L_vars

def create_first_P_L_operation(model, old_state, operation_name="P_L"):
    'Function create first P L operation.'


    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_L_vars = {}


    offsets = [
        [19, 28],  # x=0
        [61, 39],  # x=1
        [1, 6],  # x=2
        [10, 17],  # x=3
        [7, 41]  # x=4
    ]


    for z in range(slice_number):
        for x in range(5):

            new_bit = Bit(model, f"{operation_name}_new_z{z}_x{x}", (0, '*', '*', '*', 0))


            input_bits = [
                old_state[z][x],  # old_state[z][x]
                old_state[(z + offsets[x][0]) % slice_number][x],
                old_state[(z + offsets[x][1]) % slice_number][x]
            ]


            xor_vars = add_xor_general_rb_cancel(model, input_bits, new_bit, name = f"{operation_name}_new_z{z}_x{x}")


            new_state[z][x] = new_bit
            P_L_vars[f"new_z{z}_x{x}"] = xor_vars


    _add_P_L_condition_constraints(model, P_L_vars, old_state, new_state, offsets, operation_name)

    return new_state, P_L_vars


def _add_P_L_condition_constraints(model, P_L_vars, old_state, new_state, offsets, operation_name):
    'Function add P L condition constraints.'




    # x_old_state2new_state: Condition constant propagation from old state to new state
    x_old_state2new_state = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state({z},{x})2new_state({z},{x})") for x in range(5)] for z in range(slice_number)]


    # x_old_state2offset1: Condition constant propagation from old state to first offset position
    x_old_state2offset1 = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state({z},{x})2offset1({(z + offsets[x][0]) % slice_number},{x})") for x in range(5)] for z in
                           range(slice_number)]


    # x_old_state2offset2: Condition constant propagation from old state to second offset position
    x_old_state2offset2 = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state({z},{x})2offset2({(z + offsets[x][1]) % slice_number},{x})") for x in range(5)] for z in
                           range(slice_number)]


    for z in range(slice_number):
        for x in range(5):
            model.addConstr(old_state[z][x].cond == (x_old_state2new_state[z][x] + x_old_state2offset1[z][x] + x_old_state2offset2[z][x]))


    for z in range(slice_number):
        for x in range(5):
            delta_r = P_L_vars[f"new_z{z}_x{x}"]['delta_r']
            delta_b = P_L_vars[f"new_z{z}_x{x}"]['delta_b']
            has_ul = P_L_vars[f"new_z{z}_x{x}"]['has_ul']
            P_L_vars[f"new_z{z}_x{x}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'new_z{z}_x{x}_new_cond')

            input1_cond = x_old_state2new_state[z][x]  # old_state[z][x]
            input2_cond = x_old_state2offset1[(z - offsets[x][0]) % slice_number][x]
            input3_cond = x_old_state2offset2[(z - offsets[x][1]) % slice_number][x]
            model.addConstr(P_L_vars[f"new_z{z}_x{x}"]['new_cond'] <= delta_r + delta_b)
            model.addConstr(P_L_vars[f"new_z{z}_x{x}"]['new_cond']  <= 1 - has_ul)
            model.addConstr(new_state[z][x].cond >= input1_cond + input2_cond + input3_cond)
            model.addConstr(new_state[z][x].cond <= input1_cond + input2_cond + input3_cond + P_L_vars[f"new_z{z}_x{x}"]['new_cond'])
            model.addConstr(new_state[z][x].cond >= P_L_vars[f"new_z{z}_x{x}"]['new_cond'])


def create_P_S_operation(model, old_state, operation_name="P_S"):
    'Function create P S operation.'


    temp_state_1 = [[None for _ in range(5)] for _ in range(slice_number)]
    temp_state_2 = [[None for _ in range(5)] for _ in range(slice_number)]
    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_S_vars = {}


    for z in range(slice_number):
        # x=0: temp_state_1[z][0] = old_state[z][0] + old_state[z][4]
        bit0 = Bit(model, f"{operation_name}_temp1_z{z}_x0",('*','*','*',0,'*'))
        xor_vars0 = add_xor_red_cancel_only(model, [old_state[z][0], old_state[z][4]], bit0, name = f"{operation_name}_temp1_z{z}_x0")
        temp_state_1[z][0] = bit0
        P_S_vars[f"temp1_z{z}_x0"] = xor_vars0


        temp_state_1[z][1] = old_state[z][1]
        P_S_vars[f"temp1_z{z}_x1"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}

        # x=2: temp_state_1[z][2] = old_state[z][1] + old_state[z][2]
        bit2 = Bit(model, f"{operation_name}_temp1_z{z}_x2",('*','*','*',0,'*'))
        xor_vars2 = add_xor_red_cancel_only(model, [old_state[z][1], old_state[z][2]], bit2, name = f"{operation_name}_temp1_z{z}_x2")
        temp_state_1[z][2] = bit2
        P_S_vars[f"temp1_z{z}_x2"] = xor_vars2


        temp_state_1[z][3] = old_state[z][3]
        P_S_vars[f"temp1_z{z}_x3"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}

        # x=4: temp_state_1[z][4] = old_state[z][3] + old_state[z][4]
        bit4 = Bit(model, f"{operation_name}_temp1_z{z}_x4",('*','*','*',0,'*'))
        xor_vars4 = add_xor_red_cancel_only(model, [old_state[z][3], old_state[z][4]], bit4, name = f"{operation_name}_temp1_z{z}_x4")
        temp_state_1[z][4] = bit4
        P_S_vars[f"temp1_z{z}_x4"] = xor_vars4


    for z in range(slice_number):
        for x in range(5):

            and_bit_name = f"{operation_name}_and_z{z}_x{x}"
            and_bit = Bit(model, and_bit_name, ('*', '*', '*', 0, '*'))
            and_vars = add_and_no_cond(model, temp_state_1[z][(x + 1) % 5], temp_state_1[z][(x + 2) % 5], and_bit, name=and_bit_name)


            P_S_vars[f"and_z{z}_x{x}"] = and_vars


            xor_bit_name = f"{operation_name}_xor_z{z}_x{x}"
            xor_bit = Bit(model, xor_bit_name, ('*', '*', '*', 0, '*'))


            xor_vars = add_xor_red_cancel_only(model, [temp_state_1[z][x], and_bit], xor_bit, name = xor_bit_name)
            temp_state_2[z][x] = xor_bit
            P_S_vars[f"temp2_z{z}_x{x}"] = xor_vars


    for z in range(slice_number):
        # x=0: new_state[z][0] = temp_state_2[z][0] + temp_state_2[z][4]
        new_bit0 = Bit(model, f"{operation_name}_new_z{z}_x0",('*','*','*',0,'*'))
        xor_vars_new0 = add_xor_red_cancel_only(model, [temp_state_2[z][0], temp_state_2[z][4]], new_bit0, name = f"{operation_name}_new_z{z}_x0")
        new_state[z][0] = new_bit0
        P_S_vars[f"new_z{z}_x0"] = xor_vars_new0

        # x=1: new_state[z][1] = temp_state_2[z][1] + temp_state_2[z][0]
        new_bit1 = Bit(model, f"{operation_name}_new_z{z}_x1",('*','*','*',0,'*'))
        xor_vars_new1 = add_xor_red_cancel_only(model, [temp_state_2[z][1], temp_state_2[z][0]], new_bit1, name = f"{operation_name}_new_z{z}_x1")
        new_state[z][1] = new_bit1
        P_S_vars[f"new_z{z}_x1"] = xor_vars_new1


        new_state[z][2] = temp_state_2[z][2]
        P_S_vars[f"new_z{z}_x2"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}

        # x=3: new_state[z][3] = temp_state_2[z][2] + temp_state_2[z][3]
        new_bit3 = Bit(model, f"{operation_name}_new_z{z}_x3",('*','*','*',0,'*'))
        xor_vars_new3 = add_xor_red_cancel_only(model, [temp_state_2[z][2], temp_state_2[z][3]], new_bit3, name = f"{operation_name}_new_z{z}_x3")
        new_state[z][3] = new_bit3
        P_S_vars[f"new_z{z}_x3"] = xor_vars_new3


        new_state[z][4] = temp_state_2[z][4]
        P_S_vars[f"new_z{z}_x4"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}


    # _add_P_S_condition_constraints(model, P_S_vars, old_state, temp_state_1, temp_state_2, new_state, operation_name)

    return temp_state_1, temp_state_2, new_state, P_S_vars

def create_second_P_S_operation(model, old_state, operation_name="P_S"):
    'Function create second P S operation.'


    temp_state_1 = [[None for _ in range(5)] for _ in range(slice_number)]
    temp_state_2 = [[None for _ in range(5)] for _ in range(slice_number)]
    new_state = [[None for _ in range(5)] for _ in range(slice_number)]
    P_S_vars = {}


    for z in range(slice_number):
        # x=0: temp_state_1[z][0] = old_state[z][0] + old_state[z][4]
        bit0 = Bit(model, f"{operation_name}_temp1_z{z}_x0")
        xor_vars0 = add_xor_linear_rb_cancel(model, [old_state[z][0], old_state[z][4]], bit0, name = f"{operation_name}_temp1_z{z}_x0")
        temp_state_1[z][0] = bit0
        P_S_vars[f"temp1_z{z}_x0"] = xor_vars0


        temp_state_1[z][1] = old_state[z][1]
        P_S_vars[f"temp1_z{z}_x1"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}

        # x=2: temp_state_1[z][2] = old_state[z][1] + old_state[z][2]
        bit2 = Bit(model, f"{operation_name}_temp1_z{z}_x2")
        xor_vars2 = add_xor_linear_rb_cancel(model, [old_state[z][1], old_state[z][2]], bit2, name = f"{operation_name}_temp1_z{z}_x2")
        temp_state_1[z][2] = bit2
        P_S_vars[f"temp1_z{z}_x2"] = xor_vars2


        temp_state_1[z][3] = old_state[z][3]
        P_S_vars[f"temp1_z{z}_x3"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}

        # x=4: temp_state_1[z][4] = old_state[z][3] + old_state[z][4]
        bit4 = Bit(model, f"{operation_name}_temp1_z{z}_x4")
        xor_vars4 = add_xor_linear_rb_cancel(model, [old_state[z][3], old_state[z][4]], bit4, name = f"{operation_name}_temp1_z{z}_x4")
        temp_state_1[z][4] = bit4
        P_S_vars[f"temp1_z{z}_x4"] = xor_vars4


    for z in range(slice_number):
        for x in range(5):

            and_bit_name = f"{operation_name}_and_z{z}_x{x}"
            and_bit = Bit(model, and_bit_name, ('*', '*', '*', 0, '*'))
            and_vars = add_and_no_cond(model, temp_state_1[z][(x + 1) % 5], temp_state_1[z][(x + 2) % 5], and_bit, name=and_bit_name)


            P_S_vars[f"and_z{z}_x{x}"] = and_vars


            xor_bit_name = f"{operation_name}_xor_z{z}_x{x}"
            xor_bit = Bit(model, xor_bit_name, ('*', '*', '*', 0, '*'))


            xor_vars = add_xor_red_cancel_only(model, [temp_state_1[z][x], and_bit], xor_bit, name = xor_bit_name)
            temp_state_2[z][x] = xor_bit
            P_S_vars[f"temp2_z{z}_x{x}"] = xor_vars


    for z in range(slice_number):
        # x=0: new_state[z][0] = temp_state_2[z][0] + temp_state_2[z][4]
        new_bit0 = Bit(model, f"{operation_name}_new_z{z}_x0",('*','*','*',0,'*'))
        xor_vars_new0 = add_xor_red_cancel_only(model, [temp_state_2[z][0], temp_state_2[z][4]], new_bit0, name = f"{operation_name}_new_z{z}_x0")
        new_state[z][0] = new_bit0
        P_S_vars[f"new_z{z}_x0"] = xor_vars_new0

        # x=1: new_state[z][1] = temp_state_2[z][1] + temp_state_2[z][0]
        new_bit1 = Bit(model, f"{operation_name}_new_z{z}_x1",('*','*','*',0,'*'))
        xor_vars_new1 = add_xor_red_cancel_only(model, [temp_state_2[z][1], temp_state_2[z][0]], new_bit1, name = f"{operation_name}_new_z{z}_x1")
        new_state[z][1] = new_bit1
        P_S_vars[f"new_z{z}_x1"] = xor_vars_new1


        new_state[z][2] = temp_state_2[z][2]
        P_S_vars[f"new_z{z}_x2"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}

        # x=3: new_state[z][3] = temp_state_2[z][2] + temp_state_2[z][3]
        new_bit3 = Bit(model, f"{operation_name}_new_z{z}_x3",('*','*','*',0,'*'))
        xor_vars_new3 = add_xor_red_cancel_only(model, [temp_state_2[z][2], temp_state_2[z][3]], new_bit3, name = f"{operation_name}_new_z{z}_x3")
        new_state[z][3] = new_bit3
        P_S_vars[f"new_z{z}_x3"] = xor_vars_new3


        new_state[z][4] = temp_state_2[z][4]
        P_S_vars[f"new_z{z}_x4"] = {'delta_r': 0, 'delta_b': 0,'new_cond':0}


    _add_P_S_condition_constraints(model, P_S_vars, old_state, temp_state_1, temp_state_2, new_state, operation_name)

    return temp_state_1, temp_state_2, new_state, P_S_vars




def create_first_P_S_operation_first_one(model, old_state, operation_name="P_S"):
    'Function create first P S operation first one.'
    new_state = [[Bit(model, bit_type=(0, '*', '*', 0, 0))] +
                 [Bit(model, bit_type=(0, 0, 0, 0, 0)) for _ in range(2)] +[Bit(model, bit_type=(0, '*', 0, 0, 0))]+
                 [Bit(model, bit_type=(0, '*', '*', 0, 0))]
                 for _ in range(slice_number)]
    P_S_vars = dict()
    for z in range(slice_number):
        cond_1 = model.addVar(vtype=GRB.BINARY, name="cond_1"
                                                     f" _{z}")
        cond_34 = model.addVar(vtype=GRB.BINARY, name=f"cond_34_{z}")


        model.addConstr((1 - new_state[z][0].b) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - cond_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - cond_1) + (1 - new_state[z][0].r) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][0].r) >= 1)
        model.addConstr(cond_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - cond_1) + cond_34 + new_state[z][3].r >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][0].b) >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][4].r) >= 1)
        model.addConstr(old_state[z][0].r + (1 - cond_34) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(cond_1 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + cond_1 >= 1)
        model.addConstr(cond_34 + new_state[z][3].r + (1 - new_state[z][4].r) >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (cond_1, cond_34)
    return new_state, P_S_vars


def create_first_P_S_operation_first_one_constant_cond(model, old_state, operation_name="P_S"):
    'Function create first P S operation first one constant cond.'
    new_state = [[Bit(model, bit_type=(0, '*', '*', '*', 0))] +
                 [Bit(model, bit_type=(0, 0, 0, '*', 0)) for _ in range(2)] +[Bit(model, bit_type=(0, '*', 0, '*', 0))]+
                 [Bit(model, bit_type=(0, '*', '*', '*', 0))]
                 for _ in range(slice_number)]
    P_S_vars = dict()
    for z in range(slice_number):
        cond_1 = model.addVar(vtype=GRB.BINARY, name=f"cond_1_{z}")
        cond_34 = model.addVar(vtype=GRB.BINARY, name=f"cond_34_{z}")


        model.addConstr((1 - new_state[z][0].b) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - cond_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - cond_1) + (1 - new_state[z][0].r) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][0].r) >= 1)
        model.addConstr(cond_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - cond_1) + cond_34 + new_state[z][3].r >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][0].b) >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][4].r) >= 1)
        model.addConstr(old_state[z][0].r + (1 - cond_34) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(cond_1 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + cond_1 >= 1)
        model.addConstr(cond_34 + new_state[z][3].r + (1 - new_state[z][4].r) >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (cond_1, cond_34)
        for x in range(5):
            model.addConstr(1 - cond_1 >= new_state[z][x].cond)
            model.addConstr(1 - cond_34 >= new_state[z][x].cond)
            P_S_vars[f'{z}_constant_cond_{x}'] = new_state[z][x].cond
    return new_state, P_S_vars

def create_first_P_S_operation_first_one_constant_cond_three_stage(model, old_state, operation_name="P_S"):
    'Function create first P S operation first one constant cond three stage.'
    new_state = [[Bit(model, bit_type=(0, '*', '*', '*', 0))] +
                 [Bit(model, bit_type=(0, 0, 0, '*', 0)) for _ in range(2)] +[Bit(model, bit_type=(0, '*', 0, '*', 0))]+
                 [Bit(model, bit_type=(0, '*', '*', '*', 0))]
                 for _ in range(slice_number)]
    P_S_vars = dict()
    for z in range(slice_number):
        cond_1_0 = model.addVar(vtype=GRB.BINARY, name=f"cond_1_0_{z}")
        cond_1_1 = model.addVar(vtype=GRB.BINARY, name=f"cond_1_1_{z}")
        cond_34 = model.addVar(vtype=GRB.BINARY, name=f"cond_34_{z}")


        model.addConstr((1 - cond_34) + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - cond_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - cond_1_0) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - cond_1_1) + (1 - new_state[z][0].b) >= 1)
        model.addConstr((1 - cond_1_0) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - cond_1_1) + (1 - new_state[z][0].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - new_state[z][0].r) + (1 - new_state[z][0].b) >= 1)
        model.addConstr(cond_1_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(cond_1_1 + new_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].r + old_state[z][0].b + (1 - cond_34) >= 1)
        model.addConstr((1 - cond_1_0) + cond_34 + new_state[z][3].r + new_state[z][3].b >= 1)
        model.addConstr((1 - new_state[z][4].r) + (1 - new_state[z][4].b) >= 1)
        model.addConstr(cond_34 + new_state[z][3].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(cond_34 + new_state[z][3].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(cond_1_0 + (1 - new_state[z][0].b) + new_state[z][4].b >= 1)
        model.addConstr(cond_1_0 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - cond_1_1) + new_state[z][4].r + new_state[z][4].b >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (cond_1_0,cond_1_1, cond_34)
        for x in range(5):
            model.addConstr(1 - cond_1_0 >= new_state[z][x].cond)
            model.addConstr(1 - cond_1_1 >= new_state[z][x].cond)
            model.addConstr(1 - cond_34 >= new_state[z][x].cond)
            P_S_vars[f'{z}_constant_cond_{x}'] = new_state[z][x].cond
    return new_state, P_S_vars

def create_first_P_S_operation_first_one_constant_cond_padding(model, old_state, operation_name="P_S"):
    'Function create first P S operation first one constant cond padding.'
    new_state = [[Bit(model, bit_type=(0, '*', '*', '*', 0))] +
                 [Bit(model, bit_type=(0, 0, 0, '*', 0)) for _ in range(2)] +[Bit(model, bit_type=(0, '*', '*', '*', 0))]+
                 [Bit(model, bit_type=(0, '*', '*', '*', 0))]
                 for _ in range(slice_number-1)]+[[Bit(model, bit_type=(0, 0, 0, '*', 0)) for i in range(5)]]
    P_S_vars = dict()
    for z in range(slice_number-1):
        cond_1 = model.addVar(vtype=GRB.BINARY, name=f"cond_1_{z}")
        cond_34 = model.addVar(vtype=GRB.BINARY, name=f"cond_34_{z}")


        model.addConstr((1 - cond_34) + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - cond_1) + (1 - new_state[z][0].b) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - cond_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - cond_1) + (1 - new_state[z][0].r) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + (1 - new_state[z][0].b) >= 1)
        model.addConstr(cond_34 + (1 - new_state[z][0].r) + new_state[z][3].r >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr(cond_1 + new_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - old_state[z][0].b) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(cond_1 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - cond_1) + cond_34 + new_state[z][3].r + new_state[z][3].b >= 1)
        model.addConstr(old_state[z][0].r + old_state[z][0].b + (1 - cond_34) >= 1)
        model.addConstr(cond_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(cond_1 + (1 - new_state[z][0].b) + new_state[z][4].b >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr(cond_34 + (1 - new_state[z][0].b) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - old_state[z][0].b) + (1 - new_state[z][0].r) >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (cond_1, cond_34)
        for x in range(5):
            model.addConstr(1 - cond_1 >= new_state[z][x].cond)
            model.addConstr(1 - cond_34 >= new_state[z][x].cond)
            P_S_vars[f'{z}_constant_cond_{x}'] = new_state[z][x].cond
    z = slice_number-1
    P_S_vars[f'{z}_vars'] = (0, 0,0)
    model.addConstr(3 >= new_state[z][0].cond + new_state[z][1].cond + new_state[z][3].cond+ new_state[z][4].cond)
    model.addConstr(3 >= new_state[z][0].cond + new_state[z][1].cond + new_state[z][2].cond + new_state[z][4].cond)
    for x in range(5):
        P_S_vars[f'{z}_constant_cond_{x}'] = new_state[z][x].cond
    return new_state, P_S_vars

def create_first_P_S_operation_first_one_constant_cond_padding_three_stage(model, old_state, operation_name="P_S"):
    'Function create first P S operation first one constant cond padding three stage.'
    new_state = [[Bit(model, bit_type=(0, '*', '*', '*', 0))] +
                 [Bit(model, bit_type=(0, 0, 0, '*', 0)) for _ in range(2)] +[Bit(model, bit_type=(0, '*', '*', '*', 0))]+
                 [Bit(model, bit_type=(0, '*', '*', '*', 0))]
                 for _ in range(slice_number-1)]+[[Bit(model, bit_type=(0, 0, 0, '*', 0)) for i in range(5)]]
    P_S_vars = dict()
    for z in range(slice_number-1):
        cond_1_0 = model.addVar(vtype=GRB.BINARY, name=f"cond_1_0_{z}")
        cond_1_1 = model.addVar(vtype=GRB.BINARY, name=f"cond_1_1_{z}")
        cond_34 = model.addVar(vtype=GRB.BINARY, name=f"cond_34_{z}")


        model.addConstr((1 - cond_34) + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - cond_34) + (1 - new_state[z][3].r) >= 1)
        model.addConstr((1 - cond_1_0) + (1 - new_state[z][4].b) >= 1)
        model.addConstr((1 - cond_1_1) + (1 - new_state[z][0].b) >= 1)
        model.addConstr((1 - cond_1_0) + (1 - new_state[z][4].r) >= 1)
        model.addConstr((1 - cond_1_1) + (1 - new_state[z][0].r) >= 1)
        model.addConstr((1 - old_state[z][0].r) + new_state[z][0].r + new_state[z][4].r >= 1)
        model.addConstr((1 - old_state[z][0].b) + new_state[z][0].b + new_state[z][4].b >= 1)
        model.addConstr(old_state[z][0].r + (1 - new_state[z][3].r) >= 1)
        model.addConstr(old_state[z][0].b + (1 - new_state[z][3].b) >= 1)
        model.addConstr((1 - new_state[z][0].r) + (1 - new_state[z][0].b) >= 1)
        model.addConstr(cond_1_1 + new_state[z][0].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(cond_1_1 + new_state[z][0].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(old_state[z][0].r + old_state[z][0].b + (1 - cond_34) >= 1)
        model.addConstr((1 - cond_1_0) + cond_34 + new_state[z][3].r + new_state[z][3].b >= 1)
        model.addConstr((1 - new_state[z][4].r) + (1 - new_state[z][4].b) >= 1)
        model.addConstr(cond_34 + new_state[z][3].b + (1 - new_state[z][4].b) >= 1)
        model.addConstr(cond_34 + new_state[z][3].r + (1 - new_state[z][4].r) >= 1)
        model.addConstr(cond_1_0 + (1 - new_state[z][0].b) + new_state[z][4].b >= 1)
        model.addConstr(cond_1_0 + (1 - new_state[z][0].r) + new_state[z][4].r >= 1)
        model.addConstr((1 - cond_1_1) + new_state[z][4].r + new_state[z][4].b >= 1)
        new_state[z][1].r = old_state[z][0].r
        new_state[z][1].b = old_state[z][0].b
        P_S_vars[f'{z}_vars'] = (cond_1_0, cond_1_1, cond_34)
        for x in range(5):
            model.addConstr(1 - cond_1_0 >= new_state[z][x].cond)
            model.addConstr(1 - cond_1_1 >= new_state[z][x].cond)
            model.addConstr(1 - cond_34 >= new_state[z][x].cond)
            P_S_vars[f'{z}_constant_cond_{x}'] = new_state[z][x].cond
    z = slice_number-1
    P_S_vars[f'{z}_vars'] = (0, 0,0)
    model.addConstr(3 >= new_state[z][0].cond + new_state[z][1].cond + new_state[z][3].cond+ new_state[z][4].cond)
    model.addConstr(3 >= new_state[z][0].cond + new_state[z][1].cond + new_state[z][2].cond + new_state[z][4].cond)
    for x in range(5):
        P_S_vars[f'{z}_constant_cond_{x}'] = new_state[z][x].cond
    return new_state, P_S_vars
def create_first_P_S_operation_first_one_no_initial(model, operation_name="P_S"):
    'Function create first P S operation first one no initial.'
    initial_state = [[Bit(model, bit_type=(0, '*', '*', 0, 0)) for _ in range(2)] +
                     [Bit(model, bit_type=(0, 0, 0, 0, 0))] +
                     [Bit(model, bit_type=(0, '*',0, 0, 0))]+[Bit(model, bit_type=(0, '*', '*', 0, 0))]
                     for _ in range(slice_number)]
    P_S_vars = dict()
    for z in range(slice_number-1):

        cond_1 = model.addVar(vtype=GRB.BINARY, name=f"cond_1_{z}")
        cond_34 = model.addVar(vtype=GRB.BINARY, name=f"cond_34_{z}")


        model.addConstr((1 - cond_1) + (1 - initial_state[z][0].b) + (1 - initial_state[z][4].b) >= 1)
        model.addConstr((1 - cond_34) + (1 - initial_state[z][3].r) >= 1)
        model.addConstr((1 - cond_1) + (1 - initial_state[z][0].r) + (1 - initial_state[z][4].r) >= 1)
        model.addConstr((1 - initial_state[z][1].r) + (1 - initial_state[z][1].b) >= 1)
        model.addConstr((1 - initial_state[z][0].b) + initial_state[z][1].b >= 1)
        model.addConstr((1 - cond_1) + cond_34 + initial_state[z][3].r >= 1)
        model.addConstr((1 - initial_state[z][0].r) + initial_state[z][1].r >= 1)
        model.addConstr(cond_1 + initial_state[z][0].r + (1 - initial_state[z][4].r) >= 1)
        model.addConstr(cond_1 + initial_state[z][0].b + (1 - initial_state[z][4].b) >= 1)
        model.addConstr(initial_state[z][1].r + (1 - initial_state[z][3].r) >= 1)
        model.addConstr(initial_state[z][1].b + (1 - initial_state[z][4].b) >= 1)
        model.addConstr((1 - cond_1) + initial_state[z][0].r + initial_state[z][4].r + initial_state[z][0].b + initial_state[z][4].b >= 1)
        model.addConstr(cond_1 + (1 - initial_state[z][1].r) + initial_state[z][4].r >= 1)
        model.addConstr(cond_1 + (1 - initial_state[z][1].b) + initial_state[z][4].b >= 1)
        model.addConstr((1 - cond_34) + initial_state[z][1].r + initial_state[z][1].b >= 1)
        model.addConstr((1 - initial_state[z][4].r) + (1 - initial_state[z][1].b) >= 1)
        model.addConstr(cond_34 + (1 - initial_state[z][4].b) >= 1)
        model.addConstr(cond_34 + initial_state[z][3].r + (1 - initial_state[z][4].r) >= 1)

        P_S_vars[f'{z}_vars'] = (cond_1, cond_34)
    P_S_vars[f'{slice_number-1}_vars'] = (0, 0)
    return initial_state, P_S_vars



def last_Hash_collision(model, state, operation_name="P_S"):
    'Function last Hash collision.'
    P_Svars = dict()
    for z in range(slice_number):

        c0 = model.addVar(vtype=GRB.BINARY, name=f"c_0_{operation_name}")
        c1 = model.addVar(vtype=GRB.BINARY, name=f"c_1_{operation_name}")
        c2 = model.addVar(vtype=GRB.BINARY, name=f"c_2_{operation_name}")
        c3 = model.addVar(vtype=GRB.BINARY, name=f"c_3_{operation_name}")
        c4 = model.addVar(vtype=GRB.BINARY, name=f"c_4_{operation_name}")




        model.addConstr((1-state[z][0].w) + (1-state[z][4].w) + (1 - c2) >= 1)
        model.addConstr((1-state[z][1].w) + (1-state[z][3].w) + (1 - c2) >= 1)
        model.addConstr((1-state[z][2].w) + (1-state[z][4].w) + (1 - c2) >= 1)
        model.addConstr((1-state[z][0].w) + (1-state[z][3].w) + (1 - c3) >= 1)
        model.addConstr((1-state[z][1].w) + (1-state[z][4].w) + (1 - c3) >= 1)
        model.addConstr((1-state[z][0].w) + (1-state[z][2].w) + (1-state[z][4].w) + (1 - c3) >= 1)
        model.addConstr((1-state[z][1].w) + (1-state[z][2].w) + (1 - c3) >= 1)
        model.addConstr((1-state[z][3].w) + (1-state[z][4].w) + (1 - c3) >= 1)
        model.addConstr((1-state[z][0].w) + (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w) >= 1)
        model.addConstr((1 - c1) + (1 - c4) >= 1)
        model.addConstr((1-state[z][4].w) + (1 - c4) >= 1)
        model.addConstr((1-state[z][4].w) + (1 - c1) >= 1)
        model.addConstr((1-state[z][3].w) + (1 - c4) >= 1)
        model.addConstr((1-state[z][3].w) + (1 - c1) >= 1)
        model.addConstr((1 - (1-state[z][2].w)) + (1 - (1-state[z][3].w)) + (1 - (1-state[z][4].w)) + (1 - c2) >= 1)
        model.addConstr((1 - (1-state[z][1].w)) + (1 - (1-state[z][3].w)) + (1 - c2) >= 1)
        model.addConstr((1 - c0) + (1 - c3) >= 1)
        model.addConstr((1-state[z][2].w) + (1 - c4) >= 1)
        model.addConstr((1-state[z][2].w) + (1 - c1) >= 1)
        model.addConstr((1 - c0) + (1 - c2) >= 1)
        model.addConstr((1-state[z][0].w) + (1 - c0) >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1 - (1-state[z][2].w)) + (1 - (1-state[z][3].w)) + c0 + c1 + c2 >= 1)
        model.addConstr((1 - (1-state[z][1].w)) + (1 - (1-state[z][2].w)) + (1 - (1-state[z][3].w)) + (1 - (1-state[z][4].w)) + (1 - c0) >= 1)
        model.addConstr((1-state[z][1].w) + (1 - c4) >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1-state[z][3].w) + (1 - (1-state[z][4].w)) + (1 - c2) >= 1)
        model.addConstr((1-state[z][0].w) + (1 - (1-state[z][1].w)) + (1 - (1-state[z][2].w)) + (1 - (1-state[z][4].w)) + c2 + c4 >= 1)
        model.addConstr((1-state[z][1].w) + (1 - c1) >= 1)
        model.addConstr((1-state[z][3].w) + (1-state[z][4].w) + (1 - c0) >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1 - (1-state[z][1].w)) + (1 - (1-state[z][2].w)) + c0 + c1 + c2 >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1 - (1-state[z][3].w)) + (1 - (1-state[z][4].w)) + (1 - c3) >= 1)
        model.addConstr((1-state[z][0].w) + (1-state[z][2].w) + (1 - c2) >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1-state[z][1].w) + (1-state[z][2].w) + (1 - (1-state[z][3].w)) + (1 - (1-state[z][4].w)) + c2 >= 1)
        model.addConstr((1-state[z][1].w) + (1-state[z][4].w) + (1 - c0) >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1 - (1-state[z][1].w)) + (1 - (1-state[z][4].w)) + c0 + c1 + c3 >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1-state[z][1].w) + (1 - (1-state[z][2].w)) + (1-state[z][3].w) + (1 - (1-state[z][4].w)) + c3 >= 1)
        model.addConstr((1-state[z][2].w) + (1-state[z][3].w) + (1 - c0) >= 1)
        model.addConstr((1 - (1-state[z][0].w)) + (1 - (1-state[z][1].w)) + (1-state[z][2].w) + (1 - (1-state[z][3].w)) + (1-state[z][4].w) + c3 >= 1)
        model.addConstr((1-state[z][0].w) + (1 - (1-state[z][1].w)) + (1-state[z][2].w) + (1 - (1-state[z][3].w)) + (1 - (1-state[z][4].w)) + c3 >= 1)
        model.addConstr((1-state[z][0].w) + (1-state[z][1].w) + (1 - (1-state[z][2].w)) + (1 - (1-state[z][3].w)) + (1 - (1-state[z][4].w)) + c3 >= 1)
        model.addConstr((1-state[z][0].w) + (1 - (1-state[z][1].w)) + (1 - (1-state[z][2].w)) + (1 - (1-state[z][3].w)) + (1-state[z][4].w) + c3 >= 1)
        model.addConstr((1 - c3) + (1 - c4) >= 1)

        P_Svars[f'{z}_vars'] = (c0, c1, c2, c3, c4)

    return P_Svars

def new_Hash_collision(model, state, operation_name="P_S"):
    'Function new Hash collision.'
    P_Svars = dict()
    for z in range(slice_number):

        c1 = model.addVar(vtype=GRB.BINARY, name=f"c_1_{operation_name}_{z}")
        c2 = model.addVar(vtype=GRB.BINARY, name=f"c_2_{operation_name}_{z}")
        c3 = model.addVar(vtype=GRB.BINARY, name=f"c_3_{operation_name}_{z}")
        c4 = model.addVar(vtype=GRB.BINARY, name=f"c_4_{operation_name}_{z}")
        c5 = model.addVar(vtype=GRB.BINARY, name=f"c_5_{operation_name}_{z}")
        specital_c4 = model.addVar(vtype=GRB.BINARY, name=f"specital_c4_{operation_name}")




        model.addConstr(c1<=(1-state[z][0].w)+(1-state[z][1].w)+(1-state[z][2].w)+(1-state[z][3].w)+(1-state[z][4].w))
        model.addConstr(2 * c2 <= (1-state[z][0].w) + (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(3 * c3 <= (1-state[z][0].w) + (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(4 * c4 <= (1-state[z][0].w) + (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(5 * c5 <= (1-state[z][0].w) + (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(4 * specital_c4 <= (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(c1+c2+c3+c4+c5+specital_c4<=1)
        P_Svars[f'{z}_vars'] = (c1, c2, c3, c4,c5,specital_c4)

    return P_Svars
def new_simple_Hash_collision(model, state, operation_name="P_S"):
    'Function new simple Hash collision.'
    P_Svars = dict()
    for z in range(slice_number):

        c4 = model.addVar(vtype=GRB.BINARY, name=f"c_4_{operation_name}")
        c5 = model.addVar(vtype=GRB.BINARY, name=f"c_5_{operation_name}")
        specital_c4 = model.addVar(vtype=GRB.BINARY, name=f"specital_c4_{operation_name}")




        model.addConstr(4 * c4 <= (1-state[z][0].w) + (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(5 * c5 <= (1-state[z][0].w) + (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(4 * specital_c4 <= (1-state[z][1].w) + (1-state[z][2].w) + (1-state[z][3].w) + (1-state[z][4].w))
        model.addConstr(c4+c5+specital_c4<=1)
        P_Svars[f'{z}_vars'] = (c4,c5,specital_c4)

    return P_Svars

def _add_P_S_condition_constraints(model, P_S_vars, old_state, temp_state_1, temp_state_2, new_state, operation_name):
    'Function add P S condition constraints.'

    x_old_state1_output = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state_1_z{z}_to_1"),
                            model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state_1_z{z}_to_2")]
                           for z in range(slice_number)]

    x_old_state3_output = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state_3_z{z}_to_3"),
                            model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state_3_z{z}_to_4")]
                           for z in range(slice_number)]

    x_old_state4_output = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state_4_z{z}_to_4"),
                            model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_x_old_state_4_z{z}_to_0")]
                           for z in range(slice_number)]


    for z in range(slice_number):
        model.addConstr(old_state[z][1].cond >= x_old_state1_output[z][0] + x_old_state1_output[z][1])
        model.addConstr(old_state[z][3].cond >= x_old_state3_output[z][0] + x_old_state3_output[z][1])
        model.addConstr(old_state[z][4].cond >= x_old_state4_output[z][0] + x_old_state4_output[z][1])


    for z in range(slice_number):
        delta_r0 = P_S_vars[f"temp1_z{z}_x{0}"]['delta_r']
        delta_b0 = P_S_vars[f"temp1_z{z}_x{0}"]['delta_b']
        has_ul0 = P_S_vars[f"temp1_z{z}_x{0}"]['has_ul']
        P_S_vars[f"temp1_z{z}_x{0}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'temp1_z{z}_x{0}_new_cond')
        model.addConstr(P_S_vars[f"temp1_z{z}_x{0}"]['new_cond'] <= delta_r0 + delta_b0)
        model.addConstr(P_S_vars[f"temp1_z{z}_x{0}"]['new_cond'] <= 1 - has_ul0)
        model.addConstr(temp_state_1[z][0].cond <= old_state[z][0].cond + x_old_state4_output[z][1] + P_S_vars[f"temp1_z{z}_x{0}"]['new_cond'])
        model.addConstr(temp_state_1[z][0].cond >= P_S_vars[f"temp1_z{z}_x{0}"]['new_cond'])

        delta_r2 = P_S_vars[f"temp1_z{z}_x{2}"]['delta_r']
        delta_b2 = P_S_vars[f"temp1_z{z}_x{2}"]['delta_b']
        has_ul2 = P_S_vars[f"temp1_z{z}_x{2}"]['has_ul']
        P_S_vars[f"temp1_z{z}_x{2}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'temp1_z{z}_x{2}_new_cond')
        model.addConstr(P_S_vars[f"temp1_z{z}_x{2}"]['new_cond'] <= delta_r2 + delta_b2)
        model.addConstr(P_S_vars[f"temp1_z{z}_x{2}"]['new_cond'] <= 1 - has_ul2)
        model.addConstr(temp_state_1[z][2].cond <= old_state[z][2].cond + x_old_state1_output[z][1] + P_S_vars[f"temp1_z{z}_x{2}"]['new_cond'])
        model.addConstr(temp_state_1[z][2].cond >= P_S_vars[f"temp1_z{z}_x{2}"]['new_cond'])

        delta_r4 = P_S_vars[f"temp1_z{z}_x{4}"]['delta_r']
        delta_b4 = P_S_vars[f"temp1_z{z}_x{4}"]['delta_b']
        has_ul4 = P_S_vars[f"temp1_z{z}_x{4}"]['has_ul']
        P_S_vars[f"temp1_z{z}_x{4}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'temp1_z{z}_x{4}_new_cond')
        model.addConstr(P_S_vars[f"temp1_z{z}_x{4}"]['new_cond'] <= delta_r4 + delta_b4)
        model.addConstr(P_S_vars[f"temp1_z{z}_x{4}"]['new_cond'] <= 1 - has_ul4)
        model.addConstr(temp_state_1[z][4].cond <= x_old_state4_output[z][0] + x_old_state3_output[z][1] + P_S_vars[f"temp1_z{z}_x{4}"]['new_cond'])
        model.addConstr(temp_state_1[z][4].cond >= P_S_vars[f"temp1_z{z}_x{4}"]['new_cond'])



    temp_state_1_2and_1 = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_1_2and_1_z{z}_x{(x - 1) % 5}")
                            for x in range(5)] for z in range(slice_number)]

    temp_state_1_2and_2 = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_1_2and_2_z{z}_x{(x - 2) % 5}")
                            for x in range(5)] for z in range(slice_number)]

    for z in range(slice_number):
        for x in range(5):
            model.addConstr(temp_state_1[z][x].cond >= (temp_state_1_2and_1[z][x] + temp_state_1_2and_2[z][x]))

    for z in range(slice_number):
        for x in range(5):
            const_cond = P_S_vars[f"and_z{z}_x{x}"]["const_cond"]
            model.addConstr(const_cond == temp_state_1_2and_1[z][(x + 1) % 5] + temp_state_1_2and_2[z][(x + 2) % 5])


    for z in range(slice_number):
        for x in range(5):
            delta_r = P_S_vars[f"temp2_z{z}_x{x}"]['delta_r']
            delta_b = P_S_vars[f"temp2_z{z}_x{x}"]['delta_b']
            has_ul = P_S_vars[f"temp2_z{z}_x{x}"]['has_ul']
            P_S_vars[f"temp2_z{z}_x{x}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'temp2_z{z}_x{0}_new_cond')
            model.addConstr(P_S_vars[f"temp2_z{z}_x{x}"]['new_cond'] <= delta_r + delta_b)
            model.addConstr(P_S_vars[f"temp2_z{z}_x{x}"]['new_cond'] <= 1 - has_ul)
            model.addConstr(temp_state_2[z][x].cond == P_S_vars[f"temp2_z{z}_x{x}"]['new_cond'])


    temp_state_2_0_output = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_2__0_z{z}_to_0"),
                              model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_2__0_z{z}_to_1")]
                             for z in range(slice_number)]

    temp_state_2_2_output = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_2__2_z{z}_to_2"),
                              model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_2__3_z{z}_to_3")]
                             for z in range(slice_number)]

    temp_state_2_4_output = [[model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_2__4_z{z}_to_4"),
                              model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_temp_state_2__4_z{z}_to_0")]
                             for z in range(slice_number)]

    for z in range(slice_number):
        model.addConstr(temp_state_2[z][0].cond >= temp_state_2_0_output[z][0] + temp_state_2_0_output[z][1])
        model.addConstr(temp_state_2[z][2].cond >= temp_state_2_2_output[z][0] + temp_state_2_2_output[z][1])
        model.addConstr(temp_state_2[z][4].cond >= temp_state_2_4_output[z][0] + temp_state_2_4_output[z][1])


    for z in range(slice_number):
        delta_r_0 = P_S_vars[f"new_z{z}_x{0}"]['delta_r']
        delta_b_0 = P_S_vars[f"new_z{z}_x{0}"]['delta_b']
        has_ul_0 = P_S_vars[f"new_z{z}_x{0}"]['has_ul']
        P_S_vars[f"new_z{z}_x{0}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'new_z{z}_x{0}_new_cond')
        model.addConstr(P_S_vars[f"new_z{z}_x{0}"]['new_cond'] <= delta_r_0 + delta_b_0)
        model.addConstr(P_S_vars[f"new_z{z}_x{0}"]['new_cond'] <= 1 - has_ul_0)
        model.addConstr(new_state[z][0].cond <= temp_state_2_0_output[z][0] + temp_state_2_4_output[z][1] + P_S_vars[f"new_z{z}_x{0}"]['new_cond'])
        model.addConstr(new_state[z][0].cond >= P_S_vars[f"new_z{z}_x{0}"]['new_cond'])

        delta_r_1 = P_S_vars[f"new_z{z}_x{1}"]['delta_r']
        delta_b_1 = P_S_vars[f"new_z{z}_x{1}"]['delta_b']
        has_ul_1 = P_S_vars[f"new_z{z}_x{1}"]['has_ul']
        P_S_vars[f"new_z{z}_x{1}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'new_z{z}_x{1}_new_cond')
        model.addConstr(P_S_vars[f"new_z{z}_x{1}"]['new_cond'] <= delta_r_1 + delta_b_1)
        model.addConstr(P_S_vars[f"new_z{z}_x{1}"]['new_cond'] <= 1 - has_ul_1)
        model.addConstr(new_state[z][1].cond <= temp_state_2[z][1].cond + temp_state_2_0_output[z][1] + P_S_vars[f"new_z{z}_x{1}"]['new_cond'])
        model.addConstr(new_state[z][1].cond >= P_S_vars[f"new_z{z}_x{1}"]['new_cond'])


        delta_r_3 = P_S_vars[f"new_z{z}_x{3}"]['delta_r']
        delta_b_3 = P_S_vars[f"new_z{z}_x{3}"]['delta_b']
        has_ul_3 = P_S_vars[f"new_z{z}_x{3}"]['has_ul']
        P_S_vars[f"new_z{z}_x{3}"]['new_cond'] = model.addVar(vtype=GRB.BINARY, name=f'new_z{z}_x{3}_new_cond')
        model.addConstr(P_S_vars[f"new_z{z}_x{3}"]['new_cond'] <= delta_r_3 + delta_b_3)
        model.addConstr(P_S_vars[f"new_z{z}_x{3}"]['new_cond'] <= 1 - has_ul_3)
        model.addConstr(new_state[z][3].cond <= temp_state_2[z][3].cond + temp_state_2_2_output[z][1] + P_S_vars[f"new_z{z}_x{3}"]['new_cond'])
        model.addConstr(new_state[z][3].cond >= P_S_vars[f"new_z{z}_x{3}"]['new_cond'])
