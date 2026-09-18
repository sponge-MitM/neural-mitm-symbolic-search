import gurobipy as gp
from gurobipy import GRB


def _val(X):
    'Function val.'
    if isinstance(X, (int, float)):
        return int(X)
    if type(X) == gp.LinExpr:
        return int(X.getValue())
    elif type(X) == gp.QuadExpr:
        return int(X.getValue())
    elif hasattr(X, 'X'):
        return int(round(X.X))
    else:
        return int(X.x)


def write_Ascon_P(state, slice_number, row, vars, name):
    A = [[0 for z in range(slice_number)] for x in range(5)]
    B = [[None for z in range(slice_number)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()

            temp_cond = _val(state[z][x].cond)
            temp_delta_r = _val(vars[f"new_z{z}_x{x}"]['delta_r'])
            temp_delta_b = _val(vars[f"new_z{z}_x{x}"]['delta_b'])

            if temp_cond == 1 and temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'cond+delta_r+b'
            elif temp_cond == 1 and temp_delta_r > 0.5:
                B[x][z] = 'cond+delta_r'
            elif temp_cond == 1 and temp_delta_b > 0.5:
                B[x][z] = 'cond+delta_b'
            elif temp_cond == 1:
                B[x][z] = 'cond'
            elif temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            else:
                B[x][z] = None

    return A, B, row, name


def write_Ascon_initial(state, slice_number, row, vars, name):
    A = [[0 for z in range(slice_number)] for x in range(5)]
    B = [[None for z in range(slice_number)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()

    return A, B, row, name


def write_Ascon_temp_s1(state, slice_number, row, vars, name):
    A = [[0 for z in range(slice_number)] for x in range(5)]
    B = [[None for z in range(slice_number)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()

            temp_cond = _val(state[z][x].cond)
            temp_delta_r = _val(vars[f"temp1_z{z}_x{x}"]['delta_r'])
            temp_delta_b = _val(vars[f"temp1_z{z}_x{x}"]['delta_b'])

            if temp_cond == 1 and temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'cond+delta_r+b'
            elif temp_cond == 1 and temp_delta_r > 0.5:
                B[x][z] = 'cond+delta_r'
            elif temp_cond == 1 and temp_delta_b > 0.5:
                B[x][z] = 'cond+delta_b'
            elif temp_cond == 1:
                B[x][z] = 'cond'
            elif temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            else:
                B[x][z] = None

    return A, B, row, name


def write_Ascon_temp_s2(state, slice_number, row, vars, name):
    A = [[0 for z in range(slice_number)] for x in range(5)]
    B = [[None for z in range(slice_number)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()

            temp_cond = _val(state[z][x].cond)
            temp_delta_r = _val(vars[f"temp2_z{z}_x{x}"]['delta_r'])
            temp_delta_b = _val(vars[f"temp2_z{z}_x{x}"]['delta_b'])


            and_key = f"and_z{z}_x{x}"
            temp_force_zero = _val(vars[and_key]['force_zero'])
            temp_track_special = _val(vars[and_key]['track_special'])
            temp_const_cond = _val(vars[and_key]['const_cond'])

            if temp_cond == 1 and temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'cond+delta_r+b'
            elif temp_cond == 1 and temp_delta_r > 0.5:
                B[x][z] = 'cond+delta_r_track'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r_track'
            elif temp_cond == 1 and temp_delta_r > 0.5:
                B[x][z] = 'cond+delta_r'
            elif temp_cond == 1 and temp_delta_b > 0.5:
                B[x][z] = 'cond+delta_b'
            elif temp_cond == 1:
                B[x][z] = 'cond'
            elif temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            elif temp_track_special > 0.5:
                B[x][z] = 'track'
            elif temp_force_zero > 0.5:
                B[x][z] = 'force_zero'
            elif temp_const_cond > 0.5:
                B[x][z] = 'const_cond'
            else:
                B[x][z] = None

    return A, B, row, name
