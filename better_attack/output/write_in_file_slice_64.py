import gurobipy as gp


def get_value(X):
    if type(X) == int:
        return X
    elif type(X) == gp.LinExpr:
        return int(X.getValue())
    elif type(X) == gp.QuadExpr:
        return int(X.getValue())
    else:
        return int(X.x)


def write_row(state, row, name):
    A = [[[0 for z in range(64)] for y in range(5)] for x in range(5)]  # Color values: 0-2
    B = [[[0 for z in range(64)] for y in range(5)] for x in range(5)]
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A[x][y][z] = state[z][y][x]._get_type()
                B[x][y][z] = None
    return A, B, row, name


def write_row_chi(state, row, chi_vars, name):
    A = [[[0 for z in range(64)] for y in range(5)] for x in range(5)]  # Color values: 0-2
    B = [[[0 for z in range(64)] for y in range(5)] for x in range(5)]
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A[x][y][z] = state[z][y][x]._get_type()

                temp_delta_r = get_value(chi_vars[f"new_z{z}_y{y}_x{x}"]['delta_r'])
                temp_delta_b = get_value(chi_vars[f"new_z{z}_y{y}_x{x}"]['delta_b'])

                if temp_delta_r > 0.5 and temp_delta_b > 0.5:
                    B[x][y][z] = 'delta_r+b'
                elif temp_delta_r > 0.5:
                    B[x][y][z] = 'delta_r'
                elif temp_delta_b > 0.5:
                    B[x][y][z] = 'delta_b'
                else:
                    B[x][y][z] = None
    return A, B, row, name


def write_row_theta(state, row, theta_vars, name):
    A = [[[0 for z in range(64)] for y in range(5)] for x in range(5)]  # Color values: 0-2
    B = [[[0 for z in range(64)] for y in range(5)] for x in range(5)]
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A[x][y][z] = state[z][y][x]._get_type()

                temp_delta_r = get_value(theta_vars[f"new_z{z}_y{y}_x{x}"]['delta_r'])
                temp_delta_b = get_value(theta_vars[f"new_z{z}_y{y}_x{x}"]['delta_b'])

                if temp_delta_r > 0.5 and temp_delta_b > 0.5:
                    B[x][y][z] = 'delta_r+b'
                elif temp_delta_r > 0.5:
                    B[x][y][z] = 'delta_r'
                elif temp_delta_b > 0.5:
                    B[x][y][z] = 'delta_b'
                else:
                    B[x][y][z] = None

    return A, B, row, name


def write_row_C(state, row, theta_vars, name):
    A = [[0 for z in range(64)] for x in range(5)]  # Color values: 0-2
    B = [[0 for z in range(64)] for x in range(5)]
    for x in range(5):
        for z in range(64):
            A[x][z] = state[z][x]._get_type()

            temp_delta_r = get_value(theta_vars[f"C_x{x}_z{z}"]['delta_r'])
            temp_delta_b = get_value(theta_vars[f"C_x{x}_z{z}"]['delta_b'])

            if temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            else:
                B[x][z] = None

    return A, B, row, name


def write_row_D(state, row, theta_vars, name):
    A = [[0 for z in range(64)] for x in range(5)]  # Color values: 0-2
    B = [[0 for z in range(64)] for x in range(5)]
    for x in range(5):
        for z in range(64):
            A[x][z] = state[z][x]._get_type()

            temp_delta_r = get_value(theta_vars[f"D_x{x}_z{z}"]['delta_r'])
            temp_delta_b = get_value(theta_vars[f"D_x{x}_z{z}"]['delta_b'])

            if temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            else:
                B[x][z] = None
    return A, B, row, name


def write_Ascon_P(state, slice_number, row, vars, name):
    A = [[0 for z in range(64)] for x in range(5)]  # Color values: 0-2
    B = [[None for z in range(64)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()

            temp_delta_r = get_value(vars[f"new_z{z}_x{x}"]['delta_r'])
            temp_delta_b = get_value(vars[f"new_z{z}_x{x}"]['delta_b'])

            if temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            else:
                B[x][z] = None

    return A, B, row, name


def write_Ascon_initial(state, slice_number, row, vars, name):
    A = [[0 for z in range(64)] for x in range(5)]  # Color values: 0-2
    B = [[None for z in range(64)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()

    return A, B, row, name


def write_Ascon_temp_s1(state, slice_number, row, vars, name):
    A = [[0 for z in range(64)] for x in range(5)]  # Color values: 0-2
    B = [[0 for z in range(64)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()

            temp_delta_r = get_value(vars[f"temp1_z{z}_x{x}"]['delta_r'])
            temp_delta_b = get_value(vars[f"temp1_z{z}_x{x}"]['delta_b'])

            if temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            else:
                B[x][z] = None

    return A, B, row, name


def write_Ascon_temp_s2(state, slice_number, row, vars, name):
    A = [[0 for z in range(64)] for x in range(5)]  # Color values: 0-2
    B = [[0 for z in range(64)] for x in range(5)]
    for x in range(5):
        for z in range(slice_number):
            A[x][z] = state[z][x]._get_type()
            temp_delta_r = get_value(vars[f"temp2_z{z}_x{x}"]['delta_r'])
            temp_delta_b = get_value(vars[f"temp2_z{z}_x{x}"]['delta_b'])

            if temp_delta_r > 0.5 and temp_delta_b > 0.5:
                B[x][z] = 'delta_r+b'
            elif temp_delta_r > 0.5:
                B[x][z] = 'delta_r'
            elif temp_delta_b > 0.5:
                B[x][z] = 'delta_b'
            else:
                B[x][z] = None
    return A, B, row, name
