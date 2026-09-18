import gurobipy as gp
from gurobipy import GRB


def _is_bin_const(x):
    'Function is bin const.'
    return isinstance(x, (int, bool)) and int(x) in (0, 1)


def _as_bin_const(x):
    'Function as bin const.'
    if not _is_bin_const(x):
        raise ValueError(f"Expected binary constant 0/1/bool, got: {x}")
    return int(x)


def safe_add_genconstr_or(model: gp.Model, out, inputs, name: str = "safe_or"):
    'Function safe add genconstr or.'
    if inputs is None:
        raise ValueError("inputs cannot be None")

    inputs = list(inputs)
    if len(inputs) == 0:

        if _is_bin_const(out):
            if _as_bin_const(out) != 0:
                raise ValueError(f"{name}: empty OR must be 0, but out={out}")
        else:
            model.addConstr(out == 0, name=f"{name}_empty")
        return 0

    vars_only = []

    for x in inputs:
        if _is_bin_const(x):
            x = _as_bin_const(x)
            if x == 1:

                if _is_bin_const(out):
                    if _as_bin_const(out) != 1:
                        raise ValueError(f"{name}: OR already fixed to 1, but out={out}")
                else:
                    model.addConstr(out == 1, name=f"{name}_fixed_one")
                return 1

        else:
            vars_only.append(x)


    if len(vars_only) == 0:

        if _is_bin_const(out):
            if _as_bin_const(out) != 0:
                raise ValueError(f"{name}: OR already fixed to 0, but out={out}")
        else:
            model.addConstr(out == 0, name=f"{name}_all_zero")
        return 0

    if len(vars_only) == 1:

        v = vars_only[0]
        if _is_bin_const(out):
            model.addConstr(v == _as_bin_const(out), name=f"{name}_single_to_const")
            return _as_bin_const(out)
        else:
            model.addConstr(out == v, name=f"{name}_single")
            return v


    if _is_bin_const(out):
        raise ValueError(f"{name}: out is constant but OR still depends on variables")
    model.addGenConstrOr(out, vars_only, name=f"{name}_or")
    return out


def safe_add_genconstr_and(model: gp.Model, out, inputs, name: str = "safe_and"):
    'Function safe add genconstr and.'
    if inputs is None:
        raise ValueError("inputs cannot be None")

    inputs = list(inputs)
    if len(inputs) == 0:

        if _is_bin_const(out):
            if _as_bin_const(out) != 1:
                raise ValueError(f"{name}: empty AND must be 1, but out={out}")
        else:
            model.addConstr(out == 1, name=f"{name}_empty")
        return 1

    vars_only = []

    for x in inputs:
        if _is_bin_const(x):
            x = _as_bin_const(x)
            if x == 0:

                if _is_bin_const(out):
                    if _as_bin_const(out) != 0:
                        raise ValueError(f"{name}: AND already fixed to 0, but out={out}")
                else:
                    model.addConstr(out == 0, name=f"{name}_fixed_zero")
                return 0

        else:
            vars_only.append(x)


    if len(vars_only) == 0:

        if _is_bin_const(out):
            if _as_bin_const(out) != 1:
                raise ValueError(f"{name}: AND already fixed to 1, but out={out}")
        else:
            model.addConstr(out == 1, name=f"{name}_all_one")
        return 1

    if len(vars_only) == 1:

        v = vars_only[0]
        if _is_bin_const(out):
            model.addConstr(v == _as_bin_const(out), name=f"{name}_single_to_const")
            return _as_bin_const(out)
        else:
            model.addConstr(out == v, name=f"{name}_single")
            return v


    if _is_bin_const(out):
        raise ValueError(f"{name}: out is constant but AND still depends on variables")
    model.addGenConstrAnd(out, vars_only, name=f"{name}_and")
    return out


# =============================================================================
# Bit class definition
# =============================================================================

class Bit:
    'Class Bit.'

    def __init__(self, model: gp.Model, name: str = "", bit_type=''):
        'Function init.'
        self.model = model
        self.name = name
        self._color_flag = None

        if bit_type == '':
            self.ul = model.addVar(vtype=GRB.BINARY, name=f"{name}_ul")
            self.r = model.addVar(vtype=GRB.BINARY, name=f"{name}_r")
            self.b = model.addVar(vtype=GRB.BINARY, name=f"{name}_b")
            self.cond = model.addVar(vtype=GRB.BINARY, name=f"{name}_cond")
            self.w = model.addVar(vtype=GRB.BINARY, name=f"{name}_w")

            self._add_bit_constraints(name)
        else:
            self.init_type(model, bit_type, name)

    def init_type(self, model: gp.Model, bit_type, name: str = ""):
        'Function init type.'
        self.model = model
        self.name = name

        if isinstance(bit_type, str):
            if bit_type == 'gc':
                self.ul, self.r, self.b, self.cond, self.w = 0, 0, 0, 0, 0
            elif bit_type == 'cc':
                self.ul, self.r, self.b, self.cond, self.w = 0, 0, 0, 1, 0
            elif bit_type == 'lr':
                self.ul, self.r, self.b, self.cond, self.w = 0, 1, 0, 0, 0
            elif bit_type == 'ur':
                self.ul, self.r, self.b, self.cond, self.w = 1, 1, 0, 0, 0
            elif bit_type == 'lb':
                self.ul, self.r, self.b, self.cond, self.w = 0, 0, 1, 0, 0
            elif bit_type == 'ub':
                self.ul, self.r, self.b, self.cond, self.w = 1, 0, 1, 0, 0
            elif bit_type == 'lg':
                self.ul, self.r, self.b, self.cond, self.w = 0, 1, 1, 0, 0
            elif bit_type == 'ug':
                self.ul, self.r, self.b, self.cond, self.w = 1, 1, 1, 0, 0
            elif bit_type == 'u':
                self.ul, self.r, self.b, self.cond, self.w = 0, 0, 0, 0, 1
            else:
                raise ValueError(
                    "unsupported bit type: {} (supported: 'gc','cc','lr','ur','lb','ub','lg','ug','u')".format(bit_type)
                )

        elif isinstance(bit_type, tuple):
            if len(bit_type) != 5:
                raise ValueError("tuple bit_type must have length 5")

            # ul
            if bit_type[0] == '*':
                self.ul = model.addVar(vtype=GRB.BINARY, name=f"{name}_ul")
            elif bit_type[0] in [0, 1]:
                self.ul = bit_type[0]
            else:
                raise ValueError("tuple bit_type[0] must be 0/1/'*'")

            # r
            if bit_type[1] == '*':
                self.r = model.addVar(vtype=GRB.BINARY, name=f"{name}_r")
            elif bit_type[1] in [0, 1]:
                self.r = bit_type[1]
            else:
                raise ValueError("tuple bit_type[1] must be 0/1/'*'")

            # b
            if bit_type[2] == '*':
                self.b = model.addVar(vtype=GRB.BINARY, name=f"{name}_b")
            elif bit_type[2] in [0, 1]:
                self.b = bit_type[2]
            else:
                raise ValueError("tuple bit_type[2] must be 0/1/'*'")

            # cond
            if bit_type[3] == '*':
                self.cond = model.addVar(vtype=GRB.BINARY, name=f"{name}_cond")
            elif bit_type[3] in [0, 1]:
                self.cond = bit_type[3]
            else:
                raise ValueError("tuple bit_type[3] must be 0/1/'*'")

            # w
            if bit_type[4] == '*':
                self.w = model.addVar(vtype=GRB.BINARY, name=f"{name}_w")
            elif bit_type[4] in [0, 1]:
                self.w = bit_type[4]
            else:
                raise ValueError("tuple bit_type[4] must be 0/1/'*'")

            self._add_bit_constraints(name)

        else:
            raise ValueError("bit_type must be '' / str / tuple")

    def _add_bit_constraints(self, name: str = ""):
        'Function add bit constraints.'
        self.model.addConstr(self.cond + self.ul <= 1, name=f"{name}_bit_cond_ul")
        self.model.addConstr(self.cond + self.r <= 1, name=f"{name}_bit_cond_r")
        self.model.addConstr(self.cond + self.b <= 1, name=f"{name}_bit_cond_b")
        self.model.addConstr(self.cond + self.w <= 1, name=f"{name}_bit_cond_w")

        self.model.addConstr(self.w + self.ul <= 1, name=f"{name}_bit_w_ul")
        self.model.addConstr(self.w + self.r <= 1, name=f"{name}_bit_w_r")
        self.model.addConstr(self.w + self.b <= 1, name=f"{name}_bit_w_b")

        self.model.addConstr(self.ul <= self.r + self.b, name=f"{name}_bit_ul_need_color")

    def get_color_flag(self):
        'Function get color flag.'
        if self._color_flag is not None:
            return self._color_flag


        r_is_const = isinstance(self.r, int)
        b_is_const = isinstance(self.b, int)

        if r_is_const and b_is_const:
            self._color_flag = 1 if (self.r or self.b) else 0
            return self._color_flag
        if r_is_const:
            if self.r == 1:
                self._color_flag = 1
                return 1
            else:  # r=0
                self._color_flag = self.b
                return self.b
        if b_is_const:
            if self.b == 1:
                self._color_flag = 1
                return 1
            else:  # b=0
                self._color_flag = self.r
                return self.r


        cf = self.model.addVar(vtype=GRB.BINARY, name=f"{self.name}_has_color")
        safe_add_genconstr_or(self.model, cf, [self.r, self.b], name=f"{self.name}_color_or")
        self._color_flag = cf
        return cf

    def _get_type(self) -> str:
        'Function get type.'
        ul = int(self.ul) if isinstance(self.ul, int) else int(round(self.ul.X))
        r = int(self.r) if isinstance(self.r, int) else int(round(self.r.X))
        b = int(self.b) if isinstance(self.b, int) else int(round(self.b.X))
        cond = int(self.cond) if isinstance(self.cond, int) else int(round(self.cond.X))
        w = int(self.w) if isinstance(self.w, int) else int(round(self.w.X))

        if w == 1:
            return 'u'
        if cond == 1:
            return 'cc'
        if ul == 0 and r == 0 and b == 0:
            return 'gc'
        if ul == 0 and r == 1 and b == 0:
            return 'lr'
        if ul == 1 and r == 1 and b == 0:
            return 'ur'
        if ul == 0 and r == 0 and b == 1:
            return 'lb'
        if ul == 1 and r == 0 and b == 1:
            return 'ub'
        if ul == 0 and r == 1 and b == 1:
            return 'lg'
        if ul == 1 and r == 1 and b == 1:
            return 'ug'
        return 'invalid'


# =============================================================================
# 1) XOR with red cancellation only
# =============================================================================

def add_xor_red_cancel_only(model: gp.Model, in_bits, out_bit: Bit, delta_r=None, name: str = "xor_r"):
    'Function add xor red cancel only.'
    d = len(in_bits)
    if d <= 0:
        raise ValueError("in_bits must be non-empty")

    if delta_r is None:
        delta_r = model.addVar(vtype=GRB.BINARY, name=f"{name}_delta_r")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    safe_add_genconstr_or(model, out_bit.w, [x.w for x in in_bits], name=f"{name}_w_or")


    model.addConstr(delta_r <= 1 - out_bit.w, name=f"{name}_delta_r_no_white")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    any_nonlinear = model.addVar(vtype=GRB.BINARY, name=f"{name}_any_nonlinear")
    safe_add_genconstr_or(model, any_nonlinear, [x.ul for x in in_bits], name=f"{name}_any_nonlinear_or")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    S_r = gp.quicksum(bit.r for bit in in_bits)
    S_b = gp.quicksum(bit.b for bit in in_bits)

    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    model.addConstr(out_bit.b <= S_b, name=f"{name}_b_or_ub")
    model.addConstr(S_b <= d * (out_bit.b + out_bit.w), name=f"{name}_b_or_lb")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(out_bit.r + delta_r <= 1, name=f"{name}_r_delta_r_exclusive")
    model.addConstr(out_bit.r <= S_r, name=f"{name}_r_survive_ub")
    model.addConstr(2*delta_r <= S_r, name=f"{name}_r_cancel_need_red")
    for bit in in_bits:
        model.addConstr(bit.r <= out_bit.r + delta_r + out_bit.w, name=f"{name}_r_cover")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(out_bit.ul <= any_nonlinear, name=f"{name}_ul_need_N")
    model.addConstr(out_bit.ul <= out_bit.b + 1 - delta_r, name=f"{name}_ul_need_blue_or_no_delta_r")
    model.addConstr(out_bit.ul >= any_nonlinear - delta_r - out_bit.w, name=f"{name}_ul_lb1")
    model.addConstr(out_bit.ul >= any_nonlinear + out_bit.b + delta_r - 2 - out_bit.w, name=f"{name}_ul_lb2")


    model.addConstr(out_bit.ul <= out_bit.r + out_bit.b, name=f"{name}_ul_color_tight")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    if not isinstance(out_bit.cond, int):
        for i, bit in enumerate(in_bits):
            model.addConstr(out_bit.cond <= 1 - bit.ul, name=f"{name}_cond_no_ul_{i}")
            model.addConstr(out_bit.cond <= 1 - bit.w, name=f"{name}_cond_no_w_{i}")

    return {
        'delta_r': delta_r,
        'delta_b': 0,
        'has_ul':any_nonlinear,
        'new_cond': 0
    }


# =============================================================================
# 2) XOR with both red and blue cancellation
# =============================================================================

def add_xor_general_rb_cancel(model: gp.Model, in_bits, out_bit: Bit, delta_r=None, delta_b=None, name: str = "xor_full"):
    'Function add xor general rb cancel.'
    d = len(in_bits)
    if d <= 0:
        raise ValueError("in_bits must be non-empty")

    if delta_r is None:
        delta_r = model.addVar(vtype=GRB.BINARY, name=f"{name}_delta_r")
    if delta_b is None:
        delta_b = model.addVar(vtype=GRB.BINARY, name=f"{name}_delta_b")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    safe_add_genconstr_or(model, out_bit.w, [x.w for x in in_bits], name=f"{name}_w_or")

    model.addConstr(delta_r <= 1 - out_bit.w, name=f"{name}_delta_r_no_white")
    model.addConstr(delta_b <= 1 - out_bit.w, name=f"{name}_delta_b_no_white")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    any_nonlinear = model.addVar(vtype=GRB.BINARY, name=f"{name}_any_nonlinear")
    safe_add_genconstr_or(model, any_nonlinear, [x.ul for x in in_bits], name=f"{name}_any_nonlinear_or")

    S_r = gp.quicksum(bit.r for bit in in_bits)
    S_b = gp.quicksum(bit.b for bit in in_bits)

    # -------------------------------------------------------------------------
    # red: survive or cancel
    # -------------------------------------------------------------------------
    model.addConstr(out_bit.r + delta_r <= 1, name=f"{name}_r_delta_r_exclusive")
    model.addConstr(out_bit.r <= S_r, name=f"{name}_r_survive_ub")
    model.addConstr(2*delta_r <= S_r, name=f"{name}_r_cancel_need_red")
    for bit in in_bits:
        model.addConstr(bit.r <= out_bit.r + delta_r + out_bit.w, name=f"{name}_r_cover")

    # -------------------------------------------------------------------------
    # blue: survive or cancel
    # -------------------------------------------------------------------------
    model.addConstr(out_bit.b + delta_b <= 1, name=f"{name}_b_delta_b_exclusive")
    model.addConstr(out_bit.b <= S_b, name=f"{name}_b_survive_ub")
    model.addConstr(2 * delta_b <= S_b, name=f"{name}_b_cancel_need_blue")
    for bit in in_bits:
        model.addConstr(bit.b <= out_bit.b + delta_b + out_bit.w, name=f"{name}_b_cover")


    model.addConstr(delta_b <= 1 - any_nonlinear, name=f"{name}_delta_b_linear_only")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(out_bit.ul <= any_nonlinear, name=f"{name}_ul_need_N")
    model.addConstr(out_bit.ul <= out_bit.b + 1 - delta_r, name=f"{name}_ul_need_blue_or_no_delta_r")
    model.addConstr(out_bit.ul >= any_nonlinear - delta_r - out_bit.w, name=f"{name}_ul_lb1")
    model.addConstr(out_bit.ul >= any_nonlinear + out_bit.b + delta_r - 2 - out_bit.w, name=f"{name}_ul_lb2")
    model.addConstr(out_bit.ul <= out_bit.r + out_bit.b, name=f"{name}_ul_color_tight")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    if not isinstance(out_bit.cond, int):
        for i, bit in enumerate(in_bits):
            model.addConstr(out_bit.cond <= 1 - bit.ul, name=f"{name}_cond_no_ul_{i}")
            model.addConstr(out_bit.cond <= 1 - bit.w, name=f"{name}_cond_no_w_{i}")

    return {
        'delta_r': delta_r,
        'delta_b': delta_b,
        'has_ul':any_nonlinear,
        'new_cond': 0
    }


# =============================================================================
# 3) XOR with both red and blue cancellation, all inputs linear
# =============================================================================

def add_xor_linear_rb_cancel(model: gp.Model, in_bits, out_bit: Bit, delta_r=None, delta_b=None, name: str = "xor_lin_full"):
    'Function add xor linear rb cancel.'
    d = len(in_bits)
    if d <= 0:
        raise ValueError("in_bits must be non-empty")

    if delta_r is None:
        delta_r = model.addVar(vtype=GRB.BINARY, name=f"{name}_delta_r")
    if delta_b is None:
        delta_b = model.addVar(vtype=GRB.BINARY, name=f"{name}_delta_b")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    for i, bit in enumerate(in_bits):
        model.addConstr(bit.ul == 0, name=f"{name}_in_linear_{i}")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    safe_add_genconstr_or(model, out_bit.w, [x.w for x in in_bits], name=f"{name}_w_or")


    model.addConstr(delta_r <= 1 - out_bit.w, name=f"{name}_delta_r_no_white")
    model.addConstr(delta_b <= 1 - out_bit.w, name=f"{name}_delta_b_no_white")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(out_bit.ul == 0, name=f"{name}_out_linear")

    S_r = gp.quicksum(bit.r for bit in in_bits)
    S_b = gp.quicksum(bit.b for bit in in_bits)

    # -------------------------------------------------------------------------
    # red: survive or cancel
    # -------------------------------------------------------------------------
    model.addConstr(out_bit.r + delta_r <= 1, name=f"{name}_r_delta_r_exclusive")
    model.addConstr(out_bit.r <= S_r, name=f"{name}_r_survive_ub")
    model.addConstr(2*delta_r <= S_r, name=f"{name}_r_cancel_need_red")
    for bit in in_bits:
        model.addConstr(bit.r <= out_bit.r + delta_r + out_bit.w, name=f"{name}_r_cover")

    # -------------------------------------------------------------------------
    # blue: survive or cancel
    # -------------------------------------------------------------------------
    model.addConstr(out_bit.b + delta_b <= 1, name=f"{name}_b_delta_b_exclusive")
    model.addConstr(out_bit.b <= S_b, name=f"{name}_b_survive_ub")
    model.addConstr(2*delta_b <= S_b, name=f"{name}_b_cancel_need_blue")
    for bit in in_bits:
        model.addConstr(bit.b <= out_bit.b + delta_b + out_bit.w, name=f"{name}_b_cover")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    if not isinstance(out_bit.cond, int):
        for i, bit in enumerate(in_bits):
            model.addConstr(out_bit.cond <= 1 - bit.w, name=f"{name}_cond_no_w_{i}")

    return {
        'delta_r': delta_r,
        'delta_b': delta_b,
        'has_ul':0,
        'new_cond': 0
    }



# =============================================================================
# 5) AND without conditional constants
# =============================================================================

def add_and_no_cond(model: gp.Model, x: Bit, y: Bit, out_bit: Bit,
                    force_zero=None, track_special=None, name: str = "and_no_cond"):
    'Function add and no cond.'
    if force_zero is None:
        force_zero = 0
    if track_special is None:
        track_special = 0

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(x.cond == 0, name=f"{name}_x_cond_zero")
    model.addConstr(y.cond == 0, name=f"{name}_y_cond_zero")
    model.addConstr(out_bit.cond == 0, name=f"{name}_out_cond_zero")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    x_color = x.get_color_flag()
    y_color = y.get_color_flag()

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def _and_op(a, b, suffix):
        if _is_bin_const(a) and _is_bin_const(b):
            return int(a) & int(b)
        if _is_bin_const(a):
            return 0 if int(a) == 0 else b
        if _is_bin_const(b):
            return 0 if int(b) == 0 else a

        v = model.addVar(vtype=GRB.BINARY, name=f"{name}_{suffix}")
        safe_add_genconstr_and(model, v, [a, b], name=f"{name}_{suffix}")
        return v

    def _or_op(a, b, suffix):
        if _is_bin_const(a) and _is_bin_const(b):
            return int(a) | int(b)
        if _is_bin_const(a):
            return 1 if int(a) == 1 else b
        if _is_bin_const(b):
            return 1 if int(b) == 1 else a

        v = model.addVar(vtype=GRB.BINARY, name=f"{name}_{suffix}")
        safe_add_genconstr_or(model, v, [a, b], name=f"{name}_{suffix}")
        return v

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    and1 = _and_op(x.r, y.b, "and_xr_yb")
    and2 = _and_op(x.b, y.r, "and_xb_yr")
    mixed_rb = _or_op(and1, and2, "mixed_rb")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    color_and = _and_op(x_color, y_color, "color_and")
    not_mixed = model.addVar(vtype=GRB.BINARY, name=f"{name}_not_mixed")
    model.addConstr(not_mixed == 1 - mixed_rb, name=f"{name}_not_mixed")
    pure_same_color = _and_op(color_and, not_mixed, "pure_same_color")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(force_zero + track_special <= mixed_rb, name=f"{name}_ZT_only_on_mixed")
    model.addConstr(track_special <= 2 - x.r - x.b)
    model.addConstr(track_special <= 2 - y.r - y.b)

    model.addConstr(3 - (force_zero + track_special) >= x.ul + x.r + x.b, name=f"{name}_ZT_not_on_ug")
    model.addConstr(3 - (force_zero + track_special) >= y.ul + y.r + y.b, name=f"{name}_ZT_not_on_ug")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(out_bit.w >= x.w, name=f"{name}_w_from_x")
    model.addConstr(out_bit.w >= y.w, name=f"{name}_w_from_y")
    model.addConstr(out_bit.w >= mixed_rb - force_zero - track_special, name=f"{name}_w_from_mixed")

    model.addConstr(out_bit.w <= x.w + y.w + mixed_rb, name=f"{name}_w_ub_sum")
    model.addConstr(out_bit.w <= 1 - force_zero, name=f"{name}_w_no_Z")
    model.addConstr(out_bit.w <= 1 - track_special, name=f"{name}_w_no_T")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(out_bit.r >= x.r - out_bit.w - force_zero, name=f"{name}_r_lb_x")
    model.addConstr(out_bit.r >= y.r - out_bit.w - force_zero, name=f"{name}_r_lb_y")
    model.addConstr(out_bit.r >= track_special, name=f"{name}_r_lb_T")
    model.addConstr(out_bit.r <= track_special + x.r + y.r, name=f"{name}_r_ub_sum")
    model.addConstr(out_bit.r <= track_special + 1 - out_bit.w - force_zero, name=f"{name}_r_ub_clean")

    model.addConstr(out_bit.b >= x.b - out_bit.w - force_zero, name=f"{name}_b_lb_x")
    model.addConstr(out_bit.b >= y.b - out_bit.w - force_zero, name=f"{name}_b_lb_y")
    model.addConstr(out_bit.b >= track_special, name=f"{name}_b_lb_T")
    model.addConstr(out_bit.b <= track_special + x.b + y.b, name=f"{name}_b_ub_sum")
    model.addConstr(out_bit.b <= track_special + 1 - out_bit.w - force_zero, name=f"{name}_b_ub_clean")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    model.addConstr(out_bit.ul >= x.ul - out_bit.w - force_zero, name=f"{name}_ul_lb_x")
    model.addConstr(out_bit.ul >= y.ul - out_bit.w - force_zero, name=f"{name}_ul_lb_y")
    model.addConstr(out_bit.ul >= pure_same_color - out_bit.w - force_zero, name=f"{name}_ul_lb_p")
    model.addConstr(out_bit.ul >= track_special, name=f"{name}_ul_lb_T")

    model.addConstr(out_bit.ul <= track_special + x.ul + y.ul + pure_same_color, name=f"{name}_ul_ub_sum")
    model.addConstr(out_bit.ul <= track_special + 1 - out_bit.w - force_zero, name=f"{name}_ul_ub_clean")


    model.addConstr(out_bit.ul <= out_bit.r + out_bit.b, name=f"{name}_ul_color_tight")

    return {'force_zero':force_zero,
            'track_special':track_special,
            'const_cond': 0,
            }
