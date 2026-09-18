"""
operation_MILP.py - fixed version of operation_MILP
========================================================
The only change relative to base_MILP/operation_MILP.py: eliminate the repeated
"is this bit of type u" check for every bit.

Background
----------
The original `xor_with_ul_input_no_delta_b` and `and_operation` call
`_add_pure_u_indicator` for every input bit (producing an is_u indicator:
1 variable + 4 constraints) and create a has_ul_b indicator
(1 variable + 3 constraints).

In the Ascon/Keccak round functions, the same Bit object is referenced
repeatedly by multiple operations:
  * P_L: each old-state bit is used by 3 XORs (z, z-off1, z-off2);
  * P_S: temp_state_1 bits are used by 2 and + 1 xor;
  * P_S: new_state[z][2]/[z][4] directly reference temp_state_2 bit objects,
         which are reused again in the next round;
  * direct reference copies (e.g. temp_state_1[z][1] = old_state[z][1]) make the
    same object referenced across multiple rounds.

Thus the "is pure u (ul=1,r=0,b=0)" check for the same Bit is created repeatedly,
each time adding 1 variable + 4 constraints (redundant constraints do not change
the feasible region, but they significantly inflate the model size).

Fix
---
Cache the indicator variables on the Bit instance:
  * `_pure_u_var`   : indicator for whether this bit is pure u (a variable or
                      constant 0/1), created on first check and reused by all
                      subsequent operations;
  * `_has_ul_b_var` : indicator for whether this bit has ul=1 and b=1 (a variable
                      or constant 0/1), likewise created only once.

For a bit whose ul/r/b are all constant, fold directly to the fixed 0/1 without
creating any variable/constraint.
The new and old models have exactly the same feasible region over the shared
variables (only duplicate definitions and uniquely-valued redundant variables and
their constraints are removed), so all attack scripts keep the same semantics;
only the model is smaller and builds faster.

Note: the cache is stored per Bit instance; every model rebuilds all Bits, so there
is no cross-model contamination. `init_type` resets the cache to ensure indicators
do not go stale after flags are overwritten.
"""

import gurobipy as gp
from gurobipy import GRB


def get_value(X):
    """Get integer value from variable, linear expression, or constant."""
    if type(X) == int:
        return X
    if type(X) == gp.LinExpr:
        return int(X.getValue())
    return int(X.x)


class Bit:
    """
    Bit class representing one bit in the hash function.

    Each bit now has only three flags:
    - ul: nonlinear flag
    - r:  contains red flag
    - b:  contains blue flag

    (New) cached fields:
    - _pure_u_var   : indicator variable for whether this bit is pure u (ul=1,r=0,b=0), built once per Bit;
    - _has_ul_b_var : indicator variable for whether this bit has ul=1 and b=1, built once per Bit.
    """

    def __init__(self, model: gp.Model, name_prefix="", bit_type=''):
        """
        Initialize bit variables.

        Parameters:
        - model: Gurobi model object
        - name_prefix: variable name prefix
        - bit_type: supports 'lr','ur','lb','ub','lg','ug','uc','c','u' or a 3-tuple.
        """
        self.model = model
        self._pure_u_var = None
        self._has_ul_b_var = None
        if bit_type == '':
            self.ul = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_ul")
            self.r = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_r")
            self.b = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_b")
        else:
            self.init_type(model, bit_type, name_prefix)

    def init_type(self, model, bit_type, name_prefix):
        """Initialize flags based on bit type."""
        self.model = model
        # Flags are (re)assigned, so the previous indicator cache is invalidated and must be reset
        self._pure_u_var = None
        self._has_ul_b_var = None
        if type(bit_type) == str:
            if bit_type == 'lr':      # linear red bit
                self.ul, self.r, self.b = 0, 1, 0
            elif bit_type == 'ur':    # nonlinear red bit
                self.ul, self.r, self.b = 1, 1, 0
            elif bit_type == 'lb':    # linear blue bit
                self.ul, self.r, self.b = 0, 0, 1
            elif bit_type == 'ub':    # nonlinear blue bit
                self.ul, self.r, self.b = 1, 0, 1
            elif bit_type == 'lg':    # linear red-blue combination
                self.ul, self.r, self.b = 0, 1, 1
            elif bit_type in ('uc', 'c'):  # constant
                self.ul, self.r, self.b = 0, 0, 0
            elif bit_type == 'u':     # pure nonlinear bit
                self.ul, self.r, self.b = 1, 0, 0
            elif bit_type == 'ug':    # nonlinear red-blue combination
                self.ul, self.r, self.b = 1, 1, 1
            else:
                raise ValueError(
                    "Unsupported bit type: "
                    f"{bit_type} (only 'lr','ur','lb','ub','lg','ug','uc','c','u')"
                )
        elif type(bit_type) == tuple:
            if len(bit_type) != 3:
                raise ValueError("Bit tuple type must contain exactly three entries: (ul, r, b)")
            self.ul = self._init_flag(model, bit_type[0], f"{name_prefix}_ul")
            self.r = self._init_flag(model, bit_type[1], f"{name_prefix}_r")
            self.b = self._init_flag(model, bit_type[2], f"{name_prefix}_b")
        else:
            raise ValueError("bit_type must be a string or a 3-tuple")

    @staticmethod
    def _init_flag(model, flag, name):
        if flag == '*':
            return model.addVar(vtype=GRB.BINARY, name=name)
        if flag in (0, 1):
            return flag
        raise ValueError("Bit tuple entries must be 0, 1, or '*'")

    def _get_type(self) -> str:
        """Return bit type based on flag variable values."""
        ul = get_value(self.ul)
        r = get_value(self.r)
        b = get_value(self.b)

        if ul == 1 and r == 0 and b == 0:
            return 'u'
        elif ul == 0 and r == 1 and b == 0:
            return 'lr'
        elif ul == 1 and r == 1 and b == 0:
            return 'ur'
        elif ul == 0 and r == 0 and b == 1:
            return 'lb'
        elif ul == 1 and r == 0 and b == 1:
            return 'ub'
        elif ul == 0 and r == 1 and b == 1:
            return 'lg'
        elif ul == 1 and r == 1 and b == 1:
            return 'ug'
        else:
            return 'c'


def add_or(model, z, xs, name=""):
    """z = OR(xs), where each element of xs may be a {0,1} variable or a constant.

    Constant folding:
      * any item is constant 1 -> OR=1, fix z's bounds directly, zero constraints;
      * all constant           -> OR=0/1, fix z's bounds directly, zero constraints;
      * only 1 variable left   -> z == that variable, 1 constraint;
      * general case           -> minimal encoding: z >= x_i (1 per variable) + z <= sum(xs) (1).
    """
    consts = [x for x in xs if isinstance(x, int)]
    vars_ = [x for x in xs if not isinstance(x, int)]
    if any(x == 1 for x in consts):
        z.lb = z.ub = 1
        return
    if not vars_:
        z.lb = z.ub = 0
        return
    if len(vars_) == 1:
        model.addConstr(z == vars_[0], name=f"{name}_or_eq")
        return
    for x in vars_:
        model.addConstr(z >= x, name=f"{name}_or_lb")
    model.addConstr(z <= gp.quicksum(xs), name=f"{name}_or_ub")


def _add_color_or(model, output, bit, name):
    model.addConstr(output >= bit.r, name=f"{name}_r_lb")
    model.addConstr(output >= bit.b, name=f"{name}_b_lb")
    model.addConstr(output <= bit.r + bit.b, name=f"{name}_ub")


def _add_pure_u_indicator(model, output, bit, name):
    """output indicates whether bit is pure nonlinear u=(ul=1,r=0,b=0)."""
    model.addConstr(output >= bit.ul - bit.r - bit.b, name=f"{name}_lb")
    model.addConstr(output <= bit.ul, name=f"{name}_ul_ub")
    model.addConstr(output <= 1 - bit.r, name=f"{name}_r_ub")
    model.addConstr(output <= 1 - bit.b, name=f"{name}_b_ub")


def is_const_zero_bit(bit):
    """Whether bit is the constant (0,0,0) - ul/r/b are all constant 0.

    Public version (for use by the various MILP modules via `from ... import *`;
    the underscore-prefixed private name is not imported by wildcard).
    """
    return (isinstance(bit.ul, int) and bit.ul == 0 and
            isinstance(bit.r, int) and bit.r == 0 and
            isinstance(bit.b, int) and bit.b == 0)


def _is_const_zero(bit):
    """Whether bit is the constant (0,0,0) - ul/r/b are all constant 0.

    Constant zero is the identity element of XOR and the zero input of AND; it is
    recognized in order to:
      * filter identity-element inputs in XOR (their contribution to every
        constraint is 0, so filtering leaves the feasible region unchanged);
      * fold the output to a constant when all inputs are zero (the output is
        uniquely determined to be (0,0,0)).
    """
    return is_const_zero_bit(bit)


def _add_create_u_constraints(model, create_u, input1, input2, has_c1, has_c2, operation_name):
    # create_u is generated by a red-blue cross product.
    model.addConstr(create_u >= input1.r + input2.b - 1, name=f"{operation_name}_create_u_r1b2_lb")
    model.addConstr(create_u >= input2.r + input1.b - 1, name=f"{operation_name}_create_u_r2b1_lb")
    model.addConstr(create_u <= input1.r + input2.r, name=f"{operation_name}_create_u_r_ub")
    model.addConstr(create_u <= input1.b + input2.b, name=f"{operation_name}_create_u_b_ub")
    model.addConstr(create_u <= has_c1, name=f"{operation_name}_create_u_c1_ub")
    model.addConstr(create_u <= has_c2, name=f"{operation_name}_create_u_c2_ub")


def _get_pure_u_indicator(model, bit, name):
    """Return the indicator variable (or constant 0/1) for whether bit is pure u (ul=1, r=0, b=0).

    Each Bit instance creates this once and caches it; all later operations reuse
    it directly, eliminating the redundant variables and constraints caused by the
    repeated "is this bit of type u" check.
    For a bit whose ul/r/b are all constant, fold directly to the fixed value
    without creating any variable/constraint.
    A bit with constant ul=0 (e.g. type (0,*,*)) can never be pure u, so it is
    likewise folded directly to 0.
    """
    cached = getattr(bit, '_pure_u_var', None)
    if cached is not None:
        return cached
    if isinstance(bit.ul, int) and isinstance(bit.r, int) and isinstance(bit.b, int):
        value = 1 if (bit.ul == 1 and bit.r == 0 and bit.b == 0) else 0
        bit._pure_u_var = value
        return value
    if isinstance(bit.ul, int) and bit.ul == 0:
        # (0,*,*) bit: ul=0 always holds, so the pure-u indicator is always 0
        bit._pure_u_var = 0
        return 0
    output = model.addVar(vtype=GRB.BINARY, name=f"{name}_is_u")
    _add_pure_u_indicator(model, output, bit, name)
    bit._pure_u_var = output
    return output


def get_pure_u_indicator(model, bit, name):
    """Public wrapper around `_get_pure_u_indicator` (the underscore-prefixed
    private name is not imported by `import *`).

    Used when an attack script reuses the already-cached pure-u indicator of some
    bit in the main program (e.g. the no-confusion reward = 1-p, where p was
    cached by XOR inputs such as sum_of_a0a2a4).
    """
    return _get_pure_u_indicator(model, bit, name)


def _get_has_ul_b_indicator(model, bit, name):
    """Return the indicator variable (or constant 0/1) for whether bit satisfies ul=1 and b=1.

    Each Bit instance creates this once and caches it, same as _get_pure_u_indicator.
    Note: this indicator depends only on ul and b, not on r.
    """
    cached = getattr(bit, '_has_ul_b_var', None)
    if cached is not None:
        return cached
    if isinstance(bit.ul, int) and isinstance(bit.b, int):
        value = 1 if (bit.ul == 1 and bit.b == 1) else 0
        bit._has_ul_b_var = value
        return value
    if isinstance(bit.ul, int) and bit.ul == 0:
        # (0,*,*) bit: ul=0 always holds, so the ul and b indicator is always 0
        bit._has_ul_b_var = 0
        return 0
    temp = model.addVar(vtype=GRB.BINARY, name=f"{name}_has_ul_b")
    model.addConstr(temp >= bit.ul + bit.b - 1)
    model.addConstr(temp <= bit.ul)
    model.addConstr(temp <= bit.b)
    bit._has_ul_b_var = temp
    return temp


def _bits_equivalent(a, b):
    """Whether two Bits are equivalent at the variable level (each of ul/r/b is the
    same variable or the same constant).

    Used for self-operation classification: bits sharing the same set of variables
    (directly copied reference bits, or shared-variable bits produced by future
    folding/aliasing) are equivalent in the model to bits of the same object and
    should likewise trigger the self-AND simplification.
    The check is conservative: it returns True only when every flag is is-equal
    bit-by-bit (or the same constant); when different variables are forced equal by
    constraints, it returns False and falls back to the full encoding (which does
    not change the feasible region).

    Note: do not use equivalence folding for XOR - under differential propagation
    semantics the ul/r of XOR(single input) may be cancelled depending on delta_r,
    so it is not an identity mapping (confirmed by exhaustive verification).
    """
    for attr in ('ul', 'r', 'b'):
        va = getattr(a, attr)
        vb = getattr(b, attr)
        if isinstance(va, int) and isinstance(vb, int):
            if va != vb:
                return False
        elif isinstance(va, int) or isinstance(vb, int):
            # one constant and one variable: never equivalent (constant 0 cannot be is-equal to a variable)
            return False
        elif va is not vb:
            return False
    return True


def _and_operation_self(model, x, output, operation_name=''):
    """AND(x, x): simplified self-operation encoding when both inputs are the same Bit object.

    Merge and deduplicate the constraints of the original `and_operation` under
    r1=r2,b1=b2,u1=u2 (same object) term by term (exhaustively verified over all
    8 input types x 8 output values, 0 mismatches):
      * create_u = (r and b) or (r and b) = r and b, 6 constraints shrink to 3 (c>=r+b-1, c<=r, c<=b);
      * the 4 bicolor lower bounds of output.ul shrink to >=u, >=r, >=b, and the 2
        source upper bounds merge into output.ul <= u+r+b (equivalent to the
        original 2 upper bounds over the binary domain);
      * the 2 p-suppression constraints of output.r/b merge into 1 (p1==p2==p,
        cache shared), and the 2 activation lower bounds merge into 1
        (out >= x - 2p - c).
    Total: 1 auxiliary variable (create_u) + 15 constraints + 1 pure-u indicator
    (cache reused); the original required 1 variable + 26 constraints + 2 pure-u
    indicators.
    """
    # create_u = r and b
    c = model.addVar(vtype=GRB.BINARY, name=f'{operation_name}_create_u')
    model.addConstr(c >= x.r + x.b - 1, name=f"{operation_name}_create_u_lb")
    model.addConstr(c <= x.r, name=f"{operation_name}_create_u_r_ub")
    model.addConstr(c <= x.b, name=f"{operation_name}_create_u_b_ub")

    # output.ul = u  or  r  or  b
    model.addConstr(output.ul >= x.ul, name=f"{operation_name}_ul_u_lb")
    model.addConstr(output.ul >= x.r, name=f"{operation_name}_ul_r_lb")
    model.addConstr(output.ul >= x.b, name=f"{operation_name}_ul_b_lb")
    model.addConstr(output.ul <= x.ul + x.r + x.b, name=f"{operation_name}_ul_ub")

    # output.r / output.b (p is the cached pure-u indicator, built once per object)
    p = _get_pure_u_indicator(model, x, f"{operation_name}_input_0")
    model.addConstr(output.r <= x.r, name=f"{operation_name}_r_act")
    model.addConstr(output.r <= 1 - p, name=f"{operation_name}_r_p_ub")
    model.addConstr(output.r <= 1 - c, name=f"{operation_name}_r_c_ub")
    # Coefficient tightening (B1): p=1  x.r=0, the 2p term is always 0, so drop it (integer-equivalent, tighter LP)
    model.addConstr(output.r >= x.r - c, name=f"{operation_name}_r_lb")
    model.addConstr(output.b <= x.b, name=f"{operation_name}_b_act")
    model.addConstr(output.b <= 1 - p, name=f"{operation_name}_b_p_ub")
    model.addConstr(output.b <= 1 - c, name=f"{operation_name}_b_c_ub")
    model.addConstr(output.b >= x.b - c, name=f"{operation_name}_b_lb")

    return {}


def _and_operation_ul0_self(model, x, output, operation_name=''):
    """ul0 special version of AND(x, x): input is a (0,*,*) bit and both inputs are the same object.

    Under the ul0 encoding (r1=r2,b1=b2), further project out create_u:
      * output.ul = r or b   (3 constraints);
      * output.r  = r and not b  (3 constraints);
      * output.b  = b and not r  (3 constraints).
    0 auxiliary variables + 9 constraints; the original and_operation_ul0 requires
    1 variable + 20 constraints.
    Note: the input ul must be 0 (constant 0, or a variable guaranteed to be 0 by
    external constraints).
    """
    # output.ul = r or b
    model.addConstr(output.ul >= x.r, name=f"{operation_name}_ul_r_lb")
    model.addConstr(output.ul >= x.b, name=f"{operation_name}_ul_b_lb")
    model.addConstr(output.ul <= x.r + x.b, name=f"{operation_name}_ul_ub")
    # output.r = r and not b
    model.addConstr(output.r <= x.r, name=f"{operation_name}_r_act")
    model.addConstr(output.r <= 1 - x.b, name=f"{operation_name}_r_b_ub")
    model.addConstr(output.r >= x.r - x.b, name=f"{operation_name}_r_lb")
    # output.b = b and not r
    model.addConstr(output.b <= x.b, name=f"{operation_name}_b_act")
    model.addConstr(output.b <= 1 - x.r, name=f"{operation_name}_b_r_ub")
    model.addConstr(output.b >= x.b - x.r, name=f"{operation_name}_b_lb")

    return {}


def and_operation_ul0_x0(model, input1, input2, output, operation_name=''):
    """Second-round P_S x=0 AND cross-operation special case (input (0,*,*) bits).

    Applicable preconditions (guaranteed by the structure of
    create_second_P_S_operation; confirm before calling):
      * input2 = temp1[z][2] is the result of a "single-input XOR" (the other
        input old[z][2] is a filtered constant zero), so b2 == b1 (b identity
        propagation shares the variable) and r2 <= r1 (the single-input XOR r
        upper-bound constraint out.r + delta_r <= in.r);
      * both input ul values are constant 0.

    Under these preconditions the ul0 AND encoding can be compressed further
    (algebraic derivation + exhaustive verification over all 2^6 input/output
    combinations; the projected feasible region matches the original encoding with
    0 mismatches):
        c      = r1 and b          (create_u)
        out.r  = r1 and not b    out.r + c = r1
        out.b  = b and not r1    out.b + c = b
        out.ul = r2 or b          ((r1 or b) and (r2 or b) equals r2 or b when r2<=r1)
    Total: 1 auxiliary variable + 8 constraints; the original and_operation_ul0
    requires 1 variable + 20 constraints.
    Note: in this path c is not the output pure-u indicator (when r1=b=0,r2=1 the
    output is pure u but c=0), so _pure_u_var is not cached.
    """
    b1, b2 = input1.b, input2.b
    b_shared = (b1 is b2) or (isinstance(b1, int) and isinstance(b2, int) and b1 == b2)
    if not b_shared:
        raise ValueError(
            f"{operation_name}: _and_operation_ul0_x0 requires the two inputs to share b (b1=b2), "
            f"confirm the precondition (b identity propagation through single-input XOR) before calling"
        )
    for index, bit in enumerate([input1, input2]):
        if isinstance(bit.ul, int) and bit.ul != 0:
            raise ValueError(
                f"{operation_name}: _and_operation_ul0_x0 requires both inputs to have ul=0, "
                f"but input[{index}] has constant ul = {bit.ul}"
            )

    b = b1
    # c = r1 and b (c<=r1, c<=b are implied by out.r+c<=r1 / out.b+c<=b, omitted)
    c = model.addVar(vtype=GRB.BINARY, name=f'{operation_name}_create_u')
    model.addConstr(c >= input1.r + b - 1, name=f"{operation_name}_create_u_lb")
    # out.r = r1 and not b  out.r + c = r1
    model.addConstr(output.r + c <= input1.r, name=f"{operation_name}_r_eq_ub")
    model.addConstr(output.r + c >= input1.r, name=f"{operation_name}_r_eq_lb")
    # out.b = b and not r1  out.b + c = b
    model.addConstr(output.b + c <= b, name=f"{operation_name}_b_eq_ub")
    model.addConstr(output.b + c >= b, name=f"{operation_name}_b_eq_lb")
    # out.ul = r2 or b
    model.addConstr(output.ul >= input2.r, name=f"{operation_name}_ul_r2_lb")
    model.addConstr(output.ul >= b, name=f"{operation_name}_ul_b_lb")
    model.addConstr(output.ul <= input2.r + b, name=f"{operation_name}_ul_ub")

    return {}




# ============================================================================
# XAND (three-input single-output) compact encoding: whole-modeling of the
# differential propagation of y = x0  xor  (x1 and x2)
# ----------------------------------------------------------------------------
# The chi function (Keccak) and Step2 of Ascon P_S are all of the form
# x0  xor  (x1.x2); the original implementation split this into two steps
# (and_operation producing the intermediate and_bit, then
# xor_with_ul_input_no_delta_b). The intermediate bit and_bit and its pure-u /
# ul and b indicators serve only the immediately following XOR and never leak out.
# Here we give the unified encoding: project out the intermediate variables of the
# combined AND+XOR encoding (and_bit 3 variables / create_u / pure-u indicator /
# ul and b indicator), keeping only the delta_r variable; the constraints are the
# minimal linear-inequality description of the projected feasible set (pycddlib
# convex-hull facet + minimum set cover; exhaustively verified over all primary
# assignments + Gurobi solving that it is equivalent to the projection of the
# original combined encoding, 0 mismatches).
#
# Cost comparison (per chi bit):
#   original AND+XOR combined: 10~13 variables (and_bit3 + create_u1 + delta_r1 +
#                              pure-u/ul and b indicators 2~5 + output3), ~40+ constraints;
#   this XAND encoding:        4 variables (delta_r1 + output3), 34 constraints (general version).
# ============================================================================

def xand_operation(model, input0, input1, input2, output, operation_name=''):
    """XAND(y = x0  xor  (x1 and x2)) three-input single-output compact encoding (general version).
    New variables: delta_r(1) + output(3) = 4; 34 constraints.
    Obtained by projecting out the intermediate variables (and_bit/create_u/pure-u/
    ul and b indicators) of the combined AND+XOR encoding; exhaustively verified over
    all 2^12 primary assignments that the projected feasible region matches the
    original combined encoding with 0 mismatches."""
    u0, r0, b0 = input0.ul, input0.r, input0.b
    u1, r1, b1 = input1.ul, input1.r, input1.b
    u2, r2, b2 = input2.ul, input2.r, input2.b
    uy, ry, by = output.ul, output.r, output.b
    d = model.addVar(vtype=GRB.BINARY, name=f'{operation_name}_delta_r')
    # Minimal linear-inequality description of the projected feasible set (pycddlib
    # convex hull + minimum set cover), 34 constraints in total; exhaustively
    # verified to be equivalent to the projection of the AND+XOR combined encoding
    # (0 mismatches).
    model.addConstr(r0 + u1 + r1 - b1 + u2 + r2 - b2 - 5*ry + 4*by - 5*d <= 4, name=f'{operation_name}_xand_0')
    model.addConstr(b0 + r1 - b1 + r2 - b2 - uy - by - d <= 1, name=f'{operation_name}_xand_1')
    model.addConstr(u2 - r2 - b2 + ry + d <= 1, name=f'{operation_name}_xand_2')
    model.addConstr(r0 + b1 - u2 - r2 - b2 - by <= 1, name=f'{operation_name}_xand_3')
    model.addConstr(b0 + r1 - u2 - r2 - b2 - by <= 1, name=f'{operation_name}_xand_4')
    model.addConstr(4*u0 + 2*r0 + 2*u1 + r1 - 2*b1 - u2 - b2 - 4*uy - 5*ry + 4*by - 5*d <= 4, name=f'{operation_name}_xand_5')
    model.addConstr(b0 - u1 - r1 - b1 + r2 - by <= 1, name=f'{operation_name}_xand_6')
    model.addConstr(r0 - u1 - r1 - b1 + b2 - uy - ry - by - d <= 0, name=f'{operation_name}_xand_7')
    model.addConstr(- u0 + r0 - 3*b0 - 2*u1 - 2*b1 - 2*u2 - 2*b2 + 2*uy - 2*ry + 3*by - d <= 1, name=f'{operation_name}_xand_8')
    model.addConstr(- u0 - r1 - b1 - u2 + uy + by <= 1, name=f'{operation_name}_xand_9')
    model.addConstr(u0 - b0 + b1 + 3*u2 + r2 + b2 - 4*uy - ry + 2*by - 2*d <= 3, name=f'{operation_name}_xand_10')
    model.addConstr(u0 - b0 + 3*u1 + r1 + b1 + b2 - 4*uy - ry + 2*by - 2*d <= 3, name=f'{operation_name}_xand_11')
    model.addConstr(u0 - r0 - 3*b0 + 2*u1 - r1 - 2*b1 + 2*u2 - r2 - 2*b2 - 3*uy + 3*ry + 2*by <= 2, name=f'{operation_name}_xand_12')
    model.addConstr(2*u0 - 3*r0 - 3*b0 + u1 + u2 - uy + ry + 2*by + d <= 3, name=f'{operation_name}_xand_13')
    model.addConstr(- u0 - 5*r0 + b0 - 2*u1 - 2*r1 - 5*r2 + 2*b2 + 2*uy + 5*ry - 2*by + 5*d <= 3, name=f'{operation_name}_xand_14')
    model.addConstr(- u0 + r1 - u2 - r2 - b2 - ry - d <= 0, name=f'{operation_name}_xand_15')
    model.addConstr(- u0 - b0 + r1 - b1 + r2 - b2 - uy - ry + by - 2*d <= 0, name=f'{operation_name}_xand_16')
    model.addConstr(- u0 - u1 - r1 - b1 + b2 - by <= 0, name=f'{operation_name}_xand_17')
    model.addConstr(- u0 + u1 - r1 + b1 + u2 - r2 + b2 - 3*uy - by <= 0, name=f'{operation_name}_xand_18')
    model.addConstr(- u0 - 5*r0 + b0 - 5*r1 + 2*b1 - 2*u2 - 2*r2 + 2*uy + 5*ry - 2*by + 5*d <= 3, name=f'{operation_name}_xand_19')
    model.addConstr(- 2*u0 - 2*u1 - r1 - u2 - r2 - 2*b2 + 2*uy + ry + d <= 1, name=f'{operation_name}_xand_20')
    model.addConstr(- 3*u0 - u1 - b1 + r2 - 3*b2 + 3*uy - 3*ry + 2*by <= 3, name=f'{operation_name}_xand_21')
    model.addConstr(b0 - u1 - b1 - u2 - b2 - by <= 0, name=f'{operation_name}_xand_22')
    model.addConstr(4*u0 + 2*r0 - u1 - b1 + 2*u2 + r2 - 2*b2 - 4*uy - 5*ry + 4*by - 5*d <= 4, name=f'{operation_name}_xand_23')
    model.addConstr(3*u0 - 3*r0 + b0 + u1 - r1 + b1 - r2 + 2*b2 - uy + 7*ry - 5*by + 6*d <= 6, name=f'{operation_name}_xand_24')
    model.addConstr(u0 - b0 + 3*r1 + b1 + 3*r2 + b2 - uy - 4*ry + 7*by - 4*d <= 8, name=f'{operation_name}_xand_25')
    model.addConstr(u1 - r1 - b1 + ry + d <= 1, name=f'{operation_name}_xand_26')
    model.addConstr(r0 - r1 + b1 - r2 + b2 - uy - ry - d <= 1, name=f'{operation_name}_xand_27')
    model.addConstr(r1 + b2 + ry + d <= 2, name=f'{operation_name}_xand_28')
    model.addConstr(b1 + b2 - 2*uy - by <= 0, name=f'{operation_name}_xand_29')
    model.addConstr(b1 + r2 + ry + d <= 2, name=f'{operation_name}_xand_30')
    model.addConstr(uy - by + d <= 1, name=f'{operation_name}_xand_31')
    model.addConstr(u0 - uy - d <= 0, name=f'{operation_name}_xand_32')
    model.addConstr(u0 + b0 - uy <= 1, name=f'{operation_name}_xand_33')
    return {'delta_r': d, 'delta_b': 0, 'has_ul': None}



def xand_operation_ul0(model, input0, input1, input2, output, operation_name=''):
    """XAND ul0 version: inputs x0,x1,x2 have constant ul=0 (the temp1 bits of
    Ascon's second P_S Step2).
    Corresponds to the combination and_operation_ul0 + XOR (A2 projection),
    4 new variables, 12 constraints.
    Note: the input ul must be 0 (constant 0, or a variable guaranteed to be 0 by
    external constraints)."""
    for index, bit in enumerate([input0, input1, input2]):
        if isinstance(bit.ul, int) and bit.ul != 0:
            raise ValueError(
                f"{operation_name}: xand_operation_ul0 requires all three inputs to have ul=0, "
                f"but input[{index}] has constant ul = {bit.ul}"
            )
    r0, b0 = input0.r, input0.b
    r1, b1 = input1.r, input1.b
    r2, b2 = input2.r, input2.b
    uy, ry, by = output.ul, output.r, output.b
    d = model.addVar(vtype=GRB.BINARY, name=f'{operation_name}_delta_r')
    # Minimal linear-inequality description of the projected feasible set (pycddlib
    # convex hull + minimum set cover), 12 constraints in total; exhaustively
    # verified to be equivalent to the projection of the AND+XOR combined encoding
    # (0 mismatches).
    model.addConstr(- r2 - b2 + uy <= 0, name=f'{operation_name}_xand_0')
    model.addConstr(- b1 - b2 + 2*uy - ry + d <= 1, name=f'{operation_name}_xand_1')
    model.addConstr(b0 - b1 - b2 - by <= 0, name=f'{operation_name}_xand_2')
    model.addConstr(- r1 - b1 + uy <= 0, name=f'{operation_name}_xand_3')
    model.addConstr(- r0 - r1 + b1 - r2 + b2 - uy + ry - by + d <= 0, name=f'{operation_name}_xand_4')
    model.addConstr(- 2*r0 + b0 - 2*r1 - b1 - 2*r2 - b2 + 2*uy + 2*ry - by + 2*d <= 0, name=f'{operation_name}_xand_5')
    model.addConstr(- 2*b0 - b1 - b2 + uy + 2*by <= 1, name=f'{operation_name}_xand_6')
    model.addConstr(r0 - r1 - r2 - ry - d <= 0, name=f'{operation_name}_xand_7')
    model.addConstr(r1 + b2 + by <= 2, name=f'{operation_name}_xand_8')
    model.addConstr(r1 + 2*b1 + r2 + 2*b2 - uy + 4*ry - 2*by + 3*d <= 5, name=f'{operation_name}_xand_9')
    model.addConstr(b1 + r2 + by <= 2, name=f'{operation_name}_xand_10')
    model.addConstr(r1 + r2 - 2*uy - ry - 2*d <= 0, name=f'{operation_name}_xand_11')
    return {'delta_r': d, 'delta_b': 0, 'has_ul': None}



def xand_operation_ul0_x0(model, input0, input1, input2, output, operation_name=''):
    """XAND ul0x0 special version: ul0 and b1==b2 (b identity propagation shares the
    variable), r2<=r1 (x2 is a single-input XOR result, for Ascon second P_S x==0
    with old_state[z][2] constant zero).
    Corresponds to the combination and_operation_ul0_x0 + XOR (A2 projection),
    4 new variables, 12 constraints.
    Confirm that b1/b2 are shared before calling, otherwise a ValueError is raised."""
    for index, bit in enumerate([input0, input1, input2]):
        if isinstance(bit.ul, int) and bit.ul != 0:
            raise ValueError(
                f"{operation_name}: xand_operation_ul0_x0 requires all three inputs to have ul=0, "
                f"but input[{index}] has constant ul = {bit.ul}"
            )
    b1, b2 = input1.b, input2.b
    b_shared = (b1 is b2) or (isinstance(b1, int) and isinstance(b2, int) and b1 == b2)
    if not b_shared:
        raise ValueError(
            f"{operation_name}: xand_operation_ul0_x0 requires input1.b and input2.b to be shared "
            f"(b1=b2); confirm the precondition (b identity propagation through single-input XOR) before calling"
        )
    r0, b0 = input0.r, input0.b
    r1, r2 = input1.r, input2.r
    b = input1.b  # b1==b2 shared
    uy, ry, by = output.ul, output.r, output.b
    d = model.addVar(vtype=GRB.BINARY, name=f'{operation_name}_delta_r')
    # Minimal linear-inequality description of the projected feasible set (pycddlib
    # convex hull + minimum set cover), 12 constraints in total; exhaustively
    # verified to be equivalent to the projection of the AND+XOR combined encoding
    # (0 mismatches).
    model.addConstr(- b0 + r1 - b - ry + by - d <= 0, name=f'{operation_name}_xand_0')
    model.addConstr(- b0 - b + by <= 0, name=f'{operation_name}_xand_1')
    model.addConstr(- r2 - b + uy <= 0, name=f'{operation_name}_xand_2')
    model.addConstr(- r0 - r1 + b + ry - by + d <= 0, name=f'{operation_name}_xand_3')
    model.addConstr(- r0 - r1 + ry + d <= 0, name=f'{operation_name}_xand_4')
    model.addConstr(b - uy <= 0, name=f'{operation_name}_xand_5')
    model.addConstr(r0 - r1 - ry - d <= 0, name=f'{operation_name}_xand_6')
    model.addConstr(r2 - uy - d <= 0, name=f'{operation_name}_xand_7')
    model.addConstr(r1 - b + 2*uy - ry + d <= 2, name=f'{operation_name}_xand_8')
    model.addConstr(r1 + uy - ry + by <= 2, name=f'{operation_name}_xand_9')
    model.addConstr(b0 - b - by <= 0, name=f'{operation_name}_xand_10')
    model.addConstr(r1 + b + ry + d <= 2, name=f'{operation_name}_xand_11')
    return {'delta_r': d, 'delta_b': 0, 'has_ul': None}

def xor_with_ul_input_no_delta_b(model, inputs, output, operation_name='', need_has_ul=False, need_output_ul=True):
    """XOR operation with nonlinear inputs and without blue cancellation.

    need_has_ul: whether to build the has_ul indicator (OR of input ul values).
    Only the C/D computation of Keccak theta needs to read it
    (theta_vars['has_ul']); Ascon does not use it. It is not created by
    default to keep the model smaller, and has_ul is None in the returned dict.

    Projecting out has_r/has_b: these two variables only appear as OR(r_i)/OR(b_i)
    in
        delta_r <= has_r,  output.r <= has_r,  output.r >= has_r - delta_r - sum(pure_u)
        output.b <= has_b,  output.b >= has_b - sum(pure_u)
    They are never returned to the caller, so they can be projected out over the
    integers directly (same integer feasible region, unchanged LP relaxation;
    enumerated over all 0/1 assignments of 2/3 inputs, 0 mismatches):
        delta_r <= sum(r_i),  output.r <= sum(r_i),
        output.r >= r_i - delta_r - sum(pure_u)  (i),
        output.b <= sum(b_i),  output.b >= b_i - sum(pure_u)  (i).
    Each XOR saves 2 binary auxiliary variables; the constraint count stays the
    same (3 inputs) or decreases (2 inputs).

    Constant-zero folding (new): a constant (0,0,0) input is the identity element
    of XOR, contributing 0 to every constraint, so it is filtered out first; if no
    input remains after filtering, the output is uniquely determined to be (0,0,0)
    and is folded to a constant without creating any variable/constraint (integer
    feasible region and LP relaxation are both unchanged).

    need_output_ul (A1): whether the output ul channel needs to be modeled. When
    False, the has_ul_b indicator and all ul propagation constraints are skipped,
    and output.ul is folded to constant 0 (the caller must guarantee that
    output.ul is never read; e.g. the sum_of_a0a2a4 XOR in the attack scripts only
    reads output .r/.b, while delta_r is still kept). The ul channel constrains
    only output.ul itself; removing it does not change the feasible-region
    projection of any other variable.

    Two-input projection (A2): with two inputs and exactly one input having
    constant ul=0, the has_ul_b indicator (1 variable + 3 constraints) is exactly
    projected out into 4 direct constraints (2^3 exhaustive verification +
    exact Fourier-Motzkin LP projection, see the implementation comment below).
    Typical scenario: round1 P_S Step2's temp2 = temp1  xor  and_bit (temp1 is a
    (0,*,*) bit; and_bit's has_ul_b is not cached).
    """
    # Filter constant-zero inputs (identity element): after filtering, every constraint is equivalent (0's contribution is always 0)
    inputs = [i for i in inputs if not _is_const_zero(i)]
    if not inputs:
        # All constant-zero inputs: the output is uniquely determined as (0,0,0), delta_r is forced to 0
        output.r = 0
        output.b = 0
        if isinstance(output.ul, int):
            if output.ul != 0:
                raise ValueError(
                    f"{operation_name}: all-zero-input XOR requires output ul=0, "
                    f"but output.ul has constant value {output.ul}")
        else:
            output.ul = 0  # folded to a constant; downstream references simplify automatically
        has_ul = None
        if need_has_ul:
            # Keccak theta reads has_ul; with all-zero inputs has_ul is always 0, so return a variable fixed to 0
            has_ul = model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_has_ul")
            has_ul.lb = has_ul.ub = 0
        return {'delta_r': 0, 'delta_b': 0, 'has_ul': has_ul}

    delta_r = model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_delta_r")

    has_ul = None
    if need_has_ul:
        has_ul = model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_has_ul")
        add_or(model, has_ul, [i.ul for i in inputs], f"{operation_name}_has_ul")

    sum_r = gp.quicksum(i.r for i in inputs)
    sum_b = gp.quicksum(i.b for i in inputs)

    # Whether each input bit is pure u - cached and reused, evaluated once per Bit
    sum_temp_u = []
    for index, bit in enumerate(inputs):
        temp = _get_pure_u_indicator(model, bit, f"{operation_name}_input_{index}")
        sum_temp_u.append(temp)

    # ---- output.ul channel ----
    if not need_output_ul:
        # A1: the whole output ul channel is not needed (the caller guarantees
        # output.ul is not read, e.g. the sum_of_a0a2a4 XOR in attack scripts only
        # reads .r/.b). Skip the has_ul_b indicator and all ul propagation
        # constraints, and fold output.ul to constant 0 - the ul channel constrains
        # only output.ul itself; removing it does not change the feasible-region
        # projection of r/b/delta_r or any other variable.
        if isinstance(output.ul, int):
            if output.ul != 0:
                raise ValueError(
                    f"{operation_name}: need_output_ul=False requires output ul=0, "
                    f"but output.ul has constant value {output.ul}")
        else:
            output.ul = 0
            output._pure_u_var = None
            output._has_ul_b_var = None
    else:
        # A2 (two-input projection): with two inputs and exactly one input having
        # constant ul=0, the has_ul_b indicator (ub = ul and b, 1 variable + 3
        # constraints) can be exactly projected out. Let the other input be
        # (ul2,b2); after Fourier-Motzkin elimination of ub the projection is exact
        # (integer domain and LP relaxation agree; exhaustively enumerated over all
        # 2^3 input/output assignments, 0 mismatches):
        #   out.ul >= ul2 - delta_r       (was >= ul2-delta_r)
        #   out.ul <= ul2                 (was <= sum(ul_i) = ul2)
        #   out.ul + delta_r <= 1 + b2    (was <= 1-delta_r+ub merged with ub<=b2)
        #   out.ul >= ul2 + b2 - 1        (was >= ub merged with ub>=ul2+b2-1)
        # Typical scenario: round1 P_S Step2's temp2 = temp1  xor  and_bit (temp1 is a
        # (0,*,*) bit; and_bit's has_ul_b is used only by this XOR, so caching
        # gives no reuse benefit).
        a2_projected = False
        if len(inputs) == 2:
            ul_flags = [i.ul for i in inputs]
            zero_idx = [idx for idx, u in enumerate(ul_flags)
                        if isinstance(u, int) and u == 0]
            if len(zero_idx) == 1:
                other = 1 - zero_idx[0]
                if not (isinstance(ul_flags[other], int) and ul_flags[other] == 0):
                    other_bit = inputs[other]
                    model.addConstr(output.ul >= other_bit.ul - delta_r,
                                    name=f"{operation_name}_ul_other_lb")
                    model.addConstr(output.ul <= other_bit.ul,
                                    name=f"{operation_name}_ul_other_ub")
                    model.addConstr(output.ul + delta_r <= 1 + other_bit.b,
                                    name=f"{operation_name}_ul_b_ub")
                    model.addConstr(output.ul >= other_bit.ul + other_bit.b - 1,
                                    name=f"{operation_name}_ul_cross_lb")
                    a2_projected = True
        if not a2_projected:
            # Whether each input bit has ul=1 and b=1 - cached and reused, evaluated once per Bit
            sum_temp_ub = []
            for index, bit in enumerate(inputs):
                temp = _get_has_ul_b_indicator(model, bit, f"{operation_name}_input_{index}")
                sum_temp_ub.append(temp)

            # Mixed ul0 inputs: when an input ul is constant 0, output.ul >=
            # bit.ul - delta_r is always true (0 >= -delta_r), and output.ul >= ub
            # is always true (ub is always 0), so skip them.
            # Typical scenario: one input of round1 P_S's temp2 XOR is a (0,*,*) bit.
            for index, bit in enumerate(inputs):
                if isinstance(bit.ul, int) and bit.ul == 0:
                    continue
                model.addConstr(output.ul >= bit.ul - delta_r)
            model.addConstr(output.ul <= gp.quicksum([i.ul for i in inputs]))
            model.addConstr(output.ul <= 1 - delta_r + gp.quicksum(sum_temp_ub))
            for ub in sum_temp_ub:
                if isinstance(ub, int) and ub == 0:
                    continue
                model.addConstr(output.ul >= ub)

    # output.r / output.b: has_r / has_b have been projected out (see the function docstring)
    sum_u = gp.quicksum(sum_temp_u)
    # Upper-bound merge: delta_r <= sum_r and output.r <= sum_r merge into
    # delta_r+output.r <= sum_r. The model also enforces output.r + delta_r <= 1
    # (the pure-u suppression constraint below), so the integer feasible region is
    # unchanged after merging (exhaustively verified over all 0/1 assignments of
    # 2/3 inputs, 0 mismatches).
    model.addConstr(delta_r + output.r <= sum_r)
    for index, r_i in enumerate([i.r for i in inputs]):
        # Coefficient tightening (B1): drop the self pure-u term p_i - when p_i=1
        # we have r_i=0 (pure-u definition), so that term is always 0; when p_i=0
        # it is identical to the original expression. Integer-equivalent and the LP
        # relaxation is strictly tighter (sum_u - p_i <= sum_u). The benefit is
        # most visible for the 3-input P_L XOR.
        model.addConstr(output.r >= r_i - delta_r - (sum_u - sum_temp_u[index]))
    # Constant-0 pure-u indicator terms: output.r + 0 + delta_r <= 1 is dominated
    # by any nonzero term (a nonzero term imposes output.r + p + delta_r <= 1 with
    # p >= 0  output.r+delta_r <= 1), so skip it; when all are 0, keep one as the
    # only suppression upper bound of the r channel.
    has_nonzero_u = any(not (isinstance(u, int) and u == 0) for u in sum_temp_u)
    for if_u in sum_temp_u:
        if isinstance(if_u, int) and if_u == 0 and has_nonzero_u:
            continue
        model.addConstr(output.r + if_u + delta_r <= 1)

    model.addConstr(output.b <= sum_b)
    for index, b_i in enumerate([i.b for i in inputs]):
        # Coefficient tightening (B1): same as the r channel, drop the self pure-u term (when p_i=1, b_i=0).
        model.addConstr(output.b >= b_i - (sum_u - sum_temp_u[index]))
    for t_u in sum_temp_u:
        if isinstance(t_u, int) and t_u == 0:
            continue  # output.b <= 1 - 0 is always true
        model.addConstr(output.b <= 1 - t_u)

    return {
        "delta_r": delta_r,
        "delta_b": 0,
        'has_ul': has_ul,
    }


def xor_with_ul0_input_no_delta_b(model, inputs, output, operation_name=''):
    """XOR special version: the input state is (0,*,*) - all input bits have ul=0.

    Applicable scenario: the XOR of Ascon's first P_L. All bits of the first P_S's
    output (i.e. the first P_L's input) are of type (0,*,*) / (0,0,0) / (0,*,0),
    so ul is always constant 0.

    Simplifications relative to `xor_with_ul_input_no_delta_b` (substituting ul=0
    into the original constraints):
      * each input's pure-u indicator (1 variable + 4 constraints) is always 0 and
        is not created;
      * each input's ul and b indicator (1 variable + 3 constraints) is always 0 and
        is not created;
      * the output ul is always 0 (inputs have no ul, XOR produces no ul), so all
        4 groups of ul propagation constraints disappear;
      * the b pure-u suppression constraint (output.b <= 1-p) is always true and
        is removed.
    Only r/b propagation and the red-red marker delta_r remain:
        delta_r <= sum(r_i),  output.r <= sum(r_i),
        output.r >= r_i - delta_r (i),  output.r + delta_r <= 1,
        output.b <= sum(b_i),  output.b >= b_i (i).
    For a 3-input XOR: the original requires 7 auxiliary variables + ~20
    constraints, the special version only 1 + 10.

    The returned dict has the same structure as `xor_with_ul_input_no_delta_b`
    (delta_r/delta_b/has_ul), keeping attack-script objective functions and output
    scripts compatible.

    Note: the input ul must be 0 - constant 0, or a variable guaranteed to be 0 by
     external constraints (the special version does not reference input ul; the
     caller must ensure the precondition holds).

    Constant-zero folding (new): a constant (0,0,0) input is the identity element
    of XOR, so it is filtered out first; if no input remains after filtering, the
    output is uniquely determined to be (0,0,0) and is folded to a constant
    without creating any variable/constraint (integer feasible region and LP
    relaxation are both unchanged).
    """
    for index, bit in enumerate(inputs):
        if isinstance(bit.ul, int) and bit.ul != 0:
            raise ValueError(
                f"{operation_name}: xor_with_ul0_input_no_delta_b requires all inputs to have ul=0, "
                f"but input[{index}] has constant ul = {bit.ul}"
            )
    # Filter constant-zero inputs (identity element): after filtering, every constraint is equivalent (0's contribution is always 0)
    inputs = [i for i in inputs if not _is_const_zero(i)]
    if not inputs:
        # All constant-zero inputs: the output is uniquely determined as (0,0,0), delta_r is forced to 0
        output.r = 0
        output.b = 0
        if isinstance(output.ul, int):
            if output.ul != 0:
                raise ValueError(
                    f"{operation_name}: xor_with_ul0_input_no_delta_b requires output ul=0, "
                    f"but output.ul has constant value {output.ul}"
                )
        else:
            output.ul = 0  # folded to a constant; downstream references simplify automatically
        return {'delta_r': 0, 'delta_b': 0, 'has_ul': None}

    if len(inputs) == 1:
        # Single-input ul0 XOR (S5): only 1 input remains after filtering constant
        # identity elements.
        # Under differential propagation, ul/r may be cancelled depending on
        # delta_r (red can disappear), but b is strictly identity.
        # Strictly equivalent minimal encoding (algebraic + exhaustive verification):
        #   out.ul = 0 (constant);  out.b = X.b (shared variable);
        #   out.r + delta_r <= X.r;  out.r >= X.r - delta_r.
        # Original encoding: 1 variable (delta_r) + 6 constraints + an independent
        # b variable;
        # new encoding: 1 variable (delta_r) + 2 constraints, b shares the input
        # variable directly.
        # The delta_r variable must be kept: the attack-script objective function
        # (delta_total_r) references it, and eliminating it would change the
        # objective semantics (single-input red-cancellation freedom), so only the
        # constraints are simplified and b is shared.
        # Typical scenarios: round1 P_S temp1[z][2] = old[z][1]  xor  constant zero;
        #           the single-input XOR at the boundary z of round0 P_L (the z=31
        #           row's all-zero propagation).
        bit = inputs[0]
        if isinstance(output.ul, int):
            if output.ul != 0:
                raise ValueError(
                    f"{operation_name}: xor_with_ul0_input_no_delta_b requires output ul=0, "
                    f"but output.ul has constant value {output.ul}"
                )
        else:
            output.ul = 0
        output.b = bit.b  # b identity propagation: share the input b (variable or constant) directly, saving 1 variable + 2 constraints
        # Flags are (re)assigned, so the previous indicator cache is invalidated and must be reset
        output._pure_u_var = None
        output._has_ul_b_var = None
        delta_r = model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_delta_r")
        model.addConstr(output.r + delta_r <= bit.r, name=f"{operation_name}_r_ub")
        model.addConstr(output.r >= bit.r - delta_r, name=f"{operation_name}_r_lb")
        return {'delta_r': delta_r, 'delta_b': 0, 'has_ul': None}

    if isinstance(output.ul, int):
        if output.ul != 0:
            raise ValueError(
                f"{operation_name}: xor_with_ul0_input_no_delta_b requires output ul=0, "
                f"but output.ul has constant value {output.ul}"
            )
    else:
        # When all input ul are 0, the XOR output ul is always 0 (equivalent to the original output.ul <= sum(ul_i))
        model.addConstr(output.ul == 0, name=f"{operation_name}_ul_zero")

    delta_r = model.addVar(vtype=GRB.BINARY, name=f"{operation_name}_delta_r")

    sum_r = gp.quicksum(i.r for i in inputs)
    sum_b = gp.quicksum(i.b for i in inputs)

    # Upper-bound merge: delta_r <= sum_r and output.r <= sum_r merge into
    # delta_r+output.r <= sum_r. The model also keeps the output.r + delta_r <= 1
    # suppression constraint, so the integer feasible region is unchanged after
    # merging (exhaustively verified, 0 mismatches).
    model.addConstr(delta_r + output.r <= sum_r, name=f"{operation_name}_r_ub")
    for index, bit in enumerate(inputs):
        model.addConstr(output.r >= bit.r - delta_r, name=f"{operation_name}_r_lb_{index}")
    # The original output.r + pure_u_i + delta_r <= 1, with pure_u_i always 0 only 1 remains
    model.addConstr(output.r + delta_r <= 1, name=f"{operation_name}_r_suppress")
    model.addConstr(output.b <= sum_b, name=f"{operation_name}_b_ub")
    for index, bit in enumerate(inputs):
        model.addConstr(output.b >= bit.b, name=f"{operation_name}_b_lb_{index}")

    return {
        "delta_r": delta_r,
        "delta_b": 0,
        'has_ul': None,
    }


def and_operation(model, input1, input2, output, operation_name=''):
    """Compact AND encoding: 1 auxiliary variable (create_u) + 26 constraints
    (excluding the cached pure-u indicators).

    Relative to the original encoding (has_c1/has_c2/new_ul/input_u/create_u/
    has_r/has_b, 7 auxiliary variables + 36 constraints), each AND saves 6
    auxiliary variables + 10 constraints, with exactly the same integer feasible
    region (exhaustively verified over all 64 input-type combinations x 8 output
    values, 0 mismatches). The intermediate variables are "projected out" into the
    constraints:
      * output.ul = u1  or  u2  or  ((r1 or b1)  and  (r2 or b2)) via 4 bicolor lower bounds + 2
        source upper bounds;
      * create_u = (r1 and b2)  or  (r2 and b1) is kept (serving both the ul lower bound and
        the r/b suppression);
      * output.r/b use the activation/suppression constraints of r1/r2 and b1/b2
        respectively to replace has_r/has_b/input_u.

    New special-case classification:
      * when the two inputs are equivalent at the variable level (X AND X,
        including the same object and bits sharing the same set of variables): use
        the self-AND simplified encoding `_and_operation_self`
        (1 variable + 15 constraints, see that function).

    Note: a constant-zero (0,0,0) input is **not** the identity element of AND.
    Under differential propagation semantics, the output of AND((0,0,0), X) is
    (u2, r2 and not p2, b2 and not p2) - it keeps X's non-pure-u information (only the pure-u
    r/b is suppressed), so it cannot be folded to constant zero (confirmed by
    exhaustive verification).
    """
    # Same-object self AND (X AND X): constraints merged and deduplicated, see _and_operation_self
    # Note: use variable-equivalence detection rather than object identity - bits
    # sharing the same set of variables (different Bit objects whose flags
    # reference the same set of variables) are likewise treated as self-operation.
    if _bits_equivalent(input1, input2):
        return _and_operation_self(model, input1, output, operation_name)

    # create_u = (r1 and b2)  or  (r2 and b1)
    c = model.addVar(vtype=GRB.BINARY, name=f'{operation_name}_create_u')
    model.addConstr(c >= input1.r + input2.b - 1)
    model.addConstr(c >= input2.r + input1.b - 1)
    model.addConstr(c <= input1.r + input1.b)
    model.addConstr(c <= input2.r + input2.b)
    # Note: the two upper bounds c <= r1+r2 and c <= b1+b2 are implied by
    # c+output.r <= r1+r2 / c+output.b <= b1+b2 below (output.r/b >= 0), omitted.

    # output.ul = u1  or  u2  or  ((r1 or b1)  and  (r2 or b2))
    model.addConstr(output.ul >= input1.ul)
    model.addConstr(output.ul >= input2.ul)
    model.addConstr(output.ul >= input1.r + input2.r - 1)
    model.addConstr(output.ul >= input1.b + input2.b - 1)
    # The two cross-color lower bounds output.ul >= r1+b2-1 and >= b1+r2-1 are
    # captured by the definition of create_u: c=1  one of the cross-color cases
    # holds  output.ul >= c (exhaustively verified over all 0/1 assignments, 0
    # mismatches).
    model.addConstr(output.ul >= c)
    model.addConstr(output.ul <= input1.ul + input2.ul + input1.r + input1.b)
    model.addConstr(output.ul <= input1.ul + input2.ul + input2.r + input2.b)

    # output.r / output.b (p1, p2 reuse the cached pure-u indicators)
    p1 = _get_pure_u_indicator(model, input1, f"{operation_name}_input_0")
    p2 = _get_pure_u_indicator(model, input2, f"{operation_name}_input_1")
    # Upper-bound merge: c<=r1+r2 and output.r<=r1+r2 merge into c+output.r<=r1+r2 (exhaustively verified equivalent)
    model.addConstr(c + output.r <= input1.r + input2.r)
    model.addConstr(output.r <= 1 - p1)
    model.addConstr(output.r <= 1 - p2)
    model.addConstr(output.r <= 1 - c)
    # Coefficient tightening (B1): drop the self pure-u terms - p1=1  r1=0,
    # p2=1  r2=0, so those terms are always 0; when p=0 they equal the original
    # expression. Integer-equivalent and the LP relaxation is strictly tighter.
    model.addConstr(output.r >= input1.r - p2 - c)
    model.addConstr(output.r >= input2.r - p1 - c)
    model.addConstr(c + output.b <= input1.b + input2.b)
    model.addConstr(output.b <= 1 - p1)
    model.addConstr(output.b <= 1 - p2)
    model.addConstr(output.b <= 1 - c)
    model.addConstr(output.b >= input1.b - p2 - c)
    model.addConstr(output.b >= input2.b - p1 - c)

    return {}


def and_operation_ul0(model, input1, input2, output, operation_name=''):
    """AND special version: the input state is (0,*,*) - both input bits have ul=0.

    Applicable scenario: Ascon's second P_S (the AND inside chi). All bits of
    temp_state_1 have ul constant 0:
      * x=1,3 directly copy the first P_L's output (ul=0 constant);
      * x=0,2,4 are the results of ul=0-input XORs (the ul variable is forced to 0
        by the XOR constraints).

    Simplifications relative to `and_operation` (substituting ul=0 into the
    original constraints):
      * output.ul = (r1 or b1)  and  (r2 or b2), no longer depending on input ul; the 2 ul
        lower bounds disappear, and the 2 upper bounds simplify to
        output.ul <= r1+b1 and <= r2+b2;
      * the pure-u indicators (2 variables + 8 constraints) are always 0 and are
        not created, the suppression constraints output.r/b <= 1-p are always true
        and removed, and the r/b lower bounds drop the p1+p2 terms.
    Only create_u and the r/b activation/suppression remain:
      1 auxiliary variable (create_u) + 20 constraints;
      the original requires 3 auxiliary variables + 32 constraints (including the
      pure-u indicators).

    Note: the input ul must be 0 - constant 0, or a variable guaranteed to be 0 by
    external constraints (the special version does not reference input ul; the
    caller must ensure the precondition holds).

    New special-case classification:
      * when the two inputs are equivalent at the variable level (X AND X,
        including the same object and bits sharing the same set of variables): use
        the self-AND ul0 simplified encoding `_and_operation_ul0_self`
        (0 variables + 9 constraints, see that function).

    Note: a constant-zero (0,0,0) input is **not** the identity element of AND
    (same as and_operation, confirmed by exhaustive verification) and cannot be
    folded to constant zero.
    """
    for index, bit in enumerate([input1, input2]):
        if isinstance(bit.ul, int) and bit.ul != 0:
            raise ValueError(
                f"{operation_name}: and_operation_ul0 requires both inputs to have ul=0, "
                f"but input[{index}] has constant ul = {bit.ul}"
            )
    # Same-object self AND (X AND X, ul0): project out create_u, see _and_operation_ul0_self
    # Note: use variable-equivalence detection rather than object identity (same as and_operation).
    if _bits_equivalent(input1, input2):
        return _and_operation_ul0_self(model, input1, output, operation_name)

    # create_u = (r1 and b2)  or  (r2 and b1)
    c = model.addVar(vtype=GRB.BINARY, name=f'{operation_name}_create_u')
    model.addConstr(c >= input1.r + input2.b - 1)
    model.addConstr(c >= input2.r + input1.b - 1)
    model.addConstr(c <= input1.r + input1.b)
    model.addConstr(c <= input2.r + input2.b)
    # Note: the two upper bounds c <= r1+r2 and c <= b1+b2 are implied by
    # c+output.r <= r1+r2 / c+output.b <= b1+b2 below (output.r/b >= 0), omitted.

    # output.ul = (r1 or b1)  and  (r2 or b2) (u1=u2=0)
    model.addConstr(output.ul >= input1.r + input2.r - 1)
    model.addConstr(output.ul >= input1.b + input2.b - 1)
    # The two cross-color lower bounds output.ul >= r1+b2-1 and >= b1+r2-1 are
    # captured by the definition of create_u: c=1  one of the cross-color cases
    # holds  output.ul >= c (exhaustively verified over all 0/1 assignments, 0
    # mismatches).
    model.addConstr(output.ul >= c)
    model.addConstr(output.ul <= input1.r + input1.b)
    model.addConstr(output.ul <= input2.r + input2.b)

    # output.r / output.b (pure-u indicators are always 0, suppression terms removed)
    # Upper-bound merge: c<=r1+r2 and output.r<=r1+r2 merge into c+output.r<=r1+r2 (exhaustively verified equivalent)
    model.addConstr(c + output.r <= input1.r + input2.r)
    model.addConstr(output.r <= 1 - c)
    model.addConstr(output.r >= input1.r - c)
    model.addConstr(output.r >= input2.r - c)
    model.addConstr(c + output.b <= input1.b + input2.b)
    model.addConstr(output.b <= 1 - c)
    model.addConstr(output.b >= input1.b - c)
    model.addConstr(output.b >= input2.b - c)

    # create_u is exactly the indicator of output pure u (ul=1,r=0,b=0):
    # with both input ul=0, c=1  output=(1,0,0) (exhaustively verified over all
    # 2^8 input/output/auxiliary combinations, 0 mismatches). Cache it for reuse by
    # the downstream XOR, saving 1 indicator variable + 4 definition constraints.
    output._pure_u_var = c

    return {}
