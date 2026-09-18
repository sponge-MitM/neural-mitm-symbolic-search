# -*- coding: utf-8 -*-
'Module object.'
import itertools

import numpy as np



IDX = {'u0': 0, 'r0': 1, 'b0': 2, 'u1': 3, 'r1': 4, 'b1': 5,
       'u2': 6, 'r2': 7, 'b2': 8, 'uy': 9, 'ry': 10, 'by': 11, 'd': 12}
CONSTRAINTS = [
    ('r0+u1+r1-b1+u2+r2-b2-5ry+4by-5d', 4),
    ('b0+r1-b1+r2-b2-uy-by-d', 1),
    ('u2-r2-b2+ry+d', 1),
    ('r0+b1-u2-r2-b2-by', 1),
    ('b0+r1-u2-r2-b2-by', 1),
    ('4u0+2r0+2u1+r1-2b1-u2-b2-4uy-5ry+4by-5d', 4),
    ('b0-u1-r1-b1+r2-by', 1),
    ('r0-u1-r1-b1+b2-uy-ry-by-d', 0),
    ('-u0+r0-3b0-2u1-2b1-2u2-2b2+2uy-2ry+3by-d', 1),
    ('-u0-r1-b1-u2+uy+by', 1),
    ('u0-b0+b1+3u2+r2+b2-4uy-ry+2by-2d', 3),
    ('u0-b0+3u1+r1+b1+b2-4uy-ry+2by-2d', 3),
    ('u0-r0-3b0+2u1-r1-2b1+2u2-r2-2b2-3uy+3ry+2by', 2),
    ('2u0-3r0-3b0+u1+u2-uy+ry+2by+d', 3),
    ('-u0-5r0+b0-2u1-2r1-5r2+2b2+2uy+5ry-2by+5d', 3),
    ('-u0+r1-u2-r2-b2-ry-d', 0),
    ('-u0-b0+r1-b1+r2-b2-uy-ry+by-2d', 0),
    ('-u0-u1-r1-b1+b2-by', 0),
    ('-u0+u1-r1+b1+u2-r2+b2-3uy-by', 0),
    ('-u0-5r0+b0-5r1+2b1-2u2-2r2+2uy+5ry-2by+5d', 3),
    ('-2u0-2u1-r1-u2-r2-2b2+2uy+ry+d', 1),
    ('-3u0-u1-b1+r2-3b2+3uy-3ry+2by', 3),
    ('b0-u1-b1-u2-b2-by', 0),
    ('4u0+2r0-u1-b1+2u2+r2-2b2-4uy-5ry+4by-5d', 4),
    ('3u0-3r0+b0+u1-r1+b1-r2+2b2-uy+7ry-5by+6d', 6),
    ('u0-b0+3r1+b1+3r2+b2-uy-4ry+7by-4d', 8),
    ('u1-r1-b1+ry+d', 1),
    ('r0-r1+b1-r2+b2-uy-ry-d', 1),
    ('r1+b2+ry+d', 2),
    ('b1+b2-2uy-by', 0),
    ('b1+r2+ry+d', 2),
    ('uy-by+d', 1),
    ('u0-uy-d', 0),
    ('u0+b0-uy', 1),
]


def parse_expr(expr):
    'Function parse expr.'
    terms = expr.replace('+', ' +').replace('-', ' -').split()
    idxs, coefs = [], []
    for t in terms:
        if not t:
            continue
        sign = 1
        if t[0] in '+-':
            if t[0] == '-':
                sign = -1
            t = t[1:]
        num = ''
        while t and t[0].isdigit():
            num += t[0]
            t = t[1:]
        coef = sign * (int(num) if num else 1)
        idxs.append(IDX[t])
        coefs.append(coef)
    return idxs, coefs


PARSED = [(parse_expr(e), rhs) for e, rhs in CONSTRAINTS]


def xand_check(vals):
    'Function xand check.'
    for (idxs, coefs), rhs in PARSED:
        s = sum(c * vals[i] for i, c in zip(idxs, coefs))
        if s > rhs:
            return False
    return True


def main():

    n_unique_out = 0
    n_multi_out = 0
    n_d1 = 0
    lookup = {}
    for inp in itertools.product([0, 1], repeat=9):
        u0, r0, b0, u1, r1, b1, u2, r2, b2 = inp
        best_d = None
        outs = []
        for d in (0, 1):
            for uy, ry, by in itertools.product([0, 1], repeat=3):
                vals = (u0, r0, b0, u1, r1, b1, u2, r2, b2, uy, ry, by, d)
                if xand_check(vals):
                    if best_d is None or d < best_d:
                        best_d = d
                        outs = [(uy, ry, by)]
                    elif d == best_d:
                        outs.append((uy, ry, by))
        if best_d is None:
            raise RuntimeError(f'infeasible input: {inp}')
        lookup[inp] = (best_d, outs)
        if len(outs) == 1:
            n_unique_out += 1
        else:
            n_multi_out += 1
        if best_d == 1:
            n_d1 += 1

    print('XAND lookup: 512 input combinations')
    print(f'  unique output (min-d deterministic): {n_unique_out} ({n_unique_out/512*100:.1f}%)')
    print(f'  multi-output (propagation branches): {n_multi_out} ({n_multi_out/512*100:.1f}%)')
    print(f'  min delta=1 inputs: {n_d1} ({n_d1/512*100:.1f}%)')
    if n_multi_out:
        print('\nmulti-output samples (first 10):')
        cnt = 0
        for inp, (d, outs) in lookup.items():
            if len(outs) > 1:
                print(f'  input {inp} -> d={d}, outputs {len(outs)}: {outs[:6]}')
                cnt += 1
                if cnt >= 10:
                    break

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'validation', 'xand_lookup.npz'),
             inputs=np.array(list(lookup.keys()), dtype=np.uint8),
             min_d=np.array([v[0] for v in lookup.values()], dtype=np.uint8),
             n_out=np.array([len(v[1]) for v in lookup.values()], dtype=np.uint8),
             out0=np.array([v[1][0] for v in lookup.values()], dtype=np.uint8))
    print('\nlookup saved (first output only; multi-output cases need branching)')


if __name__ == '__main__':
    main()
