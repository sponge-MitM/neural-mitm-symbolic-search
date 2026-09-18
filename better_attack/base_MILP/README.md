# base_MILP - MILP model of the permutations

| Module | Models | Used by |
|---|---|---|
| `operation_MILP.py` | bit algebra: each bit carries `ul` (mixed/nonlinear), `r`, `b` flags; XOR/AND/OR encodings, pure-`u` indicators | the two models below |
| `Keccak_MILP_64.py` | Keccak-f[1600], full 64-slice state (theta, rho, pi, chi) | SHA3-384 and SHA3-512 drivers |
| `Ascon_MILP_64.py` | Ascon-p on all 64 z-slices | the Ascon-XOF drivers |

`operation_MILP.py` carries the encoding the paper describes: it separates linear from non-linear
dependence on the red variables, which is what makes the `ell'`-type reasoning of the 2025
framework expressible.
