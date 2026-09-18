# better_attack - rule-embedded MILP attack search

This tree is **self-contained**: it imports nothing from `data_gen/`, `train/` or `distill/`, only
the rule tables, which are copied into the drivers as literals (and checked by
`tools/verify_embedded_rules.py`).

```
base_MILP/   MILP model of the permutations
output/      writers that dump a solved model's states into the characteristic files
attack/      one directory per target; each target has a stage-1 and a stage-2 driver
```

## How to run

```bash
set PYTHONPATH=better_attack        # Windows (bash: export PYTHONPATH=better_attack)
python better_attack/attack/Keccak/SHA3_384/4-round/stage1_blue_search_distill.py
python better_attack/attack/Keccak/SHA3_384/4-round/stage2_red_search_distill.py
```

* Drivers are **scripts**: they execute at import time, take no CLI arguments, and expose their
  knobs (`EPSILON`, `NUM_ROUNDS`, phase time limits, rule table) as top-of-file constants.
* **Stage 1 must run before stage 2**: it writes
  `attack/<target>/blue_result/<library>.py`, which the stage-2 driver imports.
* Outputs: `blue_result/`, `red_result/`, `final_result/` under each target directory (gitignored).

## Targets

| Target | Driver | Model | epsilon |
|---|---|---|---|
| SHA3-384, 4-round preimage | `Keccak/SHA3_384/4-round/stage{1,2}_*` | `Keccak_MILP_64` (64 slices) | 0.02 |
| SHA3-512, 4-round preimage | `Keccak/SHA3_512/4-round/stage{1,2}_*` | `Keccak_MILP_64` (64 slices) | 0.02 |
| Ascon-XOF, 3-round preimage | `Ascon/AsconXOF/Ascon_XOF_3_preimage_distill.py` | `Ascon_MILP_64` | 0.001 |

## How the rules enter the model

Each driver carries a `RULES_DISTILL` list `(name, sign, weight, max_val)` taken from
`distill/rules/*.json`, renormalised so that `sum |w| = 1`, and builds

```
H(X) = sum_i w_i * f_i(X) / max_val_i          (SHA-3)
H(X) = sum_i w_i * f_i(X)                      (Ascon; see the paper's implementation-scaling note)
```

with the primary objective `F(X)` (`temp_degree` for the red search, the diffusion count for the
blue search). The augmented objective is `F(X) + eps * H(X)`, and the Ascon drivers additionally add
the small column-clear reward described in the paper.
