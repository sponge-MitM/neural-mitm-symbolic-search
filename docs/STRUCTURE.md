# Repository structure

This repository implements the pipeline of the paper
*Neural Network-Guided Meet-in-the-Middle Attacks on Sponge-Based Hash Functions*.
It is organised as **four numbered stages plus one self-contained attack library**, so that the
scripts run in the order in which the paper describes them.

## Tree

```
.
|-- docs/
|   |-- STRUCTURE.md              # this file: what each file is, what is authoritative
|   `-- PIPELINE.md               # stage-by-stage commands, inputs and outputs
|-- data_gen/                     # STAGE 1 - training-data generation
|   |-- sha3_512/                 #   Keccak: stage-A (training set) + stage-B (evaluation pool)
|   |-- sha3_384/                 #   same for SHA3-384
|   `-- ascon/                    #   Ascon: single joint generation
|-- train/                        # STAGE 2 - quality-predictor (QP) training
|   |-- sha3_512/ sha3_384/       #   NN2 joint MLP/CNN (512-256-128, dropout 0.3)
|   `-- ascon/                    #   AsconMLP (216 scalar features, 256-128-64, dropout 0.2)
|-- distill/                      # STAGE 3 - rule distillation (ridge on QP soft labels)
|   |-- sha3_stage1_distill.py    #   SHA3-512 Stage-1 table   (NN1 blue-scheme predictor)
|   |-- sha3_384_stage1_distill.py#   SHA3-384 Stage-1 table   (NN1)
|   |-- sha3_distill.py           #   SHA3-512/384 Stage-2 tables (NN2 joint), alpha = 10
|   |-- ascon_distill.py          #   Ascon table (73 MILP-linear features), alpha = 5
|   |-- proxy_metrics.py          #   baseline vs rules vs QP proxy comparison
|   |-- proxy_metrics_sd.py       #   scheme-disjoint / grouped / intra-scheme protocols
|   `-- rules/*.json              #   distilled rule tables (the single source of truth)
|-- better_attack/                # STAGE 4 - rule-embedded MILP attack search
|   |-- README.md                 #   how this tree is meant to be used
|   |-- base_MILP/                #   MILP model of the permutations
|   |   |-- operation_MILP.py     #     bit algebra (ul/r/b flags, XOR/AND/OR encodings)
|   |   |-- Keccak_MILP_64.py     #     full 64-slice Keccak-f[1600] (both SHA-3 targets)
|   |   `-- Ascon_MILP_64.py      #     Ascon-p, all 64 z-slices
|   |-- output/                   #   state writers for the attack characteristics
|   `-- attack/                   #   one directory per target, stage 1 -> stage 2
|       |-- Keccak/SHA3_384/4-round/
|       |-- Keccak/SHA3_512/4-round/
|       `-- Ascon/AsconXOF/
|-- validation/                   # N_mix -> D* proxy validation (paper, Fig. 2)
|-- tools/
|   `-- verify_embedded_rules.py  # checks every attack script against distill/rules/*.json
|-- data/                         # generated artefacts (gitignored) + committed sweep results
|   `-- README.md
|-- README.md                     # overview, installation, pipeline summary
|-- requirements.txt
`-- LICENSE
```

## Which file is authoritative?

| Concern | Use |
|---|---|
| MILP bit algebra | `base_MILP/operation_MILP.py` |
| Keccak MILP model | `base_MILP/Keccak_MILP_64.py` (64 slices, both SHA-3 targets) |
| Ascon MILP model | `base_MILP/Ascon_MILP_64.py` |
| SHA-3 Stage-1 rules | `distill/sha3_stage1_distill.py`, `distill/sha3_384_stage1_distill.py` | `sha3_distill.py --stage 1` (refused; it would overwrite the NN1 tables) |
| SHA-3 Stage-2 rules | `distill/sha3_distill.py --stage 2` |
| Ascon rules | `distill/ascon_distill.py` |
| Attack drivers | `better_attack/attack/**/*_distill.py` |

## Generated artefacts (gitignored)

| Produced by | Lands in |
|---|---|
| `data_gen/**` | `data/<primitive>/*.npz` (`sha3_512_train_1000000.npz`, `sha3_512_train_stageB_200000.npz`, `sha3_384_train_1000000.npz`, `sha3_384_train_stageB_pi1.npz`, `ascon_features_v2_full.npz`, `ascon_features_milp_full.npz`, ...) |
| `train/**` | model checkpoints under `data/<primitive>/` |
| stage-1 attack script | `better_attack/attack/<target>/blue_result/` (the blue-scheme library imported by stage 2) |
| stage-2 attack script | `better_attack/attack/<target>/red_result/`, `.../final_result/` |
| `validation/run_nmix_scan.py` | `data/validation/nmix_scan_512*.npz` |

The only `data/` files committed to the repository are the four files under `data/validation/`
(see `data/README.md`).

## Conventions

* **Rule tables are the source of truth.** `distill/rules/*.json` hold the fitted weights and the
  per-feature `max_val`; the attack scripts embed the same numbers (renormalised) as literals, and
  `tools/verify_embedded_rules.py` checks that they still agree.
* **Attack drivers run at import time.** They are scripts, not libraries: constants such as
  `EPSILON`, `NUM_ROUNDS` and the phase time limits live at the top of each file, and the search
  starts as soon as the file is executed. Run them with `better_attack/` on `PYTHONPATH` (see
  `better_attack/README.md`).
* **Data-generation and training scripts import their siblings by name** (for example
  `from Keccak import *`, `from operation import Bit`), so they must be run from their own
  directory (or with that directory on `sys.path`).
* **64 slices for SHA3-384.** Both SHA3-384 attack stages use the full 64-slice Keccak-f[1600]
  model, matching the 1600-dimensional pi1 masks used for distillation.

## Known gaps (not implemented in this repository)

* The 3-round Ascon-Hash collision driver: only the collision helpers exist in
  `base_MILP/Ascon_MILP_64.py`; `better_attack/attack/Ascon/` ships the XOF preimage drivers only.
* A complexity calculator (the `2^{111}`, `2^{365}`, ... exponents of the paper) - the drivers stop
  at `temp_degree`.
* The linear-cancellation (`ell'`) accounting of the 2025 framework.
* A generator for the 8{,}000-blue-scheme Stage-1 datasets used by the SHA-3 NN1 predictors, and for
  the blue-scheme feature files (datasets are not distributed).
* An unguided-baseline driver for the "guided vs unguided" comparison.
