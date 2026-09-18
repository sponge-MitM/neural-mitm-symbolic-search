# Neural-Assisted Symbolic Search for Meet-in-the-Middle Cryptanalysis

This repository contains the code of the paper

> **A New Neural Network-Based Framework for Cryptanalysis: Application to Sponge Functions**

It covers the complete pipeline of the paper: **training-data generation -> quality-predictor (QP) training -> rule distillation -> rule-embedded MILP attack search**, for the sponge-based hash families studied in the paper: **SHA-3 (Keccak) and Ascon**.

**Where to look first.** `docs/PIPELINE.md` lists every stage with its commands, inputs and
outputs; `docs/STRUCTURE.md` maps each file and states which modules are authoritative.

## Core idea

Automated Meet-in-the-Middle (MitM) attack search on sponge-based hash functions is formulated as MILP optimization over *symbolic configurations*: each configuration declares which state bits are controlled by the red neutral set, by the blue neutral set, or are fixed (gray). The primary objective is the attack degree `D = min(d_R, d_B, m)`; however, `D` is expensive to evaluate, so the search is guided by a cheap proxy, the number of mixed bits `N_mix` after an intermediate number of rounds. The framework has four steps:

1. **Symbolic states** - each bit carries one of five labels (constant 0/1, unknown linear, red, blue); the linear layers of the permutation propagate red/blue as separate channels, and the nonlinear layer (chi/S-box) produces mixed bits.
2. **Quality predictor (QP)** - an MLP/CNN trained on solver-generated `N_mix` labels ranks symbolic configurations *without* solving the expensive MILP.
3. **Rule distillation** - a ridge regression is fitted to the QP's **soft labels** (the QP predictions, not the raw solver labels), restricted to features the MILP can express exactly (counts, AND indicators, differences, OR-linearized active-z counts). The result is a compact rule table (see `distill/rules/`).
4. **Rule-embedded search** - the rules enter the MILP as a bounded secondary objective `max F(X) + eps * H(X)` with `0 < eps < a` (a = minimal primary-objective gap), preserving the dominance guarantee while steering the search toward promising configurations.

**Two-stage structure.** For SHA-3 the search is two-stage and uses **two distinct quality predictors** (a "double-NN" design):

- **Stage 1 (blue search)** uses a *blue-scheme predictor* (NN1): an MLP on aggregate features of the blue-only propagation that predicts the minimal `N_mix` reachable in Stage 2 for a given blue distribution. Stage-1 rules are distilled from NN1's soft labels.
- **Stage 2 (red search)** uses the *joint predictor* (NN2): an MLP on the full red+blue configuration features. Stage-2 rules are distilled from NN2's soft labels.

Both SHA3-512 and SHA3-384 follow this double-NN design. Ascon uses a single joint predictor (no separate stages).

## Pipeline at a glance

| Stage | Directory | Produces |
|---|---|---|
| 1. training-data generation | `data_gen/` | `data/<primitive>/*.npz` (features + `N_mix` labels) |
| 2. quality-predictor training | `train/` | checkpoints under `data/<primitive>/` |
| 3. rule distillation | `distill/` | `distill/rules/*.json` |
| 4. rule-embedded attack search | `better_attack/` | attack results under `blue_result/`, `red_result/`, `final_result/` |
| proxy validation (Fig. 2) | `validation/` | `data/validation/*` |

Stage-by-stage commands: `docs/PIPELINE.md`. File-by-file map, including which modules are
authoritative: `docs/STRUCTURE.md`.

## Repository layout

```
```
```
|-- docs/                      # STRUCTURE.md (file map) + PIPELINE.md (stage-by-stage runs)
|-- data_gen/                  # STAGE 1 - training-data generation (MILP search + propagation)
|   |-- sha3_512/  sha3_384/   #   Keccak stage-A + stage-B generators and symbolic propagator
|   `-- ascon/                 #   Ascon generator and symbolic propagator
|-- train/                     # STAGE 2 - quality-predictor training (NN1 blue-scheme, NN2 joint)
|   |-- sha3_512/  sha3_384/   #   MLP/CNN for SHA-3 (512-256-128, dropout 0.3)
|   `-- ascon/                 #   AsconMLP (216 scalar features, 256-128-64, dropout 0.2)
|-- distill/                   # STAGE 3 - ridge rules fitted to QP soft labels
|   |-- sha3_stage1_distill.py #   SHA3-512 Stage-1 (NN1 blue-scheme)
|   |-- sha3_384_stage1_distill.py
|   |-- sha3_distill.py        #   SHA3-512/384 Stage-2 (NN2 joint), alpha = 10
|   |-- ascon_distill.py       #   Ascon (73 MILP-linear features), alpha = 5
|   |-- proxy_metrics.py       #   baseline vs rules vs QP proxy comparison
|   |-- proxy_metrics_sd.py    #   scheme-disjoint / grouped / intra-scheme protocols
|   `-- rules/                 #   rule tables (JSON) - the single source of truth
|-- better_attack/             # STAGE 4 - rule-embedded MILP attack search (see its README)
|   |-- base_MILP/             #   MILP model of the permutations (64-slice state)
|   |-- output/                #   writers that dump the characteristic states
|   `-- attack/                #   one directory per target, stage 1 -> stage 2
|       |-- Keccak/SHA3_384/4-round/
|       |-- Keccak/SHA3_512/4-round/
|       `-- Ascon/AsconXOF/
|-- validation/                # N_mix -> D* proxy validation (paper, Fig. 2)
|-- tools/verify_embedded_rules.py   # checks the drivers against distill/rules/*.json
|-- data/                      # generated artefacts (gitignored) + committed sweep results
|-- requirements.txt
`-- README.md
```


## Requirements

- Python >= 3.10
- `numpy`, `scipy`, `scikit-learn`
- `torch` (CUDA recommended for inference speed)
- `gurobipy` - Gurobi is a commercial solver; an academic license is available at
  <https://www.gurobi.com/academia/academic-program-and-licenses/>

Install with:

```bash
pip install -r requirements.txt
```

## Data and models

The training datasets and the trained quality-predictor checkpoints are **not** distributed in this repository because of their size. To reproduce the numbers reported in the paper you need the datasets produced by `data_gen/` and the checkpoints produced by `train/` (or contact the authors).

**Dataset sizes (as reported in the paper, Table 2):**

| Primitive | Stage | Sample count | Feature dim. |
|---|---|---|---|
| SHA3-512 | stage A | 1,000,000 | 3712 |
| SHA3-512 | stage B (evaluation pool) | 200,000 | 3712 |
| SHA3-384 | stage A | 1,000,000 | 3840 |
| SHA3-384 | stage B (evaluation pool) | 200,000 | 3840 |
| Ascon    | --- | 1,000,000 | 640+640 (masks) |

The table above reports the sample counts that the paper states and that the generator defaults now produce: 1,000,000 samples for each training set (SHA-3 stage A, Ascon) and 200,000 samples for each stage-B evaluation pool (SHA3-512, SHA3-384). The scripts run until the requested count is reached; because the last parallel batch can stop once the target is within tolerance, a stored file may hold a few rows less than requested.

The scripts write all artifacts under `data/<primitive>/` (gitignored). The QP for SHA-3 is an MLP with hidden widths 512-256-128 (dropout 0.3); the Ascon predictor uses 256-128-64 (dropout 0.2); the blue-scheme predictors use 256-128-64-32 (dropout 0.2). All networks use AdamW (lr 1e-3, weight decay 1e-4), a cosine-annealing schedule, batch size 128, at most 200 epochs with early stopping (patience 30) and gradient clipping at norm 5.0; the training objective is MSE on the regression head plus a pairwise margin ranking loss on the ranking head (weight 0.3 for SHA-3 and Ascon, margin 2 throughout).

## Reproducing the pipeline

### 1. Training-data generation

Each `data_gen/<primitive>/` script implements the two-stage (or single-stage) data generation of the paper: MILP search over symbolic configurations with a deterministic objective, where each solve excludes the configurations already returned for the same scheme so that the returned configurations are distinct near-optimal ones, followed by a deterministic two-round symbolic propagation that computes the label `N_mix` (the number of mixed bits after the second nonlinear layer) for that configuration. The MILP objective itself is the marked-bit count after the first round; `N_mix` is computed outside the solver.

```bash
# SHA3-512: stage-A blue library search, then stage-A and stage-B data
python data_gen/sha3_512/search_blue_bits.py
python data_gen/sha3_512/gen_training_data_512.py            # 1,000,000 samples (default)
python data_gen/sha3_512/gen_training_data_stageB_512.py     # 200,000 samples (default)

# SHA3-384
python data_gen/sha3_384/search_blue_bits.py
python data_gen/sha3_384/gen_training_data_384.py            # 1,000,000 samples (default)
python data_gen/sha3_384/gen_training_data_stageB.py

# Ascon (single joint search)
python data_gen/ascon/ascon_gen_training.py                  # 1,000,000 samples (default)

```

The default `--n-samples` values follow the dataset sizes reported in the paper; the last parallel batch may stop slightly early (see the table above). Output `.npz` files are written to `data/<primitive>/` and are gitignored.

### 2. Quality-predictor training

```bash
# SHA3-512 / SHA3-384: joint predictor (NN2) + blue-scheme predictor (NN1)
python train/sha3_512/train_stageB_512.py --model all
python train/sha3_512/sha3_train_blue.py
python train/sha3_384/train_stageB.py --model all
python train/sha3_384/sha3_train_blue.py

# Ascon
python train/ascon/ascon_train_full.py --mode full --model all

```

Checkpoints are saved under `data/<primitive>/` (gitignored).

### 3. Rule distillation (QP soft labels -> rules)

Each `distill/*.py` script loads the quality predictor, generates soft labels on the dataset (CUDA inference), fits a ridge regression **on the soft labels** (this is the only rule-extraction method used in the paper), and writes a rule table (JSON) with:

- `spearman_vs_soft`: Spearman correlation of the rule scores with the QP soft labels (distillation fidelity);
- `spearman_vs_raw`: Spearman correlation of the rule scores with the raw solver labels (proxy quality);
- the normalized rule weights `w_norm` and per-feature `max_val` used for normalization.

```bash
# Stage-1 tables (NN1 blue-scheme predictor)
python distill/sha3_stage1_distill.py
python distill/sha3_384_stage1_distill.py

# Stage-2 tables (NN2 joint predictor)
python distill/sha3_distill.py --variant 512 --stage 2
python distill/sha3_distill.py --variant 384 --stage 2

# Ascon (single joint predictor)
python distill/ascon_distill.py
```

All fits use ridge regression with regularization `alpha = 10.0` for SHA-3 and `5.0` for Ascon; all Spearman values are computed on the held-out test split. The generated JSON tables go to `distill/rules/`.

**Embedding notes** (the JSON tables are the full distilled tables; the embedded versions in the `better_attack/attack/` scripts apply the following adjustments):

- **SHA3-512 Stage 1**: features that are exact duplicates under the blue-only propagation (`n_blue` vs `init_blue_total`, `blue_x{k}_count` vs `init_blue_x{k}`) are removed before fitting; the JSON already reflects the deduplicated fit. The `active_z` features are excluded from the embedded rule set because they require OR-variable linearization; the embedded table is renormalized accordingly (13 rules).
- **Sign convention**: `sign = +1` / MINIMIZE means a larger feature value is associated with a larger predicted `N_mix`; `sign = -1` / MAXIMIZE the opposite. `w_norm = |w| / sum|w|`; `max_val` is the maximum absolute feature value on the test set, used for scaling in the embedding.

The rules are conditional associations learned from the quality-predictor soft labels and serve as surrogate-guided search biases; they do not constitute causal claims about the round functions (see the paper).

### 4. Proxy-metric comparison (paper, applications section)

The script `distill/proxy_metrics.py` compares, on the configuration pool of each primitive, the `N_mix` distribution of

- the **baseline** (all configurations produced by the plain MILP search),
- the **rule-ranked** top-k configurations (ranking by the distilled rule scores),
- the **QP-ranked** top-k configurations (ranking by the network soft labels).

```bash
python distill/proxy_metrics.py
```

For every primitive the mean `N_mix` decreases from the baseline pool to the rule-selected subset and further to the QP-selected subset, i.e. both the rules and the network concentrate the search on configurations with fewer mixed bits; the network is better than the linear rules (see the paper for the table and for the one case in which the selected-subset standard deviation stays above the baseline).

### 5. Validation of `N_mix` as a proxy for the attack degree (paper, applications section)

This directory validates that the indirect metric `N_mix` is predictive of the final
attack degree `D* = min(d_R, d_B, m)`. The controlled sweep builds intermediate states
with exactly `N_mix` pure mixed bits `(1,0,0)` at free positions; the remaining bits
follow measured type proportions. The back half (one additional round plus the 128
matching equations) is then solved with Gurobi, maximizing `D*`.

The `validation/` directory contains the following scripts:

| Script | Purpose |
|---|---|
| `xand_lookup.py` | Generate the XAND/chi lookup table used by the fast dependency-channel propagator. Run once before the other scripts. |
| `type_stats.py` | Compute type-distribution statistics over stage-B samples after two rounds and save the proportions used by `run_nmix_scan.py`. |
| `run_nmix_scan.py` | Run the `N_mix` -> `D*` MILP sweep. Each task solves one constructed intermediate state under a Gurobi time limit. |
| `plot_nmix_scan.py` | Merge all `nmix_scan_512*.npz` files, keep the best-known `D*` per `N_mix`, and plot the `N_mix` vs. `D*` curve. |
| `propagate_milp_style.py` | Internal propagation checks and fast NumPy propagation used by `type_stats.py`; not usually run directly. |

#### Standard workflow

```bash
# 1) generate the XAND lookup table (once)
python validation/xand_lookup.py

# 2) type proportions over stage-B samples (paper: 287/66/239/56/108/31/683)
python validation/type_stats.py

# 3) the N_mix sweep itself (120 s x 3 seeds x 1 case)
python validation/run_nmix_scan.py \
    --nmix 40,45,55,60,80,90,100,110,130,140,160,170,200,210,230,250 \
    --tl 120 --seeds 3 --cases 1 --procs 8

# 4) merge all scan files and plot the N_mix vs D* curve (paper, Fig. 2)
python validation/plot_nmix_scan.py
```

#### Main options for `run_nmix_scan.py`

| Option | Default | Description |
|---|---|---|
| `--nmix` | `20,40,60,80,100,130,160,200,250` | Comma-separated `N_mix` values to scan. |
| `--tl` | `120` | Gurobi time limit in seconds for each `(N_mix, case, seed)` solve. |
| `--seeds` | `3` | Number of Gurobi random seeds per `(N_mix, case)`; the best `D*` is kept. |
| `--cases` | `1` | Number of independent type-count cases per `N_mix`. When `>1`, cases are sampled from `data/validation/type_stats_relabel.npz` if available. |
| `--procs` | `8` | Number of parallel processes. |
| `--mip-gap` | `0.0` | Gurobi relative/absolute MIP gap tolerance; `0` requests optimality proof. |
| `--out` | `nmix_scan_512.npz` | Output file name under `data/validation/`. |
| `--fresh` | `False` | Ignore an existing `--out` file; otherwise repeated runs merge and keep the best-known `D*`. |

#### Producing a more robust curve

Using multiple independent type-count cases per `N_mix` and/or a larger per-solve
budget reduces the chance that a time-limited MILP search under-estimates `D*`.
Because repeated runs with the same `--out` name merge their best-known values, you
can also refine the curve incrementally:

```bash
# 5 independent cases x 3 Gurobi seeds = 15 searches per N_mix
python validation/run_nmix_scan.py \
    --nmix 40,45,55,60,80,90,100,110,130,140,160,170,200,210,230,250 \
    --tl 600 --seeds 3 --cases 5 --procs 8

# merge all result files and plot the updated curve
python validation/plot_nmix_scan.py
```

`plot_nmix_scan.py` keeps the largest `D*` per `N_mix` across all loaded scan files and
draws a direct line plot through the remaining best-known points. Outputs are written
under `data/validation/`, including `nmix_scan_merged.npz` and
`nmix_vs_D_curve.png` / `nmix_vs_D_curve.pdf`.

The sweep shows a clear decreasing trend: `D*` decreases from 41 at `N_mix = 40` to 2 at
`N_mix = 250`, directly supporting the use of `N_mix` as the training target of the
quality predictor.

### 6. Rule-embedded MILP attack search

The `*_distill.py` attack scripts are under `better_attack/attack/` and are run in the order **Stage 1 -> Stage 2** (Stage 1 writes the blue-scheme library consumed by Stage 2). The scripts import `base_MILP`, `output`, and `attack.*` as top-level packages, so run them with `better_attack/` on `PYTHONPATH`:

```bash
# from the repository root
PYTHONPATH=better_attack python better_attack/attack/Keccak/SHA3_384/4-round/stage1_blue_search_distill.py   # writes blue_result/
PYTHONPATH=better_attack python better_attack/attack/Keccak/SHA3_384/4-round/stage2_red_search_distill.py
PYTHONPATH=better_attack python better_attack/attack/Keccak/SHA3_512/4-round/stage1_blue_search_distill.py   # writes blue_result/
PYTHONPATH=better_attack python better_attack/attack/Keccak/SHA3_512/4-round/stage2_red_search_distill.py
PYTHONPATH=better_attack python better_attack/attack/Ascon/AsconXOF/Ascon_XOF_3_preimage_distill.py
PYTHONPATH=better_attack ```

Each script embeds the distilled rule table into the MILP objective:

```
Stage 1 (blue search):   minimize diffusion + eps * H(X)
Stage 2 (red search):    maximize temp_degree + eps * H(X)
Ascon (Phase 1/2):       maximize temp_degree + RIDGE_SCALE * H(X)
```

with the dominance guarantee `0 < eps < a` where `a` is the smallest positive gap of the primary objective. **Warning:** the MILP searches are extremely time-consuming (thousands of seconds per configuration on a multi-core machine); run them on a server.

Run `python tools/verify_embedded_rules.py` to check that the rule tables embedded in the attack scripts under `better_attack/attack/` match `distill/rules/*.json`.

## Paper correspondence

| Paper concept                              | Code                                          |
|--------------------------------------------|-----------------------------------------------|
| symbolic internal states / `N_mix`         | labels in the datasets, features in `distill` |
| quality predictor (QP)                     | `train/**` (MLP/CNN checkpoints not distributed) |
| rules distilled from QP soft labels        | `distill/*.py` -> `distill/rules/*.json`      |
| rule-embedded MILP search                  | `better_attack/attack/**/*_distill.py`        |
| dominance guarantee                        | `eps`/`RIDGE_SCALE` constants in the scripts  |
| dataset sizes (paper Table 2) | default `--n-samples` of `data_gen/**` |

## License

MIT (see LICENSE). The MILP modeling code under `better_attack/base_MILP/` derives from the reference implementation of the Meet-in-the-Middle framework for sponge-based hashing (Qin et al., EUROCRYPT 2023; Dong et al., CRYPTO 2024) and is redistributed for reproducibility.
