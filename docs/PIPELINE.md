# Pipeline: stage by stage

Every stage writes files that the next stage consumes. Sizes follow the paper:
**1,000,000 samples for each training set** (SHA-3 stage A, Ascon) and
**200,000 samples for each stage-B evaluation pool** (SHA3-512, SHA3-384).

```
stage 1  data_gen/     MILP search  ->  data/<primitive>/*.npz        (features, N_mix labels)
stage 2  train/        QP training  ->  data/<primitive>/*.pt         (NN1 blue-scheme, NN2 joint)
stage 3  distill/      ridge on QP soft labels -> distill/rules/*.json
stage 4  better_attack/  rule-embedded MILP search -> attack results (blue/red/final_result)
         validation/     N_mix -> D* proxy sweep   -> data/validation/*
```

## Stage 1 - training-data generation

```bash
# SHA3-512: stage-A blue library, then the 1M training set and the 200k evaluation pool
python data_gen/sha3_512/search_blue_bits.py
python data_gen/sha3_512/gen_training_data_512.py          # 1,000,000 samples (default)
python data_gen/sha3_512/gen_training_data_stageB_512.py   #   200,000 samples (default)

# SHA3-384
python data_gen/sha3_384/search_blue_bits.py
python data_gen/sha3_384/gen_training_data_384.py          # 1,000,000 samples (default)
python data_gen/sha3_384/gen_training_data_stageB.py       #   200,000 samples, pi1 features (default)

# Ascon (single joint generation)
python data_gen/ascon/ascon_gen_training.py                # 1,000,000 samples (default)
```

* The MILP objective is the **marked-bit count after the first round**; the stored label
  `N_mix` is computed afterwards by a deterministic two-round propagation
  (`deterministic_full_forward`) as the number of mixed bits after the second nonlinear layer.
* SHA-3 features are the red/blue masks of the initial state and of the state **after the second
  pi layer** (1600 positions per type). Stage-B SHA3-384 writes `sha3_384_train_stageB_pi1.npz`.

## Stage 2 - quality-predictor training

```bash
python train/sha3_512/train_stageB_512.py --model all   # NN2 joint (also the CNN ablation)
python train/sha3_512/sha3_train_blue.py                # NN1 blue-scheme
python train/sha3_384/train_stageB.py --model all
python train/sha3_384/sha3_train_blue.py
python train/ascon/ascon_train_full.py --mode full --model all
```

## Stage 3 - rule distillation

```bash
python distill/sha3_stage1_distill.py        # SHA3-512 Stage-1 (NN1 blue-scheme soft labels)
python distill/sha3_384_stage1_distill.py    # SHA3-384 Stage-1
python distill/sha3_distill.py --variant 512 --stage 2   # SHA3-512 Stage-2 (NN2)
python distill/sha3_distill.py --variant 384 --stage 2   # SHA3-384 Stage-2
python distill/ascon_distill.py              # Ascon (73 MILP-expressible features)

# proxy-metric tables of the paper
python distill/proxy_metrics.py
python distill/proxy_metrics_sd.py --target all --protocol grouped
```

Each script fits a ridge regression **on the QP soft labels** (never on the raw labels) on the
training split and reports the fidelity/proxy Spearman correlations on the held-out test split.
`sha3_distill.py --stage 1` is refused on purpose: the Stage-1 tables come from the two NN1
scripts above.

## Stage 4 - rule-embedded attack search

Run **stage 1 first** (it writes the blue-scheme library that stage 2 imports), from the repository
root, with `better_attack/` on `PYTHONPATH`:

```bash
export PYTHONPATH=better_attack            # Windows: set PYTHONPATH=better_attack

python better_attack/attack/Keccak/SHA3_384/4-round/stage1_blue_search_distill.py
python better_attack/attack/Keccak/SHA3_384/4-round/stage2_red_search_distill.py

python better_attack/attack/Keccak/SHA3_512/4-round/stage1_blue_search_distill.py
python better_attack/attack/Keccak/SHA3_512/4-round/stage2_red_search_distill.py

python better_attack/attack/Ascon/AsconXOF/Ascon_XOF_3_preimage_distill.py
```

Search results are written next to each driver under `blue_result/`, `red_result/` and
`final_result/`. Run `python tools/verify_embedded_rules.py` to confirm that the tables embedded
in the drivers still match `distill/rules/*.json`.

## Validation sweep (paper, Fig. 2)

```bash
python validation/xand_lookup.py          # once: XAND lookup table
python validation/type_stats.py           # type proportions over stage-B samples
python validation/run_nmix_scan.py --nmix 40,45,55,60,80,90,100,110,130,140,160,170,200,210,230,250 \
    --tl 120 --seeds 3 --cases 1 --procs 8
python validation/plot_nmix_scan.py       # merges scan files and draws N_mix vs D*
```
