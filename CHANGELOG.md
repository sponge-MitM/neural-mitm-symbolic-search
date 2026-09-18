# Changelog

## v1.0.0 - initial public release

Contents: the four-stage pipeline of the paper for SHA-3 (Keccak) and Ascon - training-data
generation, quality-predictor training, rule distillation, and the rule-embedded MILP attack
search - together with the proxy-validation sweep and the rule-table checker.

Conventions used throughout this release:

* the symbolic state covers all 64 z-slices for both primitives;
* 1,000,000 samples per training set and 200,000 samples per stage-B evaluation pool;
* the rule tables in `distill/rules/` are the single source of truth and are checked against the
  embedded tables of the attack drivers by `tools/verify_embedded_rules.py`;
* sources are ASCII-only with LF line endings.

Not part of this release (see `docs/STRUCTURE.md`): the Ascon-Hash collision driver, a complexity
calculator for the time/memory exponents, the linear-cancellation accounting and the Stage-1
blue-scheme dataset generator. Training datasets and model checkpoints are not distributed.
