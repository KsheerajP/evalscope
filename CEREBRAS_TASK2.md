# Task 2 — Benchmark Compression for evalscope

**Cerebras AI Engineer Challenge — Model Quality**

> evalscope fork pinned at commit: `c14dbaf94e9129f7054ad4a184c2ff0cae2e6a5d`

---

## Overview

This fork of [modelscope/evalscope](https://github.com/modelscope/evalscope) adds three pruned benchmark adapters and a universal pruning library implementing **Discriminative Stratified Sampling (DSS)**.

| Benchmark | Full size | At 10% | Signal quality |
|-----------|-----------|--------|----------------|
| LiveCodeBench v5 | 315 samples | 31 samples | 2.84× more discriminative |
| AA-LCR | 100 samples | 30 samples | judge-noise corrected |
| MMMU (encoder probe) | ~12K samples | ~300 samples | image-encoder targeted |

---

## Install

```bash
git clone https://github.com/KsheerajP/evalscope
cd evalscope
pip install -e .
```

---

## Usage

### Step 1: Pre-compute reference scores (one-time setup)

```bash
python scripts/precompute_reference_scores.py \
    --evals-dir /path/to/challenge/Evals
```

Reads the challenge JSONL data and writes difficulty/discrimination scores to `evalscope/pruning/reference_data/`.

> **Note:** Reference data for the 3 shipped models is already bundled. Re-run only if adding new reference models.

### Step 2: Run full benchmark (baseline)

```bash
evalscope eval --model <model> --datasets live_code_bench --output ./results_full/
evalscope eval --model <model> --datasets aa_lcr --output ./results_full/
```

### Step 3: Run pruned benchmark

```bash
# LiveCodeBench — 10% (31 samples)
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "discriminative_stratified", "prune_ratio": 0.1}' \
    --output ./results_pruned/

# AA-LCR — 30% (30 samples)
evalscope eval --model <model> --datasets aa_lcr_pruned \
    --dataset-args '{"pruning_strategy": "discriminative_stratified", "prune_ratio": 0.3}' \
    --output ./results_pruned/

# MMMU — DSS on 660-sample reference set
evalscope eval --model <model> --datasets mmmu_pruned \
    --dataset-args '{"pruning_strategy": "discriminative_stratified", "prune_ratio": 0.3}' \
    --output ./results_pruned/

# MMMU — Image encoder probe on full ~12K HuggingFace dataset (Part B)
evalscope eval --model <model> --datasets mmmu_pruned \
    --dataset-args '{"pruning_strategy": "encoder_probe", "prune_ratio": 0.025}' \
    --output ./results_probe/
```

### Step 4: Compare full vs pruned

```bash
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ \
    --pruned ./results_pruned/ \
    --threshold 0.5
```

Output includes score delta, Spearman rank correlation, and go/no-go consistency verdict.

---

## Architecture

### `evalscope/pruning/` — Universal pruning library

| File | Purpose |
|------|---------|
| `core.py` | `DiscriminativeStratifiedPruner` (all 3 adapters), `EncoderProbeSelector` (MMMU Part B). Single source of truth for subject weights and visual keywords. |
| `reference_data/live_code_bench_v5.json` | Difficulty + discrimination for 315 LCB samples (3 models) |
| `reference_data/aa_lcr.json` | Scores + input_tokens for 100 AA-LCR samples |
| `reference_data/mmmu.json` | Per-subject scores + image_necessity for 660 MMMU reference samples |

### `evalscope/benchmarks/` — Three new adapters

All three inherit from their parent adapter (scoring, extraction, evaluation logic unchanged) and add index-based filtering via `sample_filter()`.

| Adapter | Parent | Strategy |
|---------|--------|----------|
| `live_code_bench_pruned` | `LiveCodeBenchAdapter` | DSS |
| `aa_lcr_pruned` | `AALCRAdapter` | DSS + LLM-judge noise correction |
| `mmmu_pruned` | `MMMUAdapter` | DSS or encoder_probe |

All three use the same `DiscriminativeStratifiedPruner.select_indices()` and accept identical `dataset-args`.

### `evalscope_ext/tools/compare_runs.py`

Fidelity comparison tool. Normalises benchmark names automatically (`live_code_bench_pruned` matches `live_code_bench`), reports score delta, Spearman ρ, and go/no-go verdict.

### `scripts/precompute_reference_scores.py`

One-time setup. Imports subject weights and visual keywords directly from `EncoderProbeSelector` in `core.py` — no duplicated constants.

---

## Handouts

- [`HANDOUT_A.md`](HANDOUT_A.md) — Technical explanation (1 page, engineer audience)
- [`HANDOUT_B.md`](HANDOUT_B.md) — Non-technical explanation (½ page, PM/customer audience)

---

## Pruning strategy rationale

**Discriminative Stratified Sampling (DSS):**
1. Partition samples into hard / medium / easy strata by difficulty
2. Within each stratum, rank by discrimination (models disagree = high signal)
3. Allocate budget proportionally across strata

Not random, not top-k, not hand-picked, not overfitted to the 3 reference models.
High-discrimination items are structurally ambiguous — a 4th unseen model will be
discriminated by the same structural features. See `HANDOUT_A.md` for full analysis.
