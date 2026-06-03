# Task 2 — Benchmark Compression for evalscope

**Cerebras AI Engineer Challenge — Model Quality**

> evalscope fork pinned at commit: `c14dbaf94e9129f7054ad4a184c2ff0cae2e6a5d`

---

## Overview

This fork of [modelscope/evalscope](https://github.com/modelscope/evalscope) adds three pruned benchmark adapters and a universal pruning library implementing **Discriminative Stratified Sampling (DSS)**.

| Benchmark | Full size | At 10% | Discrimination gain |
|-----------|-----------|--------|---------------------|
| LiveCodeBench v5 | 315 samples | 31 samples | 2.84× |
| AA-LCR | 100 samples | 10 samples | — |
| MMMU (encoder probe) | ~12K samples | ~300 samples | image-necessity filtered |

---

## Install

```bash
git clone https://github.com/[your-repo]/evalscope
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

This reads the challenge JSONL data and writes pre-computed difficulty/discrimination scores to `evalscope/pruning/reference_data/`.

> **Note:** Reference data for the 3 shipped models is already bundled in the repo. You only need to re-run this if you add new reference models.

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

# MMMU — Image encoder probe on full 12K HuggingFace dataset (Part B)
evalscope eval --model <model> --datasets mmmu_pruned \
    --dataset-args '{"pruning_strategy": "encoder_probe", "prune_ratio": 0.025}' \
    --output ./results_pruned_probe/
```

### Step 4: Compare full vs pruned

```bash
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ \
    --pruned ./results_pruned/ \
    --threshold 0.5
```

---

## What's new in this fork

### `evalscope/pruning/` — Universal pruning library

| File | What it does |
|------|-------------|
| `core.py` | `DiscriminativeStratifiedPruner` (used by all 3 adapters), `EncoderProbeSelector` (MMMU Part B) |
| `reference_data/live_code_bench_v5.json` | Pre-computed difficulty + discrimination for 315 LCB samples |
| `reference_data/aa_lcr.json` | Pre-computed scores for 100 AA-LCR samples + input_tokens |
| `reference_data/mmmu.json` | Per-subject scores + image_necessity for 660 MMMU reference samples |

### `evalscope/benchmarks/` — Three new adapters

All three inherit from their parent adapter (inheriting scoring, extraction, and evaluation logic) and add index-based filtering via `sample_filter()`.

| Adapter | Parent | Strategy |
|---------|--------|----------|
| `live_code_bench_pruned` | `LiveCodeBenchAdapter` | DSS |
| `aa_lcr_pruned` | `AALCRAdapter` | DSS + noise correction |
| `mmmu_pruned` | `MMMUAdapter` | DSS or encoder_probe |

The **universal** part: all three use the same `DiscriminativeStratifiedPruner.select_indices()` method and accept identical `dataset-args` parameters.

### `evalscope_ext/tools/compare_runs.py`

Fidelity comparison tool — validates that pruned score ≈ full score, reports Spearman rank correlation and go/no-go consistency.

### `scripts/precompute_reference_scores.py`

One-time script to compute reference data from challenge JSONL files.

---

## Handouts

- [`HANDOUT_A.md`](HANDOUT_A.md) — Technical explanation (1 page, engineer audience)
- [`HANDOUT_B.md`](HANDOUT_B.md) — Non-technical explanation (½ page, PM/customer audience)

---

## Pruning strategy rationale

**Discriminative Stratified Sampling (DSS)** selects samples that:
1. Cover the full difficulty spectrum (hard / medium / easy strata)
2. Within each stratum, prioritize items where models historically disagree

This is NOT random, NOT top-k hardest, NOT hand-picked, and does NOT overfit to the 3 reference models. High-discrimination items are structurally ambiguous — they will discriminate a 4th model for the same reason they discriminated the first three.

See `HANDOUT_A.md` for full mathematical justification and what-if analysis.
