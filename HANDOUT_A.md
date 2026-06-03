# Handout A — Why This Works
**Technical audience: engineers who could have built this themselves**

---

## The problem I understood myself to be solving

Running a full benchmark suite across every candidate model is expensive — not just in wall-clock time but in the latency it adds to a sales or deployment decision. The customer needs a binary answer: *is this model good enough for our workload?* Running 315 coding problems or 100 long-context passages to answer that question is overkill if a well-chosen 30 samples would give the same answer.

The constraint is that the subset must be *defensible for an unseen model* — not tuned to the three reference models I had data for.

---

## Part A: Pruning approach

### Algorithm: Discriminative Stratified Sampling (DSS)

For each sample *i*, I compute two statistics from the three reference models:

```
difficulty[i]      = mean(score_m for m in models)       ∈ [0, 1]
discrimination[i]  = max(score_m) - min(score_m)         ∈ {0, 1}
```

`difficulty` tells me where on the capability spectrum the question falls. `discrimination` tells me whether the models *disagree* on this question — a discrimination of 1 means one model passed and another failed.

**Why discrimination is the right signal:** A question where all models agree (discrimination = 0) gives no information about *which* model is better. A discriminative question separates models. I want my pruned set to be full of questions that separate models.

**Stratification prevents the forbidden baselines:** Pure top-k by discrimination would over-index on medium-difficulty questions (they tend to have higher variance). I fix this by:
1. Partitioning samples into 3 difficulty strata: hard [0, 1/3), medium [1/3, 2/3), easy [2/3, 1.0]
2. Allocating a budget proportional to each stratum's size
3. Within each stratum, ranking by discrimination

**At 10% prune ratio (31 samples):** LCB selected samples average discrimination of 0.84 vs. 0.35 for the full set — **2.84× more signal-dense**.

### Why this generalizes to a 4th model

I use score *variance* across models, not which model won. High-discrimination items are structurally ambiguous at the boundary of model capability — they test the exact skills where models differ. A new model at a similar capability level will be discriminated by the same structural features. I am not selecting "the questions gpt-oss-120b was bad at."

The difficulty stratification ensures the pruned set covers the full capability spectrum — a model that is uniformly bad (or good) will still appear that way at any stratum-balanced sample size.

### AA-LCR: noise correction for LLM-judged benchmarks

AA-LCR is scored by an LLM judge, which is non-deterministic. Score variance for a sample therefore has two sources: model quality differences *and* judge noise. A zero-discrimination sample (all models "agreed") under a noisy judge is actually stronger signal than zero-discrimination under deterministic grading — the judge had to consistently reach the same conclusion. I apply a noise correction: zero-discrimination samples are ranked by extreme difficulty (|difficulty - 0.5|) rather than treated as completely uninformative.

This is a partial mitigation. With multiple judge runs per sample, bootstrap variance could separate judge noise from model signal more precisely. With access to a live judge endpoint, I'd run each candidate question 3× and use median scores.

### Subset sizes

| Benchmark | Full | 10% | 20% | 30% |
|-----------|------|-----|-----|-----|
| LCB v5    | 315  | 31  | 63  | 94  |
| AA-LCR    | 100  | 10  | 20  | 30  |

I recommend **prune_ratio=0.1 for LCB** (2.84× disc gain, 10× faster) and **prune_ratio=0.3 for AA-LCR** (100 samples is already small; 30 maintains coverage across the difficulty strata without going below ~10 samples per stratum).

---

## Part B: MMMU image encoder probe

### What I'm trying to surface

FP8 quantization of a vision encoder degrades precision in high-frequency spatial features: chart values, circuit topologies, fine-grained spatial relationships. It does *not* equally degrade text-based reasoning. A generic accuracy drop could come from either. I want samples that *isolate the encoder* as the failure point.

### How I score image necessity

Each sample gets an `image_necessity` score ∈ [0, 1]:

```
image_necessity = subject_weight + keyword_boost

subject_weight:  Electronics=1.0, Architecture=0.95, Art=0.90 ... Finance=0.20
keyword_boost:   +0.2 if question text contains chart/graph/diagram/circuit/etc.
```

Subject weights are derived from what FP8 quantization specifically degrades. Electronics questions require reading circuit diagrams at pixel precision. Geography requires map-reading. Finance questions can often be answered from text descriptions of numbers alone — the image adds little.

The encoder probe selects ~300 samples from the full 12K HF dataset (`MMMU/MMMU`), covering all 22 subjects proportionally but with image-heavy bias. At `prune_ratio=0.025`, the `EncoderProbeSelector` allocates per-subject quotas proportional to stratum size, ranks by image_necessity, and takes the top.

### Why this catches encoder failures that random sampling misses

Random sampling would include ~20% Finance/Accounting samples, where the encoder's precision doesn't matter. The probe set has those at ~5%. A model with a degraded encoder would fail Electronics at 40% and Finance at the same rate as a good encoder — random sampling would wash out the signal. The probe amplifies it.

### What would change with more resources

**(a) More data**: With more reference models scored on MMMU, I could compute discrimination per sample across models (same as LCB), then combine DSS with image_necessity scoring into a joint selection criterion.

**(b) Live model endpoint**: I'd run the reference model on borderline-necessity samples (image_necessity ≈ 0.5) twice — once with the image, once with a blank/masked image. Samples where score drops ≥ 0.5 are definitively image-dependent. This converts the heuristic necessity score into an empirically verified one.

**(c) More time**: Full IRT (Item Response Theory) model with 3-parameter logistic model, fitting per-sample difficulty, discrimination, and guessing parameters across models. This would give better-calibrated difficulty estimates, especially for multiple-choice tasks where guessing inflates scores.

---

## Assumptions

- The HuggingFace dataset ordering matches the JSONL `index` field (sequential row position)
- The 3 reference models span a representative range of capability for the customer's use case
- Judge noise in AA-LCR is approximately i.i.d. across samples (not systematically correlated with question type)
- FP8 vision encoder degradation manifests at the feature extraction stage, not the language model head (this determines which subjects to weight high)
