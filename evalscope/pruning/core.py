"""
Universal benchmark pruning library for evalscope.

Implements Discriminative Stratified Sampling (DSS), an IRT-inspired approach
that selects the smallest sample set that preserves the full benchmark's go/no-go
signal. Used by live_code_bench_pruned, aa_lcr_pruned, and mmmu_pruned adapters.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

_REF_DATA_DIR = Path(__file__).parent / 'reference_data'

KNOWN_STRATEGIES = ('discriminative_stratified', 'encoder_probe')


def load_reference_scores(benchmark: str) -> dict:
    """
    Load pre-computed difficulty/discrimination scores for a benchmark.

    Args:
        benchmark: One of 'live_code_bench_v5', 'aa_lcr', 'mmmu'

    Returns:
        dict mapping str(index) → {difficulty, discrimination, ...}
        For mmmu: dict mapping subject → {str(index) → {...}}
    """
    fpath = _REF_DATA_DIR / f'{benchmark}.json'
    if not fpath.exists():
        raise FileNotFoundError(
            f'Reference scores not found at {fpath}. '
            f'Run scripts/precompute_reference_scores.py first.'
        )
    with open(fpath) as f:
        return json.load(f)


class DiscriminativeStratifiedPruner:
    """
    Discriminative Stratified Sampling (DSS) pruner.

    Selects samples that:
    1. Cover the full difficulty spectrum (hard / medium / easy strata)
    2. Within each stratum, prioritize samples where models disagree
       (high discrimination = structurally ambiguous → generalizes to new models)

    This is NOT:
    - Random sampling
    - Top-k hardest or easiest
    - Overfitted to the 3 reference models

    Generalizes to unseen models because:
    - Discriminative samples are intrinsically challenging to classify correctly
    - The difficulty strata ensure we see all kinds of questions
    - We use score *variance* (discrimination), not which model won
    """

    N_STRATA = 3
    STRATA_BOUNDS = [(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0 + 1e-9)]
    STRATA_NAMES = ['hard', 'medium', 'easy']

    def select_indices(
        self,
        scores: dict,
        prune_ratio: float,
        noise_correction: bool = False,
    ) -> list[int]:
        """
        Select a pruned subset of indices using DSS.

        Args:
            scores: {str(index): {difficulty: float, discrimination: float, ...}}
            prune_ratio: Fraction of samples to keep (0 < prune_ratio ≤ 1)
            noise_correction: If True, apply noise penalty for AA-LCR LLM-judge variance.
                              Down-weights zero-discrimination items more aggressively
                              (agreement under noisy judge is stronger signal than under
                              deterministic scoring).

        Returns:
            Sorted list of selected integer indices.
        """
        if not 0 < prune_ratio <= 1:
            raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')

        # Validate strategy
        items = list(scores.items())
        if not items:
            return []

        total = len(items)
        target_k = max(1, math.floor(prune_ratio * total))

        # Build strata
        strata: list[list[tuple[int, float, float]]] = [[] for _ in range(self.N_STRATA)]
        for idx_str, stats in items:
            idx = int(idx_str)
            difficulty = stats['difficulty']
            discrimination = stats['discrimination']

            # Noise correction for LLM-judge benchmarks (AA-LCR):
            # A sample where all models agree is more reliable than a high-discrimination
            # sample that might just be judge noise. We still include such samples but
            # rank them lower within their stratum.
            effective_disc = discrimination
            if noise_correction and discrimination == 0.0:
                # Give slight preference to extreme difficulties (very hard or very easy)
                # over mid-difficulty zero-discrimination samples
                effective_disc = -abs(difficulty - 0.5) * 0.01

            for s_idx, (lo, hi) in enumerate(self.STRATA_BOUNDS):
                if lo <= difficulty < hi:
                    strata[s_idx].append((idx, difficulty, effective_disc))
                    break
            else:
                # Edge case: difficulty == 1.0 lands in easy stratum
                strata[-1].append((idx, difficulty, effective_disc))

        # Proportional quota allocation per stratum
        selected: list[int] = []
        quotas = [max(1, round(target_k * len(s) / total)) for s in strata if s]
        # Trim to target (rounding may add up to target_k + 1)
        while sum(quotas) > target_k and quotas:
            quotas[quotas.index(max(quotas))] -= 1
        # Ensure at least 1 from each non-empty stratum
        for s_idx, stratum in enumerate(strata):
            if not stratum:
                continue
            # Sort: highest discrimination first, then most extreme difficulty as tiebreak
            stratum_sorted = sorted(stratum, key=lambda x: (-x[2], -abs(x[1] - 0.5)))
            quota = quotas.pop(0) if quotas else 1
            selected.extend(idx for idx, _, _ in stratum_sorted[:quota])

        # If rounding left us short, fill from highest-discrimination remaining
        if len(selected) < target_k:
            selected_set = set(selected)
            remaining = sorted(
                [(int(k), v['discrimination']) for k, v in scores.items() if int(k) not in selected_set],
                key=lambda x: -x[1]
            )
            for idx, _ in remaining[:target_k - len(selected)]:
                selected.append(idx)

        return sorted(set(selected))

    def summary(self, scores: dict, prune_ratio: float) -> dict:
        """Return a summary dict for logging/debugging."""
        indices = self.select_indices(scores, prune_ratio)
        total = len(scores)
        selected_scores = {k: scores[str(k)] for k in indices if str(k) in scores}
        avg_disc_full = sum(v['discrimination'] for v in scores.values()) / total if total else 0
        avg_disc_sel = sum(v['discrimination'] for v in selected_scores.values()) / len(selected_scores) if selected_scores else 0
        return {
            'total': total,
            'selected': len(indices),
            'prune_ratio': prune_ratio,
            'avg_discrimination_full': round(avg_disc_full, 4),
            'avg_discrimination_selected': round(avg_disc_sel, 4),
            'discrimination_gain': round(avg_disc_sel / avg_disc_full, 2) if avg_disc_full > 0 else None,
        }


class EncoderProbeSelector:
    """
    Image encoder probe selector for MMMU (Part B).

    Selects samples from the full MMMU dataset (~12K) that specifically
    stress the image encoder rather than general reasoning. Targets
    degradation patterns from FP8 quantization of vision encoders:
    - Precision loss in fine-grained spatial features
    - Chart/graph reading errors
    - Color/texture discrimination failures

    Used via: --dataset-args '{"pruning_strategy": "encoder_probe", "prune_ratio": 0.025}'
    """

    # Per-subject image necessity weights.
    # High = question requires seeing the image; low = text reasoning suffices.
    SUBJECT_WEIGHTS = {
        'Electronics': 1.00,
        'Architecture_and_Engineering': 0.95,
        'Mechanical_Engineering': 0.90,
        'Art': 0.90,
        'Design': 0.90,
        'Geography': 0.80,
        'Computer_Science': 0.75,
        'Materials': 0.70,
        'Energy_and_Power': 0.70,
        'Chemistry': 0.65,
        'Biology': 0.65,
        'Physics': 0.65,
        'Diagnostics_and_Laboratory_Medicine': 0.65,
        'Clinical_Medicine': 0.60,
        'Art_Theory': 0.60,
        'Basic_Medical_Science': 0.60,
        'Math': 0.55,
        'Music': 0.55,
        'Agriculture': 0.50,
        'Pharmacy': 0.45,
        'Psychology': 0.40,
        'Sociology': 0.40,
        'Economics': 0.35,
        'Manage': 0.35,
        'Marketing': 0.35,
        'Public_Health': 0.35,
        'History': 0.30,
        'Literature': 0.25,
        'Accounting': 0.20,
        'Finance': 0.20,
    }

    # Keywords in question text that signal image-dependent reasoning
    VISUAL_KEYWORDS = frozenset([
        'graph', 'chart', 'diagram', 'figure', 'table', 'plot',
        'image', 'picture', 'shown', 'depicted', 'illustrated',
        'circuit', 'schematic', 'map', 'cross-section', 'cross section',
        'legend', 'axis', 'curve', 'shape', 'color', 'colour',
    ])

    def compute_image_necessity(self, subject: str, question_text: str) -> float:
        """Score 0–1 for how much this sample requires image understanding."""
        base = self.SUBJECT_WEIGHTS.get(subject, 0.5)
        kw_boost = 0.2 if any(kw in question_text.lower() for kw in self.VISUAL_KEYWORDS) else 0.0
        return min(1.0, base + kw_boost)

    def select_from_hf_dataset(
        self,
        dataset,  # HuggingFace DatasetDict
        prune_ratio: float,
        per_subject_cap: Optional[int] = None,
    ) -> dict[str, list[int]]:
        """
        Select indices from a HuggingFace MMMU dataset.

        Args:
            dataset: HF DatasetDict with keys = subject names
            prune_ratio: Target fraction of full dataset to keep
            per_subject_cap: Max samples per subject (prevents subject imbalance)

        Returns:
            dict mapping subject → list of selected integer indices
        """
        selected: dict[str, list[int]] = {}

        for subject, split in dataset.items():
            n = len(split)
            quota = max(2, math.floor(prune_ratio * n))
            if per_subject_cap:
                quota = min(quota, per_subject_cap)

            scored = []
            for i, record in enumerate(split):
                q_text = ''
                for field in ('question', 'input', 'text'):
                    if field in record and record[field]:
                        q_text = str(record[field])
                        break
                necessity = self.compute_image_necessity(subject, q_text)
                scored.append((i, necessity))

            # Sort by image necessity descending; take top quota
            scored.sort(key=lambda x: -x[1])
            selected[subject] = [i for i, _ in scored[:quota]]

        return selected

    def select_from_reference_data(
        self,
        mmmu_scores: dict,
        prune_ratio: float,
    ) -> dict[str, list[int]]:
        """
        Select from the 660 reference samples using pre-computed image_necessity scores.

        Args:
            mmmu_scores: loaded from reference_data/mmmu.json
            prune_ratio: target fraction per subject

        Returns:
            dict mapping subject → list of selected integer indices
        """
        selected: dict[str, list[int]] = {}

        for subject, subject_scores in mmmu_scores.items():
            n = len(subject_scores)
            quota = max(1, math.floor(prune_ratio * n))

            scored = [
                (int(idx), stats.get('image_necessity', 0.5))
                for idx, stats in subject_scores.items()
            ]
            scored.sort(key=lambda x: -x[1])
            selected[subject] = [i for i, _ in scored[:quota]]

        return selected
