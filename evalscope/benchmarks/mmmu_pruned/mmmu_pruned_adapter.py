# flake8: noqa: E501
"""
MMMU pruned adapter — two pruning strategies:

1. discriminative_stratified (default): prunes the 660-sample reference set
   using difficulty/discrimination from the glm-4.5v-fp8 reference model.

2. encoder_probe: targets the FULL ~12K MMMU HuggingFace dataset (MMMU/MMMU),
   selecting samples that specifically stress the image encoder rather than
   general reasoning. Designed to surface FP8 image-encoder degradation.

Usage:
    # Reference-set pruning (660 → ~200 samples)
    evalscope eval --model <model> --datasets mmmu_pruned \\
        --dataset-args '{"prune_ratio": 0.3}'

    # Full-set encoder probe (~12K → ~300 samples)
    evalscope eval --model <model> --datasets mmmu_pruned \\
        --dataset-args '{"pruning_strategy": "encoder_probe", "prune_ratio": 0.025}'
"""
import math
from typing import Any, Dict, Optional, Union

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter as MmmuAdapter
from evalscope.constants import Tags
from evalscope.pruning import DiscriminativeStratifiedPruner, EncoderProbeSelector, load_reference_scores
from evalscope.utils.logger import get_logger

logger = get_logger()

MMMU_SUBJECTS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials',
]


def _extract_subject_from_id(record_id: str) -> str:
    """
    Extract subject name from a MMMU HuggingFace record id.

    Record id format: "{split}_{Subject}_{index}"
    Examples:
        "validation_Accounting_1"            → "Accounting"
        "validation_Architecture_and_Engineering_5" → "Architecture_and_Engineering"
    """
    parts = record_id.split('_')
    if len(parts) < 3:
        return 'unknown'
    # Everything between the first part (split name) and the last part (index)
    return '_'.join(parts[1:-1])


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Pruned)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

A pruned subset of MMMU with two strategies:

### Strategy 1: discriminative_stratified (default)
Prunes the 660-sample reference set (22 subjects × 30) using DSS. Covers all
subjects with difficulty-stratified selection. Uses pre-computed reference scores.

### Strategy 2: encoder_probe (Part B)
Scans the FULL ~12K MMMU HuggingFace dataset (MMMU/MMMU, validation split),
selecting samples that specifically stress the image encoder. Designed to surface
degradation from FP8 quantization of vision encoders.

**What FP8 encoder quantization degrades:**
- Fine-grained spatial feature extraction (charts, circuit diagrams)
- Precise value reading from graphs/tables
- Color and texture discrimination

**Selection criteria (encoder_probe):**
- Subject-level image-necessity weights (Electronics=1.0, Accounting=0.2)
- Keyword boost (+0.2) for questions mentioning chart/graph/diagram/figure/circuit
- All subjects covered proportionally, with image-heavy bias
- Selects by image-necessity rank — not by difficulty or model score

## Parameters (via --dataset-args)

- `pruning_strategy`: "discriminative_stratified" (default) or "encoder_probe"
- `prune_ratio`: Fraction of samples to keep per subject (default: 0.3)
- `per_subject_cap`: Max samples per subject for encoder_probe (default: 20)

## Examples

```bash
# DSS on 660 reference samples
evalscope eval --model <model> --datasets mmmu_pruned \\
    --dataset-args '{"prune_ratio": 0.3}'

# Image encoder probe on full ~12K HuggingFace dataset
evalscope eval --model <model> --datasets mmmu_pruned \\
    --dataset-args '{"pruning_strategy": "encoder_probe", "prune_ratio": 0.025}'
```
""",
        dataset_id='MMMU/MMMU',
        subset_list=MMMU_SUBJECTS,
        metric_list=['acc'],
        eval_split='validation',
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': '"discriminative_stratified" (default) or "encoder_probe" (Part B).',
                'value': 'discriminative_stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep per subject.',
                'value': 0.3,
            },
            'per_subject_cap': {
                'type': 'int | null',
                'description': 'Max samples per subject for encoder_probe. Null = no cap.',
                'value': 20,
            },
        },
    )
)
class MmmuPrunedAdapter(MmmuAdapter):
    """
    MMMU with DSS pruning or image-encoder probe selection.

    Inherits all multimodal prompt formatting, image loading, MCQ scoring,
    and answer extraction from MmmuAdapter. Adds per-subject index filtering
    via sample_filter().

    For encoder_probe: scans the full MMMU/MMMU HuggingFace dataset at init
    time to score every record by image-necessity, then filters to top-K per
    subject during the standard evalscope eval loop.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._strategy = self.extra_params.get('pruning_strategy', 'discriminative_stratified')
        self._prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
        self._per_subject_cap: Optional[int] = self.extra_params.get('per_subject_cap', 20)

        if self._strategy not in ('discriminative_stratified', 'encoder_probe'):
            raise ValueError(
                f'mmmu_pruned: unknown strategy "{self._strategy}". '
                f'Choose "discriminative_stratified" or "encoder_probe".'
            )

        # pruned_by_subject: subject → frozenset of 0-based sequential indices to keep
        self._pruned_by_subject: dict[str, frozenset[int]] = {}
        # per-subject counter incremented in record_to_sample to assign stable indices
        self._subject_counters: dict[str, int] = {}

        if self._strategy == 'discriminative_stratified':
            self._setup_dss()
        else:
            self._setup_encoder_probe()

    # ------------------------------------------------------------------
    # Setup methods
    # ------------------------------------------------------------------

    def _setup_dss(self):
        """Select indices from the 660 reference samples using DSS."""
        mmmu_scores = load_reference_scores('mmmu')
        pruner = DiscriminativeStratifiedPruner()

        total_selected = 0
        for subject, subject_scores in mmmu_scores.items():
            indices = pruner.select_indices(subject_scores, self._prune_ratio)
            self._pruned_by_subject[subject] = frozenset(indices)
            total_selected += len(indices)

        logger.info(
            f'[mmmu_pruned] strategy=discriminative_stratified '
            f'prune_ratio={self._prune_ratio} '
            f'selected={total_selected}/{sum(len(v) for v in mmmu_scores.values())} samples '
            f'across {len(self._pruned_by_subject)} subjects'
        )

    def _setup_encoder_probe(self):
        """
        Scan the FULL MMMU/MMMU HuggingFace validation split and select
        image-encoder-sensitive samples per subject.

        Loads each subject's validation records, scores them by image necessity
        (subject weight + question keyword boost), and takes the top-K per subject.
        Falls back to the 660 reference samples if HuggingFace is unavailable.
        """
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            logger.warning(
                '[mmmu_pruned] encoder_probe: `datasets` library not found. '
                'Falling back to 660 reference samples. '
                'Install with: pip install datasets'
            )
            self._setup_encoder_probe_from_reference()
            return

        selector = EncoderProbeSelector()
        total_selected = 0
        total_scanned = 0

        logger.info(
            f'[mmmu_pruned] encoder_probe: scanning full MMMU/MMMU HuggingFace dataset '
            f'(validation split, {len(MMMU_SUBJECTS)} subjects)...'
        )

        for subject in MMMU_SUBJECTS:
            try:
                # Load just this subject's validation split
                # HuggingFace caches after first download
                ds = hf_load_dataset('MMMU/MMMU', subject, split='validation', trust_remote_code=True)
                n = len(ds)
                total_scanned += n

                # Compute image necessity for every record in this subject
                # Uses question text + subject weight — no inference required
                scored: list[tuple[int, float]] = []
                for i, record in enumerate(ds):
                    q_text = record.get('question', '')
                    necessity = selector.compute_image_necessity(subject, q_text)
                    scored.append((i, necessity))

                # Sort by necessity descending, take top quota
                scored.sort(key=lambda x: -x[1])
                quota = max(1, math.floor(self._prune_ratio * n))
                if self._per_subject_cap is not None:
                    quota = min(quota, self._per_subject_cap)

                selected_indices = frozenset(i for i, _ in scored[:quota])
                self._pruned_by_subject[subject] = selected_indices
                total_selected += len(selected_indices)

                logger.debug(
                    f'  {subject}: {len(selected_indices)}/{n} samples selected '
                    f'(top necessity: {scored[0][1]:.2f})'
                )

            except Exception as e:
                logger.warning(
                    f'[mmmu_pruned] encoder_probe: failed to load subject "{subject}": {e}. '
                    f'Skipping subject.'
                )

        logger.info(
            f'[mmmu_pruned] encoder_probe complete: '
            f'selected={total_selected} from {total_scanned} scanned samples '
            f'across {len(self._pruned_by_subject)} subjects '
            f'(targets image-encoder-sensitive questions)'
        )

    def _setup_encoder_probe_from_reference(self):
        """Fallback: use 660 reference samples when HF is unavailable."""
        mmmu_scores = load_reference_scores('mmmu')
        selector = EncoderProbeSelector()
        selected_by_subject = selector.select_from_reference_data(mmmu_scores, self._prune_ratio)

        total_selected = 0
        for subject, indices in selected_by_subject.items():
            self._pruned_by_subject[subject] = frozenset(indices)
            total_selected += len(indices)

        logger.info(
            f'[mmmu_pruned] encoder_probe (reference fallback): '
            f'selected={total_selected} samples across {len(selected_by_subject)} subjects'
        )

    # ------------------------------------------------------------------
    # evalscope hooks
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, list]:
        """
        Assign a stable sequential index per subject, then delegate to parent.

        The index assigned here corresponds to the record's position in the
        HuggingFace dataset for that subject (0, 1, 2, ...), matching the
        indices computed in _setup_dss() and _setup_encoder_probe().
        """
        # Extract subject from the HuggingFace record id
        # Format: "{split}_{Subject}_{index}" e.g. "validation_Accounting_1"
        record_id = record.get('id', '')
        subject = _extract_subject_from_id(record_id) if record_id else 'unknown'

        if subject not in self._subject_counters:
            self._subject_counters[subject] = 0

        result = super().record_to_sample(record)
        samples = result if isinstance(result, list) else [result]
        for sample in samples:
            if sample is not None:
                sample.id = self._subject_counters[subject]
                if sample.metadata is None:
                    sample.metadata = {}
                # Store subject for use in sample_filter
                sample.metadata['_pruning_subject'] = subject

        self._subject_counters[subject] += 1
        return result

    def sample_filter(self, sample: Sample) -> bool:
        """Keep only samples whose sequential subject-index is in the pruned set."""
        if not super().sample_filter(sample):
            return False

        subject = (sample.metadata or {}).get('_pruning_subject', '')
        if not subject or subject not in self._pruned_by_subject:
            return True  # unknown subject: pass through

        return sample.id in self._pruned_by_subject[subject]
