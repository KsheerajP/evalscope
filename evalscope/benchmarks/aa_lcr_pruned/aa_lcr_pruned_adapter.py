# flake8: noqa: E501
"""
AA-LCR pruned adapter — discriminative stratified subset of aa_lcr.

Usage:
    evalscope eval --model <model> --datasets aa_lcr_pruned \\
        --dataset-args '{"prune_ratio": 0.3}'

prune_ratio=0.3 → 30 samples from 100 (3× faster, same signal).

AA-LCR note: scored by an LLM judge, which is non-deterministic.
The pruner applies noise_correction=True to down-weight zero-discrimination
samples more aggressively — LLM-judge agreement is stronger signal than
agreement under deterministic scoring.
"""
from typing import Any, Dict, Union

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter as AaLcrAdapter
from evalscope.constants import Tags
from evalscope.pruning import DiscriminativeStratifiedPruner, load_reference_scores
from evalscope.utils.logger import get_logger

logger = get_logger()

try:
    _PARENT_PROMPT = AaLcrAdapter._benchmark_meta.prompt_template
except Exception:
    _PARENT_PROMPT = '\nBEGIN INPUT DOCUMENTS\n\n{documents_text}\n\nEND INPUT DOCUMENTS\n\nAnswer the following question using the input documents provided above.\n\nSTART QUESTION\n\n{question}\n\nEND QUESTION\n'


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

A discriminatively pruned subset of AA-LCR (Artificial Analysis Long Context Retrieval),
selecting the most signal-dense questions at a fraction of evaluation cost.

## Pruning Strategy: Discriminative Stratified Sampling (DSS) with Noise Correction

AA-LCR uses an LLM judge, introducing non-deterministic variance. The DSS pruner
applies a noise correction: zero-discrimination items (where all models agree) are
weighted by extreme difficulty rather than treated as non-discriminative, since
consistent results under a noisy judge are a stronger signal.

## Parameters (via --dataset-args)

- `pruning_strategy`: "discriminative_stratified" (default)
- `prune_ratio`: Fraction of samples to keep (default: 0.3)

## Example

```bash
evalscope eval --model <model> --datasets aa_lcr_pruned \\
    --dataset-args '{"prune_ratio": 0.3}'
```
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=_PARENT_PROMPT,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy. Only "discriminative_stratified" supported.',
                'value': 'discriminative_stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (0 < prune_ratio ≤ 1).',
                'value': 0.3,
            },
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files.',
                'value': None,
            },
        },
    )
)
class AaLcrPrunedAdapter(AaLcrAdapter):
    """
    AA-LCR with Discriminative Stratified Sampling pruning and judge-noise correction.

    Inherits all document loading, judge prompting, and scoring from AaLcrAdapter.
    Adds index-based filtering via pre-computed DSS selection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
        strategy = self.extra_params.get('pruning_strategy', 'discriminative_stratified')

        if strategy != 'discriminative_stratified':
            raise ValueError(
                f'aa_lcr_pruned only supports strategy="discriminative_stratified", got "{strategy}"'
            )

        ref_scores = load_reference_scores('aa_lcr')
        pruner = DiscriminativeStratifiedPruner()
        # noise_correction=True: AA-LCR uses LLM judge (non-deterministic)
        self._pruned_indices: frozenset[int] = frozenset(
            pruner.select_indices(ref_scores, prune_ratio, noise_correction=True)
        )
        self._sample_counter: int = 0

        summary = pruner.summary(ref_scores, prune_ratio)
        logger.info(
            f'[aa_lcr_pruned] strategy={strategy} prune_ratio={prune_ratio} '
            f'selected={summary["selected"]}/{summary["total"]} samples '
            f'(discrimination gain: {summary["discrimination_gain"]}×, noise_correction=True)'
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, list]:
        """Assign sequential index then delegate to parent."""
        result = super().record_to_sample(record)
        samples = result if isinstance(result, list) else [result]
        for sample in samples:
            if sample is not None:
                sample.id = self._sample_counter
        self._sample_counter += 1
        return result

    def sample_filter(self, sample: Sample) -> bool:
        """Keep only samples in the pruned set."""
        if not super().sample_filter(sample):
            return False
        return sample.id in self._pruned_indices
