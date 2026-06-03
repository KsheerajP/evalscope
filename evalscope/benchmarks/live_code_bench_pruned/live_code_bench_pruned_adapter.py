# flake8: noqa: E501
"""
LiveCodeBench pruned adapter — discriminative stratified subset of live_code_bench.

Usage:
    evalscope eval --model <model> --datasets live_code_bench_pruned \\
        --dataset-args '{"pruning_strategy": "discriminative_stratified", "prune_ratio": 0.1}'

prune_ratio=0.1 → 31 samples from 315 (10× faster, same go/no-go signal).
"""
from typing import Any, Dict, Union

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.pruning import DiscriminativeStratifiedPruner, load_reference_scores
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

A discriminatively pruned subset of LiveCodeBench v5, selecting samples that maximize
go/no-go signal fidelity at a fraction of the evaluation cost.

## Pruning Strategy: Discriminative Stratified Sampling (DSS)

Samples are selected to:
1. Cover the full difficulty spectrum (hard / medium / easy strata)
2. Prioritize items where models historically disagree (high discrimination)

This generalizes to unseen models: high-discrimination items are structurally
ambiguous — not artifacts of the 3 reference models used to compute scores.

## Parameters (via --dataset-args)

- `pruning_strategy`: "discriminative_stratified" (default and only option for LCB)
- `prune_ratio`: Fraction of samples to keep (default: 0.3; range: 0.05–1.0)

## Example

```bash
evalscope eval --model <model> --datasets live_code_bench_pruned \\
    --dataset-args '{"prune_ratio": 0.1}'
```
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template='### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy. Only "discriminative_stratified" supported for LCB.',
                'value': 'discriminative_stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (0 < prune_ratio ≤ 1).',
                'value': 0.3,
            },
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems starting from this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {'shell_executor': {}, 'python_executor': {}},
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """
    LiveCodeBench with Discriminative Stratified Sampling pruning.

    Inherits all scoring, extraction, and evaluation logic from LiveCodeBenchAdapter.
    Adds sample filtering via a sequential index counter matched against pre-computed
    pruned indices derived from reference scores across 3 models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
        strategy = self.extra_params.get('pruning_strategy', 'discriminative_stratified')

        if strategy != 'discriminative_stratified':
            raise ValueError(
                f'live_code_bench_pruned only supports strategy="discriminative_stratified", got "{strategy}"'
            )

        ref_scores = load_reference_scores('live_code_bench_v5')
        pruner = DiscriminativeStratifiedPruner()
        self._pruned_indices: frozenset[int] = frozenset(
            pruner.select_indices(ref_scores, prune_ratio)
        )
        self._sample_counter: int = 0

        summary = pruner.summary(ref_scores, prune_ratio)
        logger.info(
            f'[live_code_bench_pruned] strategy={strategy} prune_ratio={prune_ratio} '
            f'selected={summary["selected"]}/{summary["total"]} samples '
            f'(discrimination gain: {summary["discrimination_gain"]}×)'
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, list]:
        """Assign a sequential index then delegate to parent."""
        result = super().record_to_sample(record)
        samples = result if isinstance(result, list) else [result]
        for sample in samples:
            if sample is not None:
                sample.id = self._sample_counter
        self._sample_counter += 1
        return result

    def sample_filter(self, sample: Sample) -> bool:
        """Keep only samples whose sequential index is in the pruned set."""
        if not super().sample_filter(sample):
            return False
        return sample.id in self._pruned_indices
