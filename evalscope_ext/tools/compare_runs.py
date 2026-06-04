"""
Compare full vs pruned evalscope runs to measure pruning fidelity.

Usage:
    python -m evalscope_ext.tools.compare_runs \\
        --full  ./results_full/ \\
        --pruned ./results_pruned/

Output:
    - Score fidelity per model (|score_pruned - score_full|)
    - Spearman rank correlation (if multiple models)
    - Compression ratio
    - Go/no-go consistency (does pruned agree with full on pass/fail?)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Metric keys searched in order when extracting a score from a result file
_METRIC_KEYS = ('acc', 'pass@1', 'score', 'accuracy', 'mean')

_REPORT_NAMES = frozenset(('summary.json', 'results.json', 'report.json'))


def _normalize_benchmark(name: str) -> str:
    """Strip _pruned suffix so full and pruned benchmark names match."""
    return name.replace('_pruned', '').lower()


def _extract_score(data: dict) -> Optional[float]:
    """Search common metric locations in a result dict and return the first match."""
    for location in (data, data.get('metrics', {}), data.get('results', {})):
        if not isinstance(location, dict):
            continue
        for key in _METRIC_KEYS:
            if key in location:
                return float(location[key])
    return None


def _extract_scores(run_dir: Path) -> dict[str, dict[str, float]]:
    """
    Extract model → benchmark → score from a results directory.

    Handles evalscope's output structure:
      {run_dir}/{model_name}/{benchmark}/report.json
    or flat:
      {run_dir}/report.json

    Returns: {model: {benchmark: score}}
    """
    results: dict[str, dict[str, float]] = {}

    for root, _, files in os.walk(run_dir):
        root_path = Path(root)
        for fname in files:
            if not fname.endswith('.json'):
                continue
            # Only process known report filenames or files with 'report' in name
            if fname not in _REPORT_NAMES and 'report' not in fname.lower():
                continue
            fpath = root_path / fname
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(data, dict):
                continue

            model = data.get('model') or data.get('model_id') or _infer_model_from_path(fpath, run_dir)
            benchmark = data.get('benchmark') or data.get('dataset') or _infer_benchmark_from_path(fpath, run_dir)
            score = _extract_score(data)

            if model and benchmark and score is not None:
                results.setdefault(model, {})[benchmark] = score

    return results


def _infer_model_from_path(fpath: Path, base: Path) -> str:
    try:
        parts = fpath.relative_to(base).parts
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass
    return 'unknown_model'


def _infer_benchmark_from_path(fpath: Path, base: Path) -> str:
    try:
        parts = fpath.relative_to(base).parts
        if len(parts) >= 2:
            return parts[1]
        if len(parts) == 1:
            return parts[0].replace('.json', '')
    except ValueError:
        pass
    return 'unknown_benchmark'


def spearman_correlation(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Spearman rank correlation without scipy."""
    if len(x) != len(y) or len(x) < 2:
        return None

    def rank(arr: list[float]) -> list[float]:
        sorted_idx = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for r, idx in enumerate(sorted_idx):
            ranks[idx] = r + 1.0
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(rx)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = (sum((r - mean_rx) ** 2 for r in rx) * sum((r - mean_ry) ** 2 for r in ry)) ** 0.5
    return num / den if den > 0 else 0.0


def _fmt(v: float, width: int = 8) -> str:
    return f'{v:.4f}'.rjust(width)


def compare(full_dir: Path, pruned_dir: Path, threshold: float = 0.5) -> int:
    """
    Compare full vs pruned runs and print a fidelity report.

    Returns 0 if all go/no-go decisions are consistent, 1 otherwise.
    """
    print(f'\n{"="*64}')
    print('  Pruning Fidelity Report')
    print(f'  Full run:   {full_dir}')
    print(f'  Pruned run: {pruned_dir}')
    print(f'{"="*64}\n')

    full_scores = _extract_scores(full_dir)
    pruned_scores = _extract_scores(pruned_dir)

    if not full_scores:
        print(f'ERROR: No results found in {full_dir}')
        print('  Expected: {run_dir}/{model}/{benchmark}/report.json')
        return 1
    if not pruned_scores:
        print(f'ERROR: No results found in {pruned_dir}')
        return 1

    common_models = set(full_scores) & set(pruned_scores)
    if not common_models:
        print(f'WARNING: No common models between full {sorted(full_scores)} '
              f'and pruned {sorted(pruned_scores)}.')
        print('  Check that --model is the same in both eval commands.')
        return 1

    print(f'Models compared: {sorted(common_models)}\n')
    print(f'{"Model":<30} {"Full":>8} {"Pruned":>8} {"Delta":>8} {"Status":>10}')
    print('-' * 68)

    full_list, pruned_list = [], []
    all_pass = True

    for model in sorted(common_models):
        f_benchmarks = full_scores[model]
        p_benchmarks = pruned_scores[model]

        # Pair benchmarks by normalized name (strips _pruned suffix)
        pairs: list[tuple[str, str, str]] = [
            (fb, pb, _normalize_benchmark(fb))
            for fb in f_benchmarks
            for pb in p_benchmarks
            if _normalize_benchmark(fb) == _normalize_benchmark(pb) or fb == pb
        ]
        # Fallback: if single benchmark each, pair them regardless of name
        if not pairs and len(f_benchmarks) == 1 and len(p_benchmarks) == 1:
            fb, pb = list(f_benchmarks)[0], list(p_benchmarks)[0]
            pairs = [(fb, pb, _normalize_benchmark(fb))]

        for fb, pb, display in pairs:
            fs, ps = f_benchmarks[fb], p_benchmarks[pb]
            delta = abs(ps - fs)
            full_list.append(fs)
            pruned_list.append(ps)

            consistent = (fs >= threshold) == (ps >= threshold)
            status = '✓ consistent' if consistent else '✗ MISMATCH'
            if not consistent:
                all_pass = False

            label = f'{model[:20]}/{display[:10]}'
            print(f'{label:<30} {_fmt(fs)} {_fmt(ps)} {_fmt(delta)} {status:>12}')

    print()

    if len(full_list) > 1:
        mean_delta = sum(abs(f - p) for f, p in zip(full_list, pruned_list)) / len(full_list)
        rho = spearman_correlation(full_list, pruned_list)
        print(f'Mean absolute error:      {mean_delta:.4f}')
        if rho is not None:
            print(f'Spearman rank correlation: {rho:.4f}')
    elif len(full_list) == 1:
        print(f'Score delta: {abs(full_list[0] - pruned_list[0]):.4f}')

    n_full = _count_samples(full_dir)
    n_pruned = _count_samples(pruned_dir)
    if n_full and n_pruned:
        print(f'Compression ratio:         {1 - n_pruned / n_full:.1%} reduction ({n_pruned}/{n_full} samples)')

    print()
    if all_pass:
        print('✓ Go/no-go decisions are CONSISTENT across all models.')
    else:
        print('✗ Go/no-go MISMATCH detected. Consider increasing prune_ratio.')
    print(f'{"="*64}\n')
    return 0 if all_pass else 1


def _count_samples(run_dir: Path) -> Optional[int]:
    """Estimate evaluated sample count from result files."""
    for root, _, files in os.walk(run_dir):
        for f in files:
            if not (f.endswith('.jsonl') or f.endswith('.json')):
                continue
            try:
                content = (Path(root) / f).read_text()
                if content.strip().startswith('['):
                    data = json.loads(content)
                    if isinstance(data, list):
                        return len(data)
                else:
                    lines = [l for l in content.splitlines() if l.strip()]
                    if lines:
                        return len(lines)
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Compare full vs pruned evalscope runs to measure fidelity.'
    )
    parser.add_argument('--full',   required=True, help='Path to full benchmark results directory')
    parser.add_argument('--pruned', required=True, help='Path to pruned benchmark results directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Go/no-go pass threshold (default: 0.5)')
    args = parser.parse_args()
    sys.exit(compare(Path(args.full), Path(args.pruned), threshold=args.threshold))


if __name__ == '__main__':
    main()
