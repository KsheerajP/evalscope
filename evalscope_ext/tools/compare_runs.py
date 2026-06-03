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


def _find_report_files(run_dir: Path) -> list[Path]:
    """Recursively find all JSON report files in a results directory."""
    reports = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith('.json') and 'report' in f.lower():
                reports.append(Path(root) / f)
    # Also check for summary.json or results.json patterns
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f in ('summary.json', 'results.json', 'report.json'):
                reports.append(Path(root) / f)
    return list(set(reports))


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

    # Walk directory looking for report files
    for root, dirs, files in os.walk(run_dir):
        root_path = Path(root)
        for fname in files:
            if not fname.endswith('.json'):
                continue
            fpath = root_path / fname
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Pattern 1: evalscope report with 'model' and 'metrics' keys
            if isinstance(data, dict):
                model = data.get('model') or data.get('model_id') or _infer_model_from_path(fpath, run_dir)
                benchmark = data.get('benchmark') or data.get('dataset') or _infer_benchmark_from_path(fpath, run_dir)

                score = None
                # Try common metric locations
                for key in ('acc', 'pass@1', 'score', 'accuracy', 'mean'):
                    if key in data:
                        score = float(data[key])
                        break
                if score is None and 'metrics' in data:
                    metrics = data['metrics']
                    if isinstance(metrics, dict):
                        for key in ('acc', 'pass@1', 'score', 'accuracy', 'mean'):
                            if key in metrics:
                                score = float(metrics[key])
                                break
                if score is None and 'results' in data:
                    res = data['results']
                    if isinstance(res, dict):
                        for key in ('acc', 'pass@1', 'score', 'accuracy', 'mean'):
                            if key in res:
                                score = float(res[key])
                                break

                if model and benchmark and score is not None:
                    if model not in results:
                        results[model] = {}
                    results[model][benchmark] = score

    return results


def _infer_model_from_path(fpath: Path, base: Path) -> str:
    """Infer model name from directory structure."""
    try:
        rel = fpath.relative_to(base)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass
    return 'unknown_model'


def _infer_benchmark_from_path(fpath: Path, base: Path) -> str:
    """Infer benchmark name from directory structure."""
    try:
        rel = fpath.relative_to(base)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[1]
        if len(parts) == 1:
            return parts[0].replace('.json', '')
    except ValueError:
        pass
    return 'unknown_benchmark'


def spearman_correlation(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Spearman rank correlation (no scipy required)."""
    if len(x) != len(y) or len(x) < 2:
        return None

    def rank(arr):
        sorted_idx = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for rank_val, idx in enumerate(sorted_idx):
            ranks[idx] = rank_val + 1.0
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = (sum((r - mean_rx) ** 2 for r in rx) * sum((r - mean_ry) ** 2 for r in ry)) ** 0.5
    return num / den if den > 0 else 0.0


def _fmt(v: float, width: int = 8) -> str:
    return f'{v:.4f}'.rjust(width)


def compare(full_dir: Path, pruned_dir: Path, threshold: float = 0.5) -> int:
    """
    Compare full vs pruned runs and print a fidelity report.

    Returns exit code: 0 if all models pass fidelity check, 1 otherwise.
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
        print('  Expected directory structure: {run_dir}/{model}/{benchmark}/report.json')
        return 1
    if not pruned_scores:
        print(f'ERROR: No results found in {pruned_dir}')
        return 1

    # Find models present in both runs
    full_models = set(full_scores.keys())
    pruned_models = set(pruned_scores.keys())
    common_models = full_models & pruned_models

    if not common_models:
        print(f'WARNING: No common models between full ({sorted(full_models)}) '
              f'and pruned ({sorted(pruned_models)}) runs.')
        print('  Check that --model is the same in both eval commands.')
        return 1

    print(f'Models compared: {sorted(common_models)}\n')

    # Per-model comparison
    full_list, pruned_list = [], []
    all_pass = True

    print(f'{"Model":<30} {"Full":>8} {"Pruned":>8} {"Delta":>8} {"Status":>10}')
    print('-' * 68)

    for model in sorted(common_models):
        f_benchmarks = full_scores[model]
        p_benchmarks = pruned_scores[model]

        # Match benchmarks by normalized name (strip _pruned suffix for comparison)
        def _normalize(b: str) -> str:
            return b.replace('_pruned', '').lower()

        pairs: list[tuple[str, str, str]] = []  # (full_bench, pruned_bench, display_name)
        for fb in f_benchmarks:
            for pb in p_benchmarks:
                if _normalize(fb) == _normalize(pb) or fb == pb:
                    pairs.append((fb, pb, _normalize(fb)))
        if not pairs:
            # Fallback: pair by order if only one benchmark each
            if len(f_benchmarks) == 1 and len(p_benchmarks) == 1:
                fb = list(f_benchmarks)[0]
                pb = list(p_benchmarks)[0]
                pairs = [(fb, pb, _normalize(fb))]

        for fb, pb, display in pairs:
            fs = f_benchmarks[fb]
            ps = p_benchmarks[pb]
            delta = abs(ps - fs)
            full_list.append(fs)
            pruned_list.append(ps)

            # Go/no-go consistency: using threshold
            full_pass = fs >= threshold
            pruned_pass = ps >= threshold
            consistent = full_pass == pruned_pass
            status = '✓ consistent' if consistent else '✗ MISMATCH'
            if not consistent:
                all_pass = False

            label = f'{model[:20]}/{display[:10]}'
            print(f'{label:<30} {_fmt(fs)} {_fmt(ps)} {_fmt(delta)} {status:>12}')

    print()

    # Aggregate stats
    if len(full_list) > 1:
        mean_delta = sum(abs(f - p) for f, p in zip(full_list, pruned_list)) / len(full_list)
        rho = spearman_correlation(full_list, pruned_list)
        print(f'Mean absolute error:      {mean_delta:.4f}')
        if rho is not None:
            print(f'Spearman rank correlation: {rho:.4f}')
    elif len(full_list) == 1:
        delta = abs(full_list[0] - pruned_list[0])
        print(f'Score delta: {delta:.4f}')

    # Compression info
    n_full = _count_samples(full_dir)
    n_pruned = _count_samples(pruned_dir)
    if n_full and n_pruned:
        compression = 1 - n_pruned / n_full
        print(f'Compression ratio:         {compression:.1%} reduction ({n_pruned}/{n_full} samples)')

    print()
    if all_pass:
        print('✓ Go/no-go decisions are CONSISTENT across all models.')
    else:
        print('✗ Go/no-go MISMATCH detected. Consider increasing prune_ratio.')

    print(f'{"="*64}\n')
    return 0 if all_pass else 1


def _count_samples(run_dir: Path) -> Optional[int]:
    """Try to count evaluated samples from result files."""
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith('.jsonl') or f.endswith('.json'):
                fpath = Path(root) / f
                try:
                    with open(fpath) as fp:
                        content = fp.read()
                    if content.strip().startswith('['):
                        data = json.loads(content)
                        if isinstance(data, list):
                            return len(data)
                    else:
                        # JSONL
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
    parser.add_argument('--full', required=True, help='Path to full benchmark results directory')
    parser.add_argument('--pruned', required=True, help='Path to pruned benchmark results directory')
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Go/no-go pass threshold (default: 0.5). Score >= threshold = PASS.'
    )
    args = parser.parse_args()

    exit_code = compare(Path(args.full), Path(args.pruned), threshold=args.threshold)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
