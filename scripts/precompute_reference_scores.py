"""
Pre-compute per-sample difficulty and discrimination scores from challenge JSONL data.

Run once from repo root:
    python scripts/precompute_reference_scores.py \
        --evals-dir /path/to/challenge/Evals

Outputs:
    evalscope/pruning/reference_data/live_code_bench_v5.json
    evalscope/pruning/reference_data/aa_lcr.json
    evalscope/pruning/reference_data/mmmu.json
"""
import argparse
import json
from pathlib import Path

# Import subject weights and keywords from core to avoid duplication
from evalscope.pruning.core import EncoderProbeSelector

MODELS_PART1 = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
MMMU_MODEL = 'glm-4.5v-fp8'

_SELECTOR = EncoderProbeSelector()


def compute_stats(scores_per_model: list[list[float]]) -> dict:
    """
    scores_per_model: list of score lists, one per model.
    Returns: {str(index): {difficulty, discrimination}}
    """
    if not scores_per_model or not scores_per_model[0]:
        return {}

    n_samples = len(scores_per_model[0])
    result = {}
    for i in range(n_samples):
        sample_scores = [scores[i] for scores in scores_per_model if i < len(scores)]
        if not sample_scores:
            continue
        result[str(i)] = {
            'difficulty':      round(sum(sample_scores) / len(sample_scores), 4),
            'discrimination':  round(max(sample_scores) - min(sample_scores), 4),
        }
    return result


def load_reviews_part1(evals_dir: Path, benchmark: str, score_key: str) -> dict:
    """Load per-sample scores from Part 1 JSONL reviews."""
    scores_per_model = []
    for model in MODELS_PART1:
        fpath = evals_dir / 'Part 1' / 'reviews' / f'{benchmark}__{model}.jsonl'
        if not fpath.exists():
            print(f'  WARNING: missing {fpath}')
            continue
        model_scores: dict[int, float] = {}
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                model_scores[row['index']] = float(
                    row['sample_score']['score']['value'].get(score_key, 0.0)
                )
        if not model_scores:
            continue
        max_idx = max(model_scores) + 1
        scores_per_model.append([model_scores.get(i, 0.0) for i in range(max_idx)])

    return compute_stats(scores_per_model)


def load_mmmu_reviews(evals_dir: Path) -> dict:
    """Load MMMU per-subject, per-sample scores with image necessity."""
    reviews_dir = evals_dir / 'MMMU' / 'reviews' / MMMU_MODEL
    preds_dir   = evals_dir / 'MMMU' / 'predictions' / MMMU_MODEL
    result = {}

    for review_file in sorted(reviews_dir.glob('mmmu_*.jsonl')):
        subject = review_file.stem.replace('mmmu_', '')

        # Load pass/fail scores
        sample_scores: dict[int, float] = {}
        with open(review_file) as f:
            for line in f:
                row = json.loads(line)
                sample_scores[row['index']] = float(
                    row['sample_score']['score']['value'].get('acc', 0.0)
                )

        # Load question text for keyword boost
        question_texts: dict[int, str] = {}
        pred_file = preds_dir / f'mmmu_{subject}.jsonl'
        if pred_file.exists():
            with open(pred_file) as f:
                for line in f:
                    row = json.loads(line)
                    text = ''
                    for msg in row.get('messages', []):
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get('type') == 'text':
                                    text += c.get('text', '')
                        elif isinstance(content, str):
                            text += content
                    question_texts[row['index']] = text.lower()

        subject_result = {}
        for idx, score in sample_scores.items():
            q_text = question_texts.get(idx, '')
            # Delegate necessity scoring to EncoderProbeSelector (single source of truth)
            image_necessity = _SELECTOR.compute_image_necessity(subject, q_text)
            subject_result[str(idx)] = {
                'difficulty':       round(1.0 - score, 4),  # 1 = failed, 0 = passed
                'discrimination':   0.0,                     # single model
                'image_necessity':  round(image_necessity, 4),
                'subject':          subject,
            }
        result[subject] = subject_result

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evals-dir', required=True, help='Path to challenge Evals/ directory')
    args = parser.parse_args()

    evals_dir = Path(args.evals_dir)
    out_dir = Path(__file__).parent.parent / 'evalscope' / 'pruning' / 'reference_data'
    out_dir.mkdir(parents=True, exist_ok=True)

    # LiveCodeBench
    print('Computing LCB scores...')
    lcb = load_reviews_part1(evals_dir, 'live_code_bench_v5', 'pass')
    out = out_dir / 'live_code_bench_v5.json'
    out.write_text(json.dumps(lcb, indent=2))
    print(f'  Saved {len(lcb)} samples → {out}')

    # AA-LCR
    print('Computing AA-LCR scores...')
    aalcr_base = load_reviews_part1(evals_dir, 'aa_lcr', 'acc')

    # Augment with input_tokens from predictions
    token_counts: dict[int, int] = {}
    for model in MODELS_PART1:
        fpath = evals_dir / 'Part 1' / 'predictions' / f'aa_lcr__{model}.jsonl'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                if row['index'] not in token_counts:
                    token_counts[row['index']] = row.get('metadata', {}).get('input_tokens', 0)
        break  # one model is sufficient for token counts

    aalcr = {k: {**v, 'input_tokens': token_counts.get(int(k), 0)} for k, v in aalcr_base.items()}
    out = out_dir / 'aa_lcr.json'
    out.write_text(json.dumps(aalcr, indent=2))
    print(f'  Saved {len(aalcr)} samples → {out}')

    # MMMU
    print('Computing MMMU scores...')
    mmmu = load_mmmu_reviews(evals_dir)
    out = out_dir / 'mmmu.json'
    out.write_text(json.dumps(mmmu, indent=2))
    total = sum(len(v) for v in mmmu.values())
    print(f'  Saved {len(mmmu)} subjects, {total} samples → {out}')

    print('\nDone.')


if __name__ == '__main__':
    main()
