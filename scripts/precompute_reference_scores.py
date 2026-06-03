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
import os
import re
from pathlib import Path

MODELS_PART1 = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
MMMU_MODEL = 'glm-4.5v-fp8'

# Subjects where image understanding is essential (cannot answer from text alone)
IMAGE_NECESSITY = {
    'Electronics': 1.0,
    'Architecture_and_Engineering': 0.95,
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
    'Basic_Medical_Science': 0.60,
    'Mechanical_Engineering': 0.80,
    'Mathematics': 0.55,
    'Agriculture': 0.50,
    'Psychology': 0.40,
    'Sociology': 0.40,
    'Economics': 0.35,
    'Manage': 0.35,
    'Marketing': 0.35,
    'Art_Theory': 0.60,
    'History': 0.30,
    'Literature': 0.25,
    'Accounting': 0.20,
    'Finance': 0.20,
    'Public_Health': 0.35,
    'Pharmacy': 0.40,
    'Music': 0.55,
}

VISUAL_KEYWORDS = frozenset(['graph', 'chart', 'diagram', 'figure', 'table', 'plot', 'image', 'picture', 'shown', 'depicted', 'illustrated'])


def compute_stats(scores_per_model: list[list[float]]) -> dict:
    """
    scores_per_model: list of score lists, one per model
    Returns: {index: {difficulty, discrimination}}
    """
    if not scores_per_model or not scores_per_model[0]:
        return {}

    n_samples = len(scores_per_model[0])
    result = {}

    for i in range(n_samples):
        sample_scores = [scores[i] for scores in scores_per_model if i < len(scores)]
        if not sample_scores:
            continue
        difficulty = sum(sample_scores) / len(sample_scores)
        discrimination = max(sample_scores) - min(sample_scores)
        result[str(i)] = {
            'difficulty': round(difficulty, 4),
            'discrimination': round(discrimination, 4),
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
        model_scores = {}
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                idx = row['index']
                score = row['sample_score']['score']['value'].get(score_key, 0.0)
                model_scores[idx] = float(score)
        # Convert to ordered list
        max_idx = max(model_scores.keys()) + 1
        scores = [model_scores.get(i, 0.0) for i in range(max_idx)]
        scores_per_model.append(scores)

    return compute_stats(scores_per_model)


def load_mmmu_reviews(evals_dir: Path) -> dict:
    """Load MMMU per-subject, per-sample scores."""
    reviews_dir = evals_dir / 'MMMU' / 'reviews' / MMMU_MODEL
    preds_dir = evals_dir / 'MMMU' / 'predictions' / MMMU_MODEL

    result = {}

    for review_file in sorted(reviews_dir.glob('mmmu_*.jsonl')):
        subject = review_file.stem.replace('mmmu_', '')

        # Load scores
        sample_scores = {}
        with open(review_file) as f:
            for line in f:
                row = json.loads(line)
                idx = row['index']
                score = float(row['sample_score']['score']['value'].get('acc', 0.0))
                sample_scores[idx] = score

        # Load questions for keyword analysis
        pred_file = preds_dir / f'mmmu_{subject}.jsonl'
        question_texts = {}
        if pred_file.exists():
            with open(pred_file) as f:
                for line in f:
                    row = json.loads(line)
                    # Extract question text from messages or input
                    msgs = row.get('messages', [])
                    text = ''
                    for msg in msgs:
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get('type') == 'text':
                                    text += c.get('text', '')
                        elif isinstance(content, str):
                            text += content
                    question_texts[row['index']] = text.lower()

        # Compute image necessity with keyword boost
        base_necessity = IMAGE_NECESSITY.get(subject, 0.5)
        subject_result = {}
        for idx, score in sample_scores.items():
            q_text = question_texts.get(idx, '')
            keyword_boost = 0.2 if any(kw in q_text for kw in VISUAL_KEYWORDS) else 0.0
            image_necessity = min(1.0, base_necessity + keyword_boost)
            subject_result[str(idx)] = {
                'difficulty': round(1.0 - score, 4),  # higher = harder (1=failed, 0=passed)
                'discrimination': 0.0,  # single model, no discrimination
                'image_necessity': round(image_necessity, 4),
                'subject': subject,
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

    # Augment with input_tokens metadata
    aalcr = {}
    preds_dir = evals_dir / 'Part 1' / 'predictions'
    token_counts = {}
    for model in MODELS_PART1:
        fpath = preds_dir / f'aa_lcr__{model}.jsonl'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                if row['index'] not in token_counts:
                    token_counts[row['index']] = row.get('metadata', {}).get('input_tokens', 0)
        break  # only need one model for token counts

    for idx_str, stats in aalcr_base.items():
        idx = int(idx_str)
        aalcr[idx_str] = {
            **stats,
            'input_tokens': token_counts.get(idx, 0),
        }
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
