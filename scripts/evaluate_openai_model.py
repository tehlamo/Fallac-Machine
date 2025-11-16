import argparse
import json
import subprocess
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

LABELS = [
    "ad hominem", "ad populum", "appeal to emotion", "circular reasoning",
    "equivocation", "fallacy of credibility", "fallacy of extension",
    "fallacy of logic", "fallacy of relevance", "false causality",
    "false dilemma", "faulty generalization", "intentional",
    "miscellaneous", "none"
]

TESTS = [
    {
        'name': 'efficiency',
        'file': 'tests/test_efficiency.txt'
    },
    {
        'name': 'confidence',
        'file': 'tests/test_confidence.txt'
    },
    {
        'name': 'accuracy',
        'file': 'tests/test_accuracy.txt'
    },
]


def expected_labels_for(text: str, test_name: str):
    sentences = sent_tokenize(text)
    exp = ['none'] * len(sentences)
    low = [s.lower() for s in sentences]

    if test_name == 'efficiency':
        return sentences, exp

    if test_name == 'confidence':
        for i, s in enumerate(low):
            if 'celebrity entrepreneur' in s:
                exp[i] = 'fallacy of credibility'
        return sentences, exp

    if test_name == 'accuracy':
        for i, s in enumerate(low):
            if 'everyone knows' in s or 'already agreed' in s:
                exp[i] = 'ad populum'
            if 'enabled feature flags on monday' in s and 'tuesday' in s:
                exp[i] = 'false causality'
            if 'think of the families' in s:
                exp[i] = 'appeal to emotion'
            if s.strip().startswith('either you approve'):
                exp[i] = 'false dilemma'
            if 'your analysis is worthless' in s or 'failed a class' in s:
                exp[i] = 'ad hominem'
            if 'if this forecast is correct' in s and 'revenue is rising' in s:
                exp[i] = 'circular reasoning'
        return sentences, exp

    return sentences, exp


def run_detector(python_exe: Path, model_id: str, file_path: Path, out_path: Path):
    cmd = [
        str(python_exe), 'detect_fallacies_openai.py',
        '--model', model_id,
        '--file', str(file_path),
        '--output', str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    data = json.loads(out_path.read_text(encoding='utf-8'))
    return data


def apply_threshold(preds, threshold: float):
    filtered = []
    for p in preds:
        label = p.get('fallacy_type', 'none')
        conf = float(p.get('confidence', 0))
        if conf < threshold:
            label = 'none'
        filtered.append({**p, 'fallacy_type': label})
    return filtered


def compute_metrics(expected: list[str], predicted: list[str]):
    classes = LABELS
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    for y_true, y_pred in zip(expected, predicted):
        if y_pred == y_true:
            tp[y_true] += 1
        else:
            fp[y_pred] += 1
            fn[y_true] += 1
    metrics = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2*p*r)/(p+r) if (p+r) > 0 else 0.0
        metrics[c] = {'precision': p, 'recall': r, 'f1': f1, 'tp': tp[c], 'fp': fp[c], 'fn': fn[c]}
    overall_acc = sum(1 for a,b in zip(expected, predicted) if a==b) / len(expected) if expected else 0.0
    return metrics, overall_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenAI detector with threshold')
    parser.add_argument('--model', required=True, help='Fine-tuned model id')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold (default 0.6)')
    parser.add_argument('--out', default='openai_eval_results.json', help='Output JSON summary file')
    args = parser.parse_args()

    python_exe = Path('FMenv') / 'Scripts' / 'python.exe'

    report = {
        'model': args.model,
        'threshold': args.threshold,
        'tests': []
    }

    for t in TESTS:
        file_path = Path(t['file'])
        text = file_path.read_text(encoding='utf-8')
        sentences, expected = expected_labels_for(text, t['name'])

        tmp_out = Path(f"_tmp_{t['name']}.json")
        data = run_detector(python_exe, args.model, file_path, tmp_out)
        preds = data.get('fallacies', [])
        preds = apply_threshold(preds, args.threshold)
        predicted = [p.get('fallacy_type', 'none') for p in preds]

        metrics, acc = compute_metrics(expected, predicted)
        report['tests'].append({
            'name': t['name'],
            'file': t['file'],
            'num_sentences': len(sentences),
            'accuracy': acc,
            'per_class': metrics
        })
        try:
            tmp_out.unlink()
        except Exception:
            pass

    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Wrote evaluation to {args.out}")


if __name__ == '__main__':
    main()
