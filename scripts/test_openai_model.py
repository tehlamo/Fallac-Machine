import os
import time
import json
import argparse
import subprocess
from pathlib import Path

DEFAULT_TESTS = [
    {
        "name": "Ad Hominem",
        "text": "Are you stupid?",
        "expected": ["ad hominem"],
    },
    {
        "name": "Ad Populum",
        "text": "If most of these scientists are saying it, then it must be true!",
        "expected": ["ad populum"],
    },
    {
        "name": "False Causality",
        "text": "The economy improved and then policies changed. The policies must have caused the improvement.",
        "expected": ["false causality"],
    },
    {
        "name": "Appeal to Emotion",
        "text": "Think of the children who will suffer!",
        "expected": ["appeal to emotion"],
    },
    {
        "name": "Circular Reasoning",
        "text": "If you didn't break it, then why is it broken?",
        "expected": ["circular reasoning"],
    },
    {
        "name": "Faulty Generalization",
        "text": "Some teens were rude, so all teenagers are disrespectful.",
        "expected": ["faulty generalization"],
    },
    {
        "name": "False Dilemma",
        "text": "You’re either with us or against us.",
        "expected": ["false dilemma"],
    },
    {
        "name": "Equivocation",
        "text": "A feather is light. What is light cannot be dark. Therefore, a feather cannot be dark.",
        "expected": ["equivocation"],
    },
    {
        "name": "Fallacy of Credibility",
        "text": "According to the famous billionaire, this policy is perfect, so it must be true.",
        "expected": ["fallacy of credibility"],
    },
    {
        "name": "Fallacy of Extension",
        "text": "If we allow students to use calculators, soon they won't learn math at all.",
        "expected": ["fallacy of extension"],
    },
    {
        "name": "Fallacy of Relevance",
        "text": "Your argument about climate policy is invalid because you drive a car.",
        "expected": ["fallacy of relevance"],
    },
    {
        "name": "Fallacy of Logic",
        "text": "If A then B. B happened, so A must be true.",
        "expected": ["fallacy of logic"],
    },
    {
        "name": "Multiple Mixed",
        "text": "Are you stupid? Everyone agrees this is right, so you should too.",
        "expected": ["ad hominem", "ad populum"],
    },
]


def load_model_id(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    if Path('fine_tuned_model.txt').exists():
        return Path('fine_tuned_model.txt').read_text(encoding='utf-8').strip()
    raise SystemExit('Error: provide --model or ensure fine_tuned_model.txt exists')


def run_case(python_exe: str, model_id: str, case: dict, idx: int, out_dir: Path) -> dict:
    out_path = out_dir / f"openai_test_{idx}.json"
    cmd = [
        python_exe, 'detect_fallacies_openai.py',
        '--model', model_id,
        '--text', case['text'],
        '--output', str(out_path)
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    duration = t1 - t0

    ok = result.returncode == 0 and out_path.exists()
    detected = []
    if ok:
        try:
            data = json.loads(out_path.read_text(encoding='utf-8'))
            detected = [f['fallacy_type'] for f in data.get('fallacies', [])]
        except Exception:
            ok = False

    return {
        'name': case['name'],
        'expected': case['expected'],
        'detected': detected,
        'ok': ok,
        'seconds': duration,
        'output_file': str(out_path),
        'stderr': result.stderr if result.stderr else ''
    }


def main():
    parser = argparse.ArgumentParser(description='Timed tests for OpenAI fallacy detector')
    parser.add_argument('--model', type=str, help='Fine-tuned OpenAI model id (defaults to fine_tuned_model.txt)')
    parser.add_argument('--outdir', type=str, default='openai_tests', help='Output directory for results')
    args = parser.parse_args()

    python_exe = str((Path('.') / 'FMenv' / 'Scripts' / 'python.exe').resolve())
    model_id = load_model_id(args.model)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.perf_counter()
    for i, case in enumerate(DEFAULT_TESTS, 1):
        res = run_case(python_exe, model_id, case, i, out_dir)
        results.append(res)
        print(f"[{i}/{len(DEFAULT_TESTS)}] {case['name']}: {res['seconds']:.2f}s, ok={res['ok']}")
    total_end = time.perf_counter()

    summary = {
        'model': model_id,
        'num_tests': len(DEFAULT_TESTS),
        'num_ok': sum(1 for r in results if r['ok']),
        'total_seconds': total_end - total_start,
        'avg_seconds_per_test': (total_end - total_start) / len(DEFAULT_TESTS),
        'cases': results,
    }

    out_summary = out_dir / 'summary.json'
    out_summary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"\nWrote summary to {out_summary}")


if __name__ == '__main__':
    main()
