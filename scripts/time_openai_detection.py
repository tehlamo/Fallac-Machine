import time
import json
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Time OpenAI fallacy detection on a file')
    parser.add_argument('--model', required=True, help='Fine-tuned OpenAI model id')
    parser.add_argument('--file', required=True, help='Input text file path')
    parser.add_argument('--output', default='openai_long_test.json', help='Output JSON path')
    args = parser.parse_args()

    python_exe = str((Path('.') / 'FMenv' / 'Scripts' / 'python.exe').resolve())
    output_path = Path(args.output)

    cmd = [
        python_exe, 'detect_fallacies_openai.py',
        '--model', args.model,
        '--file', args.file,
        '--output', str(output_path)
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()

    elapsed = t1 - t0

    ok = result.returncode == 0 and output_path.exists()
    payload = {
        'model': args.model,
        'input_file': args.file,
        'output_file': str(output_path),
        'elapsed_seconds': elapsed,
        'ok': ok,
        'stderr': result.stderr if result.stderr else ''
    }

    # If result JSON exists, merge in
    if ok:
        try:
            data = json.loads(output_path.read_text(encoding='utf-8'))
            payload['detector_output'] = data
        except Exception:
            payload['detector_output'] = None

    summary_path = Path('openai_long_test_summary.json')
    summary_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(f"Elapsed: {elapsed:.2f}s; ok={ok}; summary -> {summary_path}")

if __name__ == '__main__':
    main()
