import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVICES = [
    ('Base', 'service_base:app'),
    ('ONNX', 'service_onnx:app'),
    ('ONNX + Batch', 'service_onnx_batched:app'),
]

SCENARIOS = [
    {'name': 'Low Load', 'concurrency': 10, 'texts_per_request': 1, 'num_requests': 200},
    {'name': 'Medium Load', 'concurrency': 20, 'texts_per_request': 3, 'num_requests': 200},
    {'name': 'High Load', 'concurrency': 20, 'texts_per_request': 10, 'num_requests': 200},
    {'name': 'Heavy Load', 'concurrency': 50, 'texts_per_request': 10, 'num_requests': 200},
    {'name': 'Extreme Load', 'concurrency': 100, 'texts_per_request': 10, 'num_requests': 200},
]

PORT = 8000
URL = f'http://localhost:{PORT}/embed'
HEALTH_URL = f'http://localhost:{PORT}/health'


def wait_for_service(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def parse_output(output: str) -> dict:
    metrics = {}
    for key, pattern in [
        ('throughput', r'Throughput:\s+([\d.]+)\s+req/s'),
        ('p50', r'Latency p50:\s+([\d.]+)\s+ms'),
        ('p90', r'Latency p90:\s+([\d.]+)\s+ms'),
        ('p95', r'Latency p95:\s+([\d.]+)\s+ms'),
        ('p99', r'Latency p99:\s+([\d.]+)\s+ms'),
        ('mean', r'Latency mean:\s+([\d.]+)\s+ms'),
        ('std_dev', r'Latency std dev:\s+([\d.]+)\s+ms'),
    ]:
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def main():
    if not os.path.exists('model.onnx'):
        print('model.onnx not found — running convert_to_onnx.py first...')
        subprocess.run([sys.executable, 'convert_to_onnx.py'], check=True)
        print()

    results = {}

    for name, module in SERVICES:
        print(f'\n{"=" * 60}')
        print(f'  {name} ({module})')
        print(f'{"=" * 60}')

        proc = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', module, '--host', '127.0.0.1', '--port', str(PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            if not wait_for_service():
                print(f'  FAILED to start {name}')
                continue

            print('  Ready.\n')
            results[name] = {}

            for s in SCENARIOS:
                label = f'{s["name"]} (c={s["concurrency"]}, t={s["texts_per_request"]})'
                print(f'  {label} ... ', end='', flush=True)

                r = subprocess.run(
                    [
                        sys.executable,
                        'benchmark.py',
                        '--url',
                        URL,
                        '--num-requests',
                        str(s['num_requests']),
                        '--concurrency',
                        str(s['concurrency']),
                        '--texts-per-request',
                        str(s['texts_per_request']),
                        '--warmup',
                        '10',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                m = parse_output(r.stdout)
                if m:
                    results[name][s['name']] = m
                    print(f'{m["throughput"]:.1f} req/s  p50={m["p50"]:.1f}  p99={m["p99"]:.1f}')
                else:
                    print('FAILED')
                    if r.stderr:
                        print(f'    stderr: {r.stderr[:200]}')

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            time.sleep(2)  # let port release

    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nDone. Results saved to benchmark_results.json')


if __name__ == '__main__':
    main()
