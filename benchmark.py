import argparse
import asyncio
import random
import statistics
import time

import aiohttp
import numpy as np

SAMPLE_TEXTS = [
    'Привет, как дела?',
    'Машинное обучение — это интересно.',
    'Оптимизация инференса позволяет ускорить работу модели.',
    'Динамическое батчирование объединяет несколько запросов.',
    'ONNX Runtime предоставляет кроссплатформенный инференс.',
]


async def send_request(
    session: aiohttp.ClientSession, url: str, texts: list[str]
) -> tuple[float, bool]:
    payload = {'texts': texts}
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            is_success = resp.status == 200
            await resp.json()
    except Exception:
        is_success = False

    return time.perf_counter() - start, is_success


async def run_benchmark(
    url: str,
    num_requests: int,
    concurrency: int,
    texts_per_request: int,
    warmup_requests: int,
):
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        print(f'Performing warmup ({warmup_requests} requests)...')
        for _ in range(warmup_requests):
            await send_request(session, url, random.choices(SAMPLE_TEXTS, k=texts_per_request))
        print('Warmup completed.\n')

        sem = asyncio.Semaphore(concurrency)
        latencies: list[float] = []
        success_count = 0

        async def bounded_request():
            nonlocal success_count
            async with sem:
                lat, is_success = await send_request(
                    session, url, random.choices(SAMPLE_TEXTS, k=texts_per_request)
                )
                latencies.append(lat)
                if is_success:
                    success_count += 1

        print(f'Starting benchmark: {num_requests} requests with concurrency {concurrency}...')
        t0 = time.perf_counter()

        tasks = [bounded_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - t0

    latencies.sort()
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    pmax = max(latencies)
    pmin = min(latencies)
    std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    throughput = num_requests / elapsed
    success_rate = (success_count / num_requests) * 100

    print(f'\n{"=" * 50}')
    print(f'Benchmark Results: {url}')
    print(f'Requests: {num_requests}, Concurrency: {concurrency}, Texts/req: {texts_per_request}')
    print(f'{"=" * 50}')
    print(f'Total time:      {elapsed:.2f} s')
    print(f'Throughput:      {throughput:.2f} req/s')
    print(f'Success Rate:    {success_rate:.1f}%')
    print('-' * 50)
    print(f'Latency p50:     {p50 * 1000:.1f} ms')
    print(f'Latency p90:     {p90 * 1000:.1f} ms')
    print(f'Latency p95:     {p95 * 1000:.1f} ms')
    print(f'Latency p99:     {p99 * 1000:.1f} ms')
    print(f'Latency mean:    {statistics.mean(latencies) * 1000:.1f} ms')
    print(f'Latency min/max: {pmin * 1000:.1f} / {pmax * 1000:.1f} ms')
    print(f'Latency std dev: {std_dev * 1000:.1f} ms')
    print(f'{"=" * 50}')


def main():
    parser = argparse.ArgumentParser(description='Benchmark embedding service')
    parser.add_argument('--url', default='http://localhost:8000/embed')
    parser.add_argument('--num-requests', type=int, default=200)
    parser.add_argument('--concurrency', type=int, default=10)
    parser.add_argument('--texts-per-request', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            args.url, args.num_requests, args.concurrency, args.texts_per_request, args.warmup
        )
    )


if __name__ == '__main__':
    main()
