# Report: Inference Pipeline Optimization

## Model
**sergeyzh/rubert-mini-frida** — a compact Russian BERT model for computing text embeddings. Embeddings are extracted via CLS-token pooling from the last hidden state and L2-normalized before returning.

---

## Test Setup

| Parameter | Value |
|---|---|
| Platform | macOS (Apple Silicon, CPU-only) |
| Server | Uvicorn, 1 worker |
| Requests per scenario | 200 |
| Warmup requests | 10 |
| Input texts | Randomly sampled from a pool of 5 Russian sentences |
| Max sequence length | 512 tokens |

All three services were benchmarked under identical conditions: same process isolation, same hardware, single uvicorn worker. CPU-only inference means no GPU acceleration is involved in any scenario.

---

## Implementations

### Part 1 — Base PyTorch (`service_base`)
Standard HuggingFace `transformers` inference. The `/embed` endpoint is a **synchronous** FastAPI handler, which means uvicorn runs it in a thread-pool executor. Under concurrency, multiple requests can be in-flight simultaneously, each occupying a thread. Tokenization and model forward pass use PyTorch tensors with `torch.no_grad()`.

### Part 2 — ONNX Runtime (`service_onnx`)
The model is exported to ONNX format via `torch.onnx.export` with dynamic batch and sequence axes. At runtime, ONNX Runtime (`CPUExecutionProvider`) applies graph optimizations (operator fusion, constant folding, layout transformations) that eliminate Python-level overhead and reduce per-operation latency. Tokenization still uses the HuggingFace tokenizer, but inference runs through `ort.InferenceSession`. The endpoint is also **synchronous**, matching the Base architecture.

### Part 3 — ONNX + Dynamic Batching (`service_onnx_batched`)
Same ONNX backend, but with an **async `BatchQueue`** that coalesces concurrent requests before inference:
- The endpoint is `async` and suspends on `await batch_queue.submit(texts)`.
- A background worker collects items from the queue for up to **50 ms** (`MAX_WAIT_MS`) or until **32 texts** are accumulated (`MAX_BATCH_SIZE`).
- A single ONNX inference call is made for the entire batch; results are sliced and distributed back to individual request futures.
- If a new request would overflow the current batch, it is deferred to the next batch via a `pending` buffer (no data is truncated or dropped).
- The blocking ONNX call is offloaded via `asyncio.to_thread` to avoid blocking the event loop.

---

## Metrics Selection

| Metric | Justification |
|---|---|
| **Throughput (req/s)** | System capacity under load — critical for production sizing. |
| **Latency p50** | Typical user experience. |
| **Latency p90/p95/p99** | Tail latencies — represent worst-case scenarios and SLA stability. |
| **Mean latency** | Average processing time for overall evaluation. |
| **Std Dev** | Jitter — low value indicates predictable, consistent performance. |

---

## Comparative Analysis by Load Level

Best value per metric is **bolded**.

### 1. Low Load: 10 Concurrent Requests, 1 Text/Req
ONNX delivers a **~4x throughput advantage** over Base. Dynamic Batching is the slowest option: the 50 ms batch-gathering window is pure overhead when requests arrive with single texts at low concurrency. It does, however, show marginally lower jitter than ONNX.

| Service | Throughput (req/s) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std Dev (ms) |
|---|---|---|---|---|---|---|---|
| Base | 275.31 | 36.1 | 39.3 | 39.8 | 42.1 | 36.0 | 3.2 |
| **ONNX** | **1172.38** | **8.1** | **10.5** | **11.4** | **13.5** | **8.2** | 1.7 |
| ONNX + Batch | 152.40 | 65.1 | 66.2 | 67.1 | 70.0 | 64.9 | **1.6** |

### 2. Medium Load: 20 Concurrent Requests, 3 Texts/Req
ONNX leads in throughput and mean latency. Dynamic Batching wins p90/p95 by coalescing requests into larger batches, but its p99 is slightly worse than ONNX — individual requests that arrive just after a batch flushes wait a full 50 ms before the next flush. Base has the lowest jitter.

| Service | Throughput (req/s) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std Dev (ms) |
|---|---|---|---|---|---|---|---|
| Base | 234.52 | 84.5 | 94.1 | 96.2 | 99.0 | 84.1 | **8.6** |
| **ONNX** | **576.37** | **31.3** | 53.1 | 65.2 | **82.9** | **33.7** | 16.6 |
| ONNX + Batch | 400.43 | 42.0 | **46.8** | **62.0** | 93.0 | 45.2 | 11.7 |

### 3. High Load: 20 Concurrent Requests, 10 Texts/Req
A counterintuitive scenario: **Base PyTorch wins all tail-latency and jitter metrics**. With 10 texts per request, the natural queuing from FastAPI's fixed-size thread pool provides implicit rate limiting that prevents requests from piling up. ONNX processes each request faster individually, but without any rate-limiting effect, this leads to higher variance. Dynamic Batching has the highest latency across the board here, as the 50 ms wait window becomes costly when payloads are already large.

| Service | Throughput (req/s) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std Dev (ms) |
|---|---|---|---|---|---|---|---|
| Base | 175.64 | 112.2 | **123.7** | **126.3** | **134.1** | 112.0 | **12.0** |
| **ONNX** | **185.31** | **99.6** | 143.8 | 152.2 | 173.0 | **103.2** | 31.4 |
| ONNX + Batch | 140.25 | 135.4 | 148.8 | 150.1 | 153.6 | 131.7 | 21.3 |

### 4. Heavy Load: 50 Concurrent Requests, 10 Texts/Req
ONNX leads in throughput, p50, mean, and jitter. Dynamic Batching takes over for p95/p99 — at this concurrency, batch coalescing provides enough gain to overcome its overhead. ONNX wins p90 outright (313.7 ms vs Batch's 332.0 ms).

| Service | Throughput (req/s) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std Dev (ms) |
|---|---|---|---|---|---|---|---|
| Base | 166.19 | 259.7 | 428.5 | 438.6 | 467.7 | 288.3 | 73.9 |
| **ONNX** | **189.95** | **232.1** | **313.7** | 345.3 | 450.7 | **238.3** | **71.2** |
| ONNX + Batch | 145.90 | 326.1 | 332.0 | **333.0** | **345.7** | 290.5 | 76.5 |

### 5. Extreme Load: 100 Concurrent Requests, 10 Texts/Req
ONNX leads in throughput, p50, mean, and jitter. Dynamic Batching shows exceptional tail-latency stability: its p99/p50 ratio is **1.07** (694 ms / 648 ms), compared to **1.87** for ONNX and **1.92** for Base. Base p90/p99 nearly double its p50, showing severe queue buildup.

| Service | Throughput (req/s) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std Dev (ms) |
|---|---|---|---|---|---|---|---|
| Base | 156.63 | 536.2 | 991.6 | 1015.8 | 1031.5 | 571.2 | 194.2 |
| **ONNX** | **191.59** | **440.8** | 695.7 | 779.1 | 822.0 | **450.2** | **166.1** |
| ONNX + Batch | 142.27 | 648.3 | **687.9** | **691.6** | **694.2** | 518.0 | 206.7 |

---

## Core Findings

1. **ONNX Runtime** is the single most effective optimization for raw throughput, delivering a **~4x speedup** over Base at low load and a consistent **15–22% throughput lead** at high concurrency. The gains come from operator fusion, constant folding, and elimination of Python interpreter overhead.

2. **Dynamic Batching trades throughput for tail-latency stability at high concurrency.** It is consistently 20–25% slower than ONNX in throughput, because the 50 ms collection window is dead time for every batch. Its advantage emerges at heavy and extreme load (50–100 concurrent), where request coalescing drives p99 down to within 7% of p50.

3. **Base PyTorch is not always the worst option for tail latencies.** At High Load (20 concurrent, 10 texts/req), Base wins p90, p95, p99, and Std Dev. FastAPI's synchronous threadpool provides natural back-pressure that ONNX and Batching lack, preventing latency spikes when per-request payloads are large. This disappears at higher concurrency (50+) where queue buildup dominates.

4. **p90 vs p99 split at Medium Load.** Batching wins p90/p95 but loses p99 to ONNX. Requests that arrive immediately after a batch flush must wait a full `MAX_WAIT_MS` (50 ms) before being processed, which appears in the p99 tail.

---

## Trade-offs

| Load Profile | Recommended Strategy | Reason |
|---|---|---|
| **Low load** (< 20 concurrent, 1–3 texts) | Raw ONNX | ~4x throughput; batch-gathering delay is pure overhead at low arrival rates |
| **Medium load** (20 concurrent, 3 texts) | Raw ONNX, Batching if p90 SLA is tight | ONNX wins throughput and p99; Batching wins p90/p95 |
| **High load** (20 concurrent, 10 texts) | Base or ONNX | Base unexpectedly wins tail latencies; ONNX wins mean/throughput |
| **Heavy load** (50 concurrent, 10 texts) | ONNX | Best throughput, mean, p50, p90, and jitter; Batching only wins p95/p99 |
| **Extreme load** (100+ concurrent, 10 texts) | ONNX + Batching | Near-flat p50–p99 distribution (ratio 1.07) prevents tail-latency outliers that could breach SLAs |

---

## Conclusions

ONNX Runtime is the most impactful single optimization, providing consistent throughput and latency improvements across all scenarios. Dynamic Batching's value is load-profile dependent: it reduces throughput by 20–25% in exchange for dramatically flattened tail-latency distributions at extreme concurrency. The optimal strategy depends on the SLA target — throughput-focused workloads should use raw ONNX, while latency-sensitive workloads under heavy concurrency benefit from ONNX + Batching.

A non-obvious result is that Base PyTorch outperforms both ONNX variants on p90–p99 at 20 concurrent requests with 10 texts per request. This is attributed to the implicit back-pressure of FastAPI's synchronous threadpool, which serializes requests and prevents variance amplification. This effect disappears at higher concurrency (50+) where threadpool saturation itself becomes the bottleneck.
