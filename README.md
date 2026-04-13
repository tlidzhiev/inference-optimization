# Inference Optimization

Embedding inference pipeline for [rubert-mini-frida](https://huggingface.co/sergeyzh/rubert-mini-frida) with three optimization levels:

- **Part 1** — Base inference with `transformers`
- **Part 2** — ONNX Runtime inference
- **Part 3** — ONNX + Dynamic batching

Detailed performance comparison and optimization analysis are available in the main report — [REPORT.md](REPORT.md)

## 🛠️ Installation

**Prerequisites:** Python 3.12.11

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/inference-optimization.git
cd inference-optimization

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12.11
source .venv/bin/activate

# Install dependencies via uv
uv sync --all-groups

# Install pre-commit
pre-commit install
```

## 🚀 Usage

### Part 1 — Base Inference

```bash
uv run fastapi dev service_base.py
# http://127.0.0.1:8000/docs
```

### Part 2 — ONNX Inference

Export the model to ONNX once:

```bash
uv run convert_to_onnx.py
# Saves model.onnx in the project root
```

Then start the service:

```bash
uv run fastapi dev service_onnx.py
# http://127.0.0.1:8000/docs
```

### Part 3 — ONNX + Dynamic Batching

Requires `model.onnx` (see Part 2). Collects requests for up to 50 ms and merges them into a single batch (max 32 texts).

```bash
uv run fastapi dev service_onnx_batched.py
# http://127.0.0.1:8000/docs
```

## API

All services expose the same two endpoints.

### `POST /embed`

```json
{"texts": ["text one", "text two"]}
```

- `texts` — list of 1 to 32 strings (validated)

Response:

```json
{"embeddings": [[0.12, -0.34, ...], [0.56, 0.78, ...]]}
```

Embeddings are L2-normalized CLS-token vectors.

### `GET /health`

Returns `{"status": "ok"}` when the service is ready.

## 📊 Benchmarking

### Single run

Start a service in one terminal, then in another:

```bash
# Basic usage (defaults: 200 requests, concurrency 10, 1 text/req, warmup 10)
uv run benchmark.py

# Full options
uv run benchmark.py \
  --url http://localhost:8000/embed \
  --num-requests 500 \
  --concurrency 20 \
  --texts-per-request 3 \
  --warmup 10
```

Reported metrics: latency (p50/p90/p95/p99/mean/min/max/std_dev), throughput (req/s), success rate.

### Automated full benchmark

Starts each service automatically, runs all 5 load scenarios, and saves results to `benchmark_results.json`:

```bash
uv run run_benchmarks.py
```

## 🧪 Tests

```bash
uv run pytest tests/
```
