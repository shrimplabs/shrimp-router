# Shrimp Vision Router

A tiny OpenAI-compatible router for pooling local VLM workers across a LAN, with per-backend concurrency limits, health checks, and simple model routing.

## Goal

Shrimp Vision Router sits between clients and local vision-language model servers such as `mlx_vlm.server`, vLLM, Ollama, LM Studio, or any OpenAI-compatible VLM endpoint.

It exposes a stable OpenAI-compatible API:

```text
POST /v1/chat/completions
GET  /health
```

and forwards requests to configured backends while protecting each machine with explicit concurrency limits.

## Initial Use Case

Run model servers on multiple LAN machines:

```sh
python3 -m mlx_vlm.server --host 0.0.0.0 --port 8080
# or, on CUDA hosts:
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8080
```

Then point clients at the router instead of a single local machine.

## Planned Features

- OpenAI-compatible `/v1/chat/completions` proxy
- YAML configuration for backend workers
- Model-to-backend routing across MLX, CUDA, Ollama, LM Studio, and other OpenAI-compatible workers
- Per-backend concurrency caps
- Health checks
- Request timeouts and one retry on compatible backends
- Structured request logs
- Optional LAN-only bearer token

## Non-Goals

- No swarm-controller-specific task logic
- No database in the first version
- No model serving implementation; backend workers serve models themselves
