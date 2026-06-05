# Shrimp Router

An intelligent OpenAI-compatible gateway for routing LLM and VLM requests across multiple subscription providers and local hardware nodes.

Sits between your agents and your backends. Handles quota tracking, task-type routing, vision detection, weighted load balancing across a VLM node pool, and automatic fallback on 429s — without any changes to the agents themselves.

## Architecture

```
agents / swarm-controller
        ↓  POST /v1/chat/completions
  shrimp-router :8090
        ↓
  routing logic:
  ┌─────────────────────────────────────────────────┐
  │ vision request? (image_url in messages)          │
  │   → weighted round-robin: M4-1, M4-2, 3070      │
  │                                                  │
  │ text request + X-Task-Type header                │
  │   bug/polish    → Kimi → OpenCode → MiniMax      │
  │   feature/refactor → MiniMax → OpenCode → Kimi   │
  │   research/plan → OpenCode (cheap) → Kimi        │
  │                                                  │
  │ quota exhausted? → skip to next provider         │
  │ 429?            → backoff + try next             │
  └─────────────────────────────────────────────────┘
        ↓
  MiniMax / Kimi / OpenCode Go  (subscription text)
  M4-mini-1 / M4-mini-2 / 3070 (local VLM nodes)
```

## Quickstart

### 1. Start VLM nodes

**On each M4 Mac Mini:**
```bash
git clone git@github.com:shrimplabs/shrimp-router.git
./scripts/start-vlm-apple.sh
```

**On the 3070 PC:**
```bash
./scripts/start-vlm-cuda.sh
```

### 2. Start the whole cluster from your main Mac

Edit the hostnames at the top of `scripts/start-cluster.sh`, then:

```bash
./scripts/start-cluster.sh
```

SSHs into all three nodes, starts their servers, waits 15s, health-checks each one.

### 3. Start the router

```bash
./scripts/start-router.sh
```

First run copies `config.example.yaml` → `config.yaml`. Edit with your hostnames and API keys, then run again.

### 4. Point your agents at the router

In swarm-controller `config.json`:
```json
{
  "llm_providers": {
    "minimax": {
      "base_url": "http://localhost:8090/v1",
      "model": "MiniMax-M3",
      "format": "openai"
    }
  }
}
```

That's it. The router handles everything else.

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit:

```yaml
listen:
  host: "0.0.0.0"
  port: 8090

backends:
  minimax:
    base_url: "https://api.minimax.chat/v1"
    models: ["MiniMax-M3"]
    auth_env: "MINIMAX_API_KEY"      # reads key from environment variable
    weight: 2
    tags: ["text"]
    quota:
      window_seconds: 18000          # 5-hour window
      max_requests: 3000

  vlm-m4-1:
    base_url: "http://m4-1.local:8081/v1"
    models: ["Qwen2.5-VL-7B-Instruct-4bit"]
    tags: ["vision", "vlm"]
    weight: 1

  vlm-3070:
    base_url: "http://3070.local:11434/v1"
    models: ["Qwen2.5-VL-7B-Instruct-4bit"]
    tags: ["vision", "vlm"]
    weight: 2                        # 3070 gets 2x traffic share

routing:
  vision_backends: [vlm-m4-1, vlm-m4-2, vlm-3070]
  default_backends: [minimax, kimi, opencode]
  task_type_backends:
    bug:     [kimi, opencode, minimax]
    feature: [minimax, opencode, kimi]
    qa:      [minimax, opencode]
```

See `config.example.yaml` for the full example with all backends and task types.

## swarm-controller integration

shrimp-router is designed to be the LLM gateway for [swarm-controller](https://github.com/shrimplabs/swarm-controller). Agents send all LLM calls to it via `X-Task-Type` and `X-Phase` headers so routing, quota, and fallback are handled centrally.

### Setup

```bash
# Clone alongside swarm-controller
git clone https://github.com/shrimplabs/shrimp-router.git ~/workspace/shrimp-router
cd ~/workspace/shrimp-router
python3 -m venv .venv && .venv/bin/pip install -e .
cp config.example.yaml config.yaml   # then edit with your API keys
```

In swarm-controller `.env`:
```bash
MINIMAX_API_KEY=sk-...
KIMI_API_KEY=sk-...
OPENCODE_API_KEY=sk-...
OPENAI_API_KEY=sk-...    # same as OPENCODE_API_KEY — required by headroom
```

In swarm-controller `config.json`, point the minimax provider at the router:
```json
{
  "llm_providers": {
    "minimax": {
      "base_url": "http://localhost:8090/v1",
      "model": "MiniMax-M3"
    }
  }
}
```

`launch.sh` in swarm-controller will start shrimp-router automatically if it finds it at `~/workspace/shrimp-router` (override with `SHRIMP_ROUTER_DIR`).

### Phase pipeline routing

The swarm sends `X-Phase` headers to route different pipeline phases to different models:

```
X-Phase: plan    → opencode-plan  (Kimi K2.6 — best agentic reasoning)
X-Phase: scout   → opencode-scout (DeepSeek V4 Flash — cheap, 1M ctx)
X-Phase: work    → minimax        (MiniMax M3 — proven for implementation)
```

Configure in `config.yaml`:
```yaml
routing:
  phase_backends:
    plan:   [opencode-plan, minimax]
    scout:  [opencode-scout, minimax]
    work:   [minimax]
  task_type_backends:
    bug:      [kimi, opencode-plan, minimax]
    feature:  [minimax, opencode-plan, kimi]
    research: [opencode-scout, minimax]
    qa:       [minimax]
```

## Task-type routing

Add an `X-Task-Type` header and the router picks the preferred backend:

```
X-Task-Type: bug       → Kimi first (fast, cheap)
X-Task-Type: feature   → MiniMax first (large context)
X-Task-Type: research  → OpenCode DeepSeek Flash (cheapest)
X-Task-Type: qa        → MiniMax (vision capable)
```

Falls back automatically if the preferred backend is rate-limited or at quota.

## Vision routing

Any request containing an `image_url` content part is automatically routed to the VLM node pool. The pool uses weighted round-robin — set `weight: 2` on the 3070 to give it twice the share of requests vs each M4.

## Quota tracking

Each backend has a sliding-window counter. When a backend hits `max_requests` or returns a 429, it's skipped for the remainder of the window automatically.

Check status:
```bash
curl http://localhost:8090/health
```

```json
{
  "ok": true,
  "backends": {"minimax": true, "kimi": true},
  "quota": {
    "minimax": {"used": 1240, "remaining": 1760, "rate_limited": false},
    "kimi":    {"used": 430,  "remaining": 1570, "rate_limited": false}
  }
}
```

## Node scripts

| Script | Run on | Does |
|--------|--------|------|
| `scripts/start-vlm-apple.sh` | M4 Mac Mini | Installs mlx_vlm if needed, starts vision server on `:8081` |
| `scripts/start-vlm-cuda.sh` | 3070 PC | Installs ollama if needed, pulls model, serves on `:11434` |
| `scripts/start-cluster.sh` | Main Mac | SSHs into all nodes, starts servers, health-checks after 15s |
| `scripts/start-router.sh` | Main Mac | Starts the gateway on `:8090` |

## Backend wire formats

Each backend has two format fields that control how requests and responses are translated:

| Field | Values | Default | Purpose |
|-------|--------|---------|---------|
| `format` | `anthropic`, `openai` | `anthropic` | Wire format the backend **expects** for requests |
| `response_format` | `anthropic`, `openai`, `auto` | `auto` | Format the backend **returns** in responses |

`auto` means the router infers from `format`: openai backends return OpenAI responses, anthropic backends return Anthropic responses.

The `/v1/chat/completions` endpoint always returns OpenAI format to callers. The `/v1/messages` endpoint always returns Anthropic format. Translation happens automatically based on these fields.

**Example: OpenCode Go (OpenAI-compatible backend behind Anthropic caller)**
```yaml
opencode:
  base_url: "http://localhost:8886/v1"   # headroom proxy
  models: ["kimi-k2"]
  format: "openai"           # send OpenAI /chat/completions
  response_format: "openai"  # backend returns OpenAI format
  auth_env: "OPENCODE_API_KEY"
```

> **Note for OpenCode Go users**: headroom's `anyllm/openai` backend reads `OPENAI_API_KEY`,
> not `OPENCODE_API_KEY`. Set both in your `.env`:
> ```bash
> OPENCODE_API_KEY=sk-...
> OPENAI_API_KEY=sk-...   # same value — required alias for headroom
> ```
> Or in your `launch.sh`: `OPENAI_API_KEY="$OPENCODE_API_KEY" headroom proxy ...`

## Circuit breakers

Each backend can have an independent circuit breaker to avoid hammering a down provider:

```yaml
backends:
  minimax:
    base_url: "https://api.minimax.chat/v1"
    models: ["MiniMax-M3"]
    circuit_breaker:
      failure_threshold: 3   # trip after 3 consecutive failures
      cooldown_seconds: 60   # stay open for 60s, then probe
```

When a breaker trips, the backend is skipped for `cooldown_seconds`. After cooldown, one probe request is allowed through — success closes the breaker, failure restarts the cooldown.

Circuit breaker state is visible in `/health`:
```json
{
  "circuit_breakers": {
    "minimax": {"state": "closed", "failure_count": 0}
  }
}
```

## Health check timeout

```yaml
backends:
  slow-backend:
    base_url: "http://remote:8080/v1"
    health_check_timeout_seconds: 5.0   # default: 3.0
```

## Using headroom as a caching/token proxy

[headroom](https://github.com/headroom-ai/headroom) sits between shrimp-router and upstream APIs to provide prompt caching, token tracking, and rate limit smoothing:

```
agents → shrimp-router :8090 → headroom :8888 → MiniMax API
                              → headroom :8886 → OpenCode Go
```

In `config.yaml`, point backends at headroom instead of the upstream URL:
```yaml
backends:
  minimax:
    base_url: "http://localhost:8888/v1"   # headroom, not api.minimax.chat
    format: "anthropic"

  opencode-plan:
    base_url: "http://localhost:8886/v1"   # headroom openai proxy
    format: "openai"
    auth_env: "OPENCODE_API_KEY"
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `MINIMAX_API_KEY` | MiniMax auth |
| `KIMI_API_KEY` | Kimi auth |
| `OPENCODE_API_KEY` | OpenCode Go auth |
| `OPENAI_API_KEY` | Required alias for headroom's anyllm/openai backend (set to same value as `OPENCODE_API_KEY`) |
| `OPENROUTER_API_KEY` | OpenRouter auth (pay-per-use fallback) |
| `SHRIMP_ROUTER_CONFIG` | Path to config.yaml (default: `./config.yaml`) |
| `M4_1_HOST` | Hostname for first M4 mini (default: `m4-1.local`) |
| `M4_2_HOST` | Hostname for second M4 mini (default: `m4-2.local`) |
| `PC_3070_HOST` | Hostname for 3070 PC (default: `3070.local`) |
| `VLM_SSH_USER` | SSH username for cluster script (default: current user) |
