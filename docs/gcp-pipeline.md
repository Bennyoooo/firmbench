# SaaSBench — Google Cloud / Vertex AI pipeline

The GCP counterpart to the Fireworks pipeline (`agent.py` + `rft.py`). Same environment,
same verifier, same RFT recipe — the only difference is the inference + fine-tuning
backend: **Gemini (and Gemma) on Vertex AI**.

| File | Fireworks equivalent | What |
|------|----------------------|------|
| `agent_vertex.py` | `agent.py` | Single-agent inference harness (Gemini/Gemma on Vertex AI) |
| `rft_gcp.py` | `rft.py` | Reward-filtered **SFT** via Vertex AI supervised tuning (+ offline `--selftest`) |
| `rl_grpo.py` | — | **True RL** via GRPO on Gemma open weights (+ offline `--selftest`) |

All three are **self-contained** (import only from `sim.py` / `run.py`, with `rl_grpo.py`
also reusing prompt helpers from `agent_vertex.py`) so they evolve independently of the
Fireworks pipeline.

### SFT vs RL — which is which

- `rft_gcp.py` (and `rft.py`) do **reward-*filtered* SFT** (STaR / expert iteration):
  sample trajectories, keep the winners, supervised-finetune on them. Reward is only a
  *filter* — it never pushes *down* bad actions, and there's no advantage or KL term.
  Mechanically it's SFT. Great as a warm-start.
- `rl_grpo.py` does **true policy-gradient RL** (GRPO): group-relative advantage +
  PPO-style clipped surrogate + KL-to-reference. Good episodes are reinforced, bad ones
  actively suppressed. This requires **open weights** (Gemma) — you can't gradient-RL a
  closed model like Gemini, and Vertex's managed RLHF is deprecated.
- Standard recipe: **SFT init → RL** (warm-start with `rft_gcp.py`, then `rl_grpo.py`).

## Setup

```bash
pip install -r requirements-gcp.txt        # google-genai + google-cloud-storage
gcloud auth application-default login       # Application Default Credentials
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1    # optional (default us-central1)
export VERTEX_MODEL=gemini-2.0-flash-001    # optional; any tunable Vertex model
export GCS_BUCKET=your-bucket               # only needed for --run (tuning data staging)
```

(See `.env.gcp.example`.) Without these, both scripts fall back gracefully: the agent
runs the scripted baseline to verify wiring, and the RFT loop runs offline via `--selftest`.

## Inference (leaderboard)

```bash
# Standalone — runs the Vertex agent over a few seeds, reports profit vs oracle.
python3 agent_vertex.py
```

## RFT — "the curve bends"

```bash
# Offline validation — no GCP needed. A mock model whose skill rises each iteration
# drives the whole rollout -> filter -> dataset -> eval loop and prints the bending curve.
python3 rft_gcp.py --selftest --iterations 3
#   iter 0     274.5
#   iter 3    8396.6   ############   (oracle ceiling 26871)

# Real run — needs Vertex AI tuning access + a GCS bucket.
export GOOGLE_CLOUD_PROJECT=...  GCS_BUCKET=...
python3 rft_gcp.py --run --iterations 2 --rollouts 4 --model gemini-2.0-flash-001
```

Writes `rft_gcp_out/sft_iter*.jsonl` (curated chat datasets) and `rft_gcp_out/curve.json`.
The fine-tune step (`finetune_vertex`) converts winners to Vertex tuning JSONL, stages
them on GCS, launches a `client.tunings.tune(...)` job, polls to completion, and returns
the tuned model endpoint — which is fed straight back in as the next iteration's model.

## True RL (GRPO) — `rl_grpo.py`

Genuine policy-gradient RL with verifiable rewards: per world, sample a GROUP of full
episodes, grade each with the verifier, compute a group-relative advantage (no critic),
and update the policy with a clipped surrogate + KL. SaaSBench's verifier *is* the reward.

```bash
# Offline RL demo — no GPU/torch. The REAL GRPO loop optimizes a categorical policy over
# SaaSBench probe->exploit schedules; held-out reward bends as it learns the optimum.
python3 rl_grpo.py --selftest --iterations 40
#   iter  0   0.388
#   iter 40   0.520   (learns switch round k≈7; +34% on held-out seeds)

# Real GRPO on Gemma open weights — needs a GPU.
pip install -r requirements-rl.txt
export RL_BASE_MODEL=google/gemma-2-2b-it
python3 rl_grpo.py --run --iterations 30 --group-size 8 \
    --train-seeds 8 --eval-seeds 8
```

Writes `rl_grpo_out/grpo_curve.json` and (for `--run`) a `grpo_gemma_lora/` adapter.
Multi-turn credit assignment is trajectory-level: the episode's group-relative advantage
is broadcast across every turn's tokens.

### Running GRPO as a Vertex AI custom-training job

Vertex's managed RLHF is deprecated, so on-weights RL runs as a **custom-training job**
(your container, Vertex's GPU). Package `rl_grpo.py --run` into a training container and
submit with a GPU machine type (e.g. `g2-standard-12` + L4, or an A100), e.g.:

```bash
gcloud ai custom-jobs create --region="$GOOGLE_CLOUD_LOCATION" \
    --display-name=firmbench-grpo \
    --worker-pool-spec=machine-type=g2-standard-12,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=YOUR_IMAGE
```

The resulting LoRA adapter can be merged and deployed to a Vertex endpoint, then used for
inference via the OpenAI-compatible path below.

## Running it through HUD

The HUD environment (`env.py` / `tasks.py`) is backend-agnostic — it just needs a model
endpoint. Two ways to point HUD at Vertex:

**1. Gemini API (simplest, native HUD agent):**
```bash
hud eval tasks.py gemini --model gemini-2.0-flash-001 \
    --task-ids market_discovery_seed42 -y --max-steps 80
```

**2. Vertex AI via the OpenAI-compatible endpoint** (uses your GCP project + ADC, and
   also serves a fine-tuned or Gemma Model Garden endpoint):
```bash
export OPENAI_API_KEY="$(gcloud auth print-access-token)"
LOC=${GOOGLE_CLOUD_LOCATION:-us-central1}
hud eval tasks.py openai_compatible \
    --base-url "https://${LOC}-aiplatform.googleapis.com/v1beta1/projects/${GOOGLE_CLOUD_PROJECT}/locations/${LOC}/endpoints/openapi" \
    --model "google/${VERTEX_MODEL:-gemini-2.0-flash-001}" \
    --task-ids market_discovery_seed42 -y --max-steps 80
```
For a model produced by `rft_gcp.py --run`, pass its tuned-model endpoint as `--model`.

## Notes on Gemma

Gemini is the managed, directly tunable path on Vertex (closest analog to Fireworks'
`firectl` SFT). Gemma open models are served by deploying them from **Vertex Model
Garden** to an endpoint; point `VERTEX_MODEL` / `--model` at that endpoint to use Gemma
for inference. Supervised tuning of Gemma weights is a Vertex custom-training job (out of
scope for the managed `tunings.tune` path used here).
