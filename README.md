# FirmBench

A verifiable RL environment for **autonomous business agents** — built for the
**HUD Frontier/RSI RL Environments Hackathon** (HUD W25 x YC).

One agent runs a simulated firm: it must reverse-engineer a hidden demand structure
in a numeric user population by running experiments, then build the right product,
market it to the right people, and price it right. **Reward = profit**, graded on
secret held-out users the agent never saw feedback from.

## Quick Start (HUD)

```bash
# Install
pip install hud-python fastmcp openai
# or: uv sync

# Run against Claude (or any HUD-supported model)
cp .env.example .env   # fill in HUD_API_KEY
hud eval hud_tasks.py claude --task-ids market_discovery_seed42 -y --max-steps 30

# Deploy as a hosted environment
hud deploy .
```

## Quick Start (standalone, no HUD)

```bash
# Learnability check (no deps, no keys)
python3 sim.py

# Full pipeline: eval baselines → REINFORCE training → re-eval
python3 run.py

# LLM agent (needs Fireworks key)
pip install openai
export FIREWORKS_API_KEY=...
python3 agent.py
```

## Layout

```
sim.py          deterministic market sim: generate_world, FirmEnv, oracle/naive/scripted
agent.py        LLM agent harness (Fireworks, structured-JSON actions)
run.py          verifier (secret held-out + tripwires) + head-to-head eval + REINFORCE
hud_env.py      HUD v6 environment: MCP tools + verifier grading
hud_tasks.py    HUD task definitions (3 seeds)
Dockerfile.hud  deployable image
```

## How it works

**The hidden world** (randomized per episode): 8 pain points, 8 features, a hidden
`solves: pain → feature` mapping, and 5,000 numeric users with skewed pain popularity
and varying willingness-to-pay.

**The agent's tools** (via MCP):
- `probe_market(target_pains, spend)` — cheap discovery campaign → returns diagnostics
- `build_feature(feature_id)` — build a feature ($300)
- `set_price(price)` — set the product price
- `run_campaign(target_pains, spend)` — full marketing push
- `get_state()` — current round, cash, price, built features
- `end_round()` — commit actions, advance to next round

**Grading** (deterministic, no LLM judge): 20% of users are held out — the agent never
gets campaign results from them. On episode end, the verifier replays every action on
those users and computes true held-out profit. A cheat tripwire flags policies whose
visible profit diverges from held-out profit.

**What the agent must learn:** the pain popularity distribution (biggest needs), the
pain→feature mapping (build the right things), and price elasticity. All via experiments.

## Leaderboard (HUD, 3 seeds)

| Model | Seed 42 | Seed 123 | Seed 7 | Mean Reward |
|---|---|---|---|---|
| **GPT-5 Mini** | 0.000 | 0.043 | 0.089 | **0.044** |
| Claude Sonnet 4.6 | 0.000 | 0.000 | 0.007 | 0.002 |
| Gemini 3.5 Flash | 0.000 | 0.000 | 0.000 | 0.000 |
| GPT-4o | 0.000 | 0.000 | 0.000 | 0.000 |
| *Scripted (reference)* | — | — | — | *~0.70* |
| *Oracle (reference)* | — | — | — | *~0.83* |

All frontier models are far below the scripted baseline — the environment is genuinely
hard and there's a wide gap for RL training to close.

## Docs

- **[PLAN.md](./PLAN.md)** — full design spec + progress tracker
- `DEPRECATED-plan-*.md` — earlier design candidates (kept for reference)
