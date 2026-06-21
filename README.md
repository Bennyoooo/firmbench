# FirmBench

A verifiable RL environment for **autonomous business agents** — built for the
**HUD Frontier/RSI RL Environments Hackathon** (HUD W25 x YC).

One agent runs a simulated firm: it must reverse-engineer a hidden, persona-structured
demand population by running experiments, then build the right features, target the right
segments on the right channel, and price for retention. **Reward = discovery efficiency**
(profit ÷ a strong oracle reference, clipped to [0,1]) — comparable across the
domain-randomized worlds.

> **Phase A (current):** the market is now a hybrid persona population — hidden segments
> with correlated pains, per-user elasticity, channel preference, a quality bar, and a
> subscription/churn (LTV) model — all behind `Config` flags so `Config()` is still
> exactly v1 and `Config.phase_a()` turns the full market on. See [PLAN.md](./PLAN.md).

## Quick Start (HUD)

```bash
# Install
pip install hud-python fastmcp openai
# or: uv sync

# Run against Claude (or any HUD-supported model)
cp .env.example .env   # fill in HUD_API_KEY
hud eval tasks.py claude --task-ids market_discovery_seed42 -y --max-steps 80

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

## Real RFT — "the curve bends"

`rft.py` runs **rejection-sampling fine-tuning** (expert iteration / STaR): generate
rollouts → grade with the held-out verifier → keep the best non-flagged, profitable
trajectory per world → fine-tune the model on those turns → re-evaluate → repeat. The
verifier is both the reward *and* the curator of the training set — only profitable,
non-cheating episodes become SFT data.

```bash
# Offline validation — no key needed. A mock model whose skill rises each iteration
# drives the whole rollout→filter→dataset→eval loop and prints the bending curve.
python3 rft.py --selftest --iterations 3
#   iter 0   -733.5
#   iter 3  15524.1  ######################   (oracle ceiling 26321)

# Real run — needs Fireworks fine-tuning access.
export FIREWORKS_API_KEY=...
firectl signin                        # https://docs.fireworks.ai/tools-sdks/firectl
python3 rft.py --run --iterations 2 --rollouts 4 \
    --model accounts/fireworks/models/llama-v3p2-3b-instruct
```

Writes `rft_out/sft_iter*.jsonl` (the curated SFT datasets) and `rft_out/curve.json`
(the eval curve). Swap `--model` to Llama 3.2 1B for the absolute cheapest e2e.

## HUD-native on-policy RL — `rft_hud.py`

`rft.py` is off-policy imitation (reject → SFT on winners). `rft_hud.py` is **true
on-policy RL through HUD's own training service** (`hud.train`, Tinker-backed): roll
out the env through the HUD gateway → each `hud.Run` carries token-level samples +
the env's reward → one **grouped policy-gradient step** (GRPO: advantages normalized
within each world's rollout group) → `optim_step` checkpoints and promotes the new
weights, so the next rollout is automatically on-policy. No datasets, no redeploy.

```
rft.py     : rollout -> reject -> SFT(cross_entropy) on winners -> redeploy   (off-policy)
rft_hud.py : rollout -> grouped policy gradient -> checkpoint promote          (on-policy)
```

```bash
# Offline — no HUD, no keys. A mock gateway+trainer whose skill rises per step
# drives the real loop (per-seed grouping, GRPO group checks, curve), graded with
# the env's true disc_eff reward via the local sim.
python3 rft_hud.py --selftest --steps 4
#   step 0  0.059
#   step 4  0.660   ##########################   (oracle ceiling 1.000)

# Real run — on-policy RL through HUD.
hud login
hud models fork accounts/fireworks/models/<base>      # -> a trainable gateway slug
python3 rft_hud.py --run --model accounts/<team>/models/<forked> \
    --steps 5 --group-size 8 --loss importance_sampling
```

Built-in losses: `importance_sampling` (default on-policy PG), `ppo`, `cispo`, `dro`
(discover the live set via `TrainingClient.available_losses()`). Writes
`rft_hud_out/curve.json`. Built-in losses run server-side — no local torch needed
(only `forward_backward_custom` requires `pip install 'hud-python[train]'`).

## Layout

```
sim.py          deterministic Phase A market: generate_world (segments/channels/LTV behind
                Config flags), FirmEnv, oracle/naive/scripted, ablation_gate, replay_profit
agent.py        LLM agent harness (Fireworks, structured-JSON actions)
run.py          discovery-efficiency grader (Verifier) + head-to-head eval + toy REINFORCE
rft.py          real RFT: rejection-sampling fine-tuning on Fireworks (+ offline selftest)
rft_hud.py      HUD-native on-policy RL (GRPO via hud.train; gateway rollouts + offline selftest)
scorer.py       NL artifact translator: ad copy -> craft, spec -> quality (LLM, fast_mode)
renderer.py     renders ad copy / specs to HTML artifacts
env.py          HUD v6 environment: MCP tools + discovery-efficiency grading
tasks.py        HUD task definitions (3 seeds)
Dockerfile.hud  deployable image
```

## How it works

**The hidden world** (randomized per episode): 8 pain points, 8 features, a hidden
`solves: pain → feature` mapping, and ~5,000 users drawn from hidden **segments**
(personas). Each segment clusters certain pains and has its own willingness-to-pay, price
elasticity, preferred marketing **channel**, quality bar, and churn rate — with per-user
noise on top (hybrid population). Every latent is behind a `Config` flag: `Config()` is
exactly the v1 market; `Config.phase_a()` turns the full market on.

**The agent's tools** (via MCP):
- `probe_market(target_pains, spend, ad_copy?, channel?)` — discovery campaign → audience,
  tries, purchases, revenue, and bounce reasons (`bounced_quality` vs `bounced_price`)
- `build_feature(feature_id?, spec?)` — build a feature ($300); a better NL spec → higher
  implementation quality (which both converts and retains)
- `set_price(price)` — drives both conversion and churn
- `run_campaign(target_pains, spend, ad_copy?, channel?)` — full marketing push
- `get_state()` — round, horizon, cash, price, built features, pain/feature names
- `end_round()` — commit actions, advance to next round

It's a **subscription business**: subscribers pay every round (recurring revenue) and
churn if price is too high or quality too low, so the agent optimizes lifetime value.

**Grading** (deterministic, no user holdout): reward = **discovery efficiency =
profit ÷ oracle**, clipped to [0,1]. The env is execution-based (it computes profit — the
agent can't fake it) and domain-randomized per episode, so generalization is measured by
**held-out eval seeds**, not a within-episode user split. (The earlier secret-held-out +
tripwire scheme was dropped: it flagged honest agents and caught nothing an
execution-based env can't already prevent.)

**What the agent must learn:** which pains are biggest (segment demand), the pain→feature
mapping, which channel reaches which segment, price elasticity, and quality/retention —
all via experiments.

## Eval

**Phase A policy leaderboard** (10 held-out seeds; reward = discovery efficiency = profit ÷ oracle):

| Policy | disc.eff | mean profit |
|---|---|---|
| naive (no discovery) | 0.021 | 14,346 |
| scripted (experimenter) | 0.066 | 46,105 |
| oracle (LTV-optimal reference) | 1.000 | 691,176 |

Two skills stack here. Disciplined **discovery** (scripted vs naive) is ~3× better — but
both sit far below the oracle, because the **subscription/LTV** game (acquire-then-coast,
price for retention, don't burn cash on a saturated finite market) is a deeper skill
neither heuristic captures. That leaves wide headroom for RL — scripted reaches only ~7%
of the ceiling. Run it yourself: `python3 -c "from sim import ablation_gate; ablation_gate()"`.

**Frontier-model leaderboard (v1 market, HUD, 3 seeds)** — all far below the scripted
baseline, leaving wide headroom for RL:

| Model | Seed 42 | Seed 123 | Seed 7 | Mean |
|---|---|---|---|---|
| **GPT-5 Mini** | 0.000 | 0.043 | 0.089 | **0.044** |
| Claude Sonnet 4.6 | 0.000 | 0.000 | 0.007 | 0.002 |
| Gemini 3.5 Flash | 0.000 | 0.000 | 0.000 | 0.000 |
| GPT-4o | 0.000 | 0.000 | 0.000 | 0.000 |

(Re-running the model leaderboard on the **Phase A** env needs `hud deploy .` first — the
local code is Phase A, the hosted image is still v1.)

## Docs

- **[PLAN.md](./PLAN.md)** — full design spec + progress tracker
- `DEPRECATED-plan-*.md` — earlier design candidates (kept for reference)
