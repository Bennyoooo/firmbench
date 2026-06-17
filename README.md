# FirmBench

Project planning for the **HUD Frontier/RSI RL Environments Hackathon** (HUD W25 x YC).
Goal: a verifiable RL environment for **agentic collaboration** — a team of agents runs
a business together, graded automatically by profit.

## Chosen design

- **[PLAN-market-discovery.md](./PLAN-market-discovery.md)** — *Market-discovery firm
  (single-agent).* One agent must reverse-engineer a hidden demand structure in a numeric
  user population (via experiments), then build + market + price the right product. **The
  LLM judge is a translator, not a grader** — it turns artifacts (ad copy, specs) into
  parameters that drive a deterministic market. **Reward = profit.** Primary diagnostic:
  *discovery efficiency* (regret vs a computable oracle). Multi-agent + *coordination tax*
  = future stretch.

## Earlier candidate (kept for reference)

- **[PLAN-original-regret.md](./PLAN-original-regret.md)** — first version:
  collaborative agent-firm on a retail/vending sim with pure-numeric levers and
  regret-vs-optimal reward. Cleanest verifiable signal; superseded by the
  market-discovery design, which keeps the regret guarantee *and* adds realistic
  artifact-driven market dynamics.

All versions share the same core machinery: a deterministic simulator + partial-
observability agent roles + computable reference baselines.

## Code

- **[sim.py](./sim.py)** — Phase 1: the deterministic market-discovery simulation core
  (pure stdlib, no deps). Contains `generate_world`, the gym-like `FirmEnv`
  (`reset`/`step`, reward = round profit), an `OraclePolicy` reference, a `NaivePolicy`
  floor, and a `ScriptedExperimenter` that probes demand → discovers the hidden
  pain→feature mapping → exploits.

Run the learnability check:

```bash
python3 sim.py
```

Expected shape (means over seeds): **naive loses money << scripted experimenter <=
oracle** — i.e. the environment is learnable and specifically rewards experimentation,
not spam or luck.

- **[agent.py](./agent.py)** — Phase 2: single-agent LLM harness. Drives `FirmEnv` with
  structured-JSON actions (robust across cheap open models) via the Fireworks
  OpenAI-compatible API. Same `.reset()/.act()` interface as the policies, so it reuses
  the runner + oracle. Falls back to the scripted baseline if no key is set (wiring check).

Run the LLM agent (cheap Fireworks model):

```bash
pip install -r requirements.txt
export FIREWORKS_API_KEY=...
export FIREWORKS_MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct   # optional
python3 agent.py
```

Next: leaderboard over several models/seeds, then Phase 3 (NL artifact translators).
