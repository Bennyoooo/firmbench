# FirmBench

Project planning for the **HUD Frontier/RSI RL Environments Hackathon** (HUD W25 x YC).
Goal: a verifiable RL environment for **agentic collaboration** — a team of agents runs
a business together, graded automatically by profit.

## Chosen design

- **[PLAN-market-discovery.md](./PLAN-market-discovery.md)** — *Market-discovery firm.*
  4 agents (Builder / Marketer / Pricer / Coordinator) must reverse-engineer a hidden
  demand structure in a numeric 10,000-user population, then build + market + price the
  right product. **The LLM judge is a translator, not a grader** — it turns artifacts
  (ad copy, specs) into parameters that drive a deterministic market. **Reward = profit.**
  Diagnostics (both computable because the world is numeric): *discovery efficiency*
  (regret vs oracle) and *coordination tax*.

## Earlier candidate (kept for reference)

- **[PLAN-original-regret.md](./PLAN-original-regret.md)** — first version:
  collaborative agent-firm on a retail/vending sim with pure-numeric levers and
  regret-vs-optimal reward. Cleanest verifiable signal; superseded by the
  market-discovery design, which keeps the regret guarantee *and* adds realistic
  artifact-driven market dynamics.

All versions share the same core machinery: a deterministic simulator + partial-
observability agent roles + computable reference baselines.
