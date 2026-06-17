# FirmBench

Project planning for the **HUD Frontier/RSI RL Environments Hackathon** (HUD W25 x YC).
Goal: a verifiable RL environment for **agentic collaboration** — a team of agents runs
a business together, graded automatically.

Two candidate plans (pick one to build):

- **[PLAN-original-regret.md](./PLAN-original-regret.md)** — *Collaborative agent-firm,
  retail/vending sim.* Pure-numeric levers, **regret-vs-optimal reward** on paired seeds.
  Cleanest, lowest-variance, most defensible verifiable signal. Strongest "RL environment"
  story. Headline metric: **coordination tax**.

- **[PLAN-saas-hybrid.md](./PLAN-saas-hybrid.md)** — *SaaS startup, artifact-hybrid.*
  Agent C-suite (CEO/CTO/CMO/CFO) makes decisions AND writes real artifacts (specs, copy,
  pricing, investor updates) scored by LLM rubrics that feed the sim. More realistic and
  flashier, but trades away the exact regret guarantee (valuation is the hard reward anchor).

Both share the same core machinery (deterministic simulator + partial-observability agent
roles + computable reference baselines).
