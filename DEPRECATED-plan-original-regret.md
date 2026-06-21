# HUD Frontier/RSI RL Environments Hackathon — Project Plan & Discussion

> Working doc capturing the full brainstorm + final plan for the HUD (W25) x YC
> RSI hackathon. Project: a **verifiable RL environment for agentic collaboration**
> ("collaborative agent-firm"), delivered as an environment + eval/leaderboard.

---

## 0. Hackathon context

**HUD Frontier/RSI RL Environments Hackathon** (HUD W25 + YC co-hosted).
- Focus: frontier RL environments, post-training datasets, evals, RFT workflows.
- Thesis: *"You can improve models at anything you can verify. The only question
  left is: what will you teach them?"*
- Encouraged domains: coding, ML research, robotics, manufacturing, gaming,
  **agentic collaboration**, **autonomous businesses**.
- Format: 24h build. Prizes incl. guaranteed YC interview, robo-dog, Mac Minis, etc.
- Sponsors incl. HUD (env platform), Modal (GPU/sandboxes), Fireworks (RFT on open
  models), DeepMind, Daytona, Anthropic, Exa, Hillclimb, Protege, Antim Labs (physical AI sim).

---

## 1. Background research: TheAgentCompany (the jumping-off point)

Studied `TheAgentCompany` (CMU, WebArena team) to understand what a good
task/eval environment looks like.

### What it is
- A **task-completion benchmark**, not a business simulation. The "company" is a
  realistic *backdrop* (tools, data, coworkers), not the thing being measured.
- 175 tasks, each a Docker image. Agent works against 4 real self-hosted services
  (GitLab, ownCloud, Plane, RocketChat) + local `/workspace`.
- Roles: SDE (69), HR (29), PM (28), Admin (15), DS (14), Finance (12), misc (~8).
- 41 tasks involve LLM-backed NPC coworkers (RocketChat, built on Sotopia).

### How it evaluates (the key learnings)
- **Result/execution-based grading**: an encrypted `evaluator.py` inspects the
  *actual final state* of services/files after the agent finishes. Not step-matching.
- **Weighted checkpoints with partial credit**: `grade_checkpoints()` returns a
  `Result` of `Checkpoint(total, result)` objects → you see *how far* an agent got.
- Grading mechanisms across tasks:
  - Deterministic state checks (RocketChat 80, GitLab 31, ownCloud 25, Plane 13)
  - Execution-based (run the DB/server, assert exact values)
  - LLM-as-judge (48 tasks; `evaluate_with_llm(content, predicate)` → yes/no; 6 use vision)
  - Trajectory-based (last resort; all 175 accept a trajectory file)
- Scoring strategies: default sum; `bonus_for_completing_final`; `bonus_for_completing_any`.
- Anti-cheat: evaluator ships encrypted (Fernet), decrypted at grade time with
  `DECRYPTION_KEY='theagentcompany is all you need'`. `/utils` off-limits to agent.
- Needs an **environment LLM** (separate from the agent under test) to power NPCs +
  LLM judges. Baseline used Claude 3.5 Sonnet. OpenHands baseline capped at
  100 iterations / $4 per task.

### Key takeaway / limitation
- It measures **"can an agent be a reliable employee on bounded, assigned tasks?"**
- It does **NOT** measure: business KPIs (revenue/profit), long-horizon/cumulative
  state (tasks are isolated, services reset between them), prioritization, or true
  multi-agent collaboration (NPCs are scripted, not collaborators).
- HUD's platform structures environments almost identically (task spec + evaluator +
  sandbox), so all of this knowledge transfers directly.

---

## 2. Idea exploration

### Guiding principle
A winning RL env = **cheap, automatic, ground-truth reward** + **tasks frontier
models partially fail** + **buildable/resettable in 24h**.

### Why "autonomous business" felt off at first
Real business success is long-horizon, noisy, and not automatically verifiable —
you can't cleanly compute "did this agent run a good company." **Fix: put a
deterministic simulator underneath it** so profit is computable and resettable.

### Candidate directions considered
1. **Autonomous business sim** (chosen seed) — simulator-backed economic env;
   reward = profit/regret over a long horizon. Tests multi-step planning.
2. **Autonomous ML research (RSI)** — agent improves ML code to max a held-out
   metric; reward = the score. Most on-theme (RSI). Prior art: MLE-bench, RE-Bench.
3. **Multi-agent collaboration** — multiple real agents coordinate to ship a
   deliverable; reward = tests pass. Fixes TheAgentCompany's single-agent gap.
4. **Verifiable finance/ops env** — financial modeling / OR / scheduling with exact
   ground-truth answers. Underexplored, trivially gradeable.

---

## 3. Vending-Bench reference (resolving the "luck" question)

- **Vending-Bench** (Andon Labs, early 2025) is a **simulation** — a single LLM
  agent runs a virtual vending-machine business (search wholesalers, email
  suppliers, order inventory, set prices, pay daily fee). Sales come from a demand
  model; supplier/email responses from a sub-LLM. Metric = **net worth** over a very
  long horizon. Distinct from **Project Vend** (Andon + Anthropic), the *real*
  physical-fridge experiment with the famous meltdowns.
- Headline finding: models do sub-tasks fine but **collapse over long horizons**
  (lose track of cash/inventory, stop ordering, spiral). Huge run-to-run variance.
- Capability measured: **long-horizon coherence** (state-tracking, memory,
  consistent strategy, error recovery) — NOT one-shot reasoning.

### "Isn't it just luck?" — how to make reward reflect skill
1. **Demand is learnable, not unknowable** — agent sees its own sales each step, so
   a capable agent experiments and infers the demand curve. That *is* the skill.
2. **Long horizon** averages out single-event luck.
3. **Multiple runs** → report distribution, not single numbers.
4. Honest caveat: Vending-Bench variance is *still huge* — its biggest weakness, and
   the thing our project improves on.

### The fix that turns a "casino" into a capability measure
- **Reward = regret vs. optimal**: since we own the simulator, compute the best
  achievable profit *given the demand that actually occurred*; grade on
  `optimal − agent`. Luck cancels. **Single most important design decision.**
- **Paired seeds (common random numbers)**: run every model on identical demand
  realizations → compare skill, not luck draws.
- **Average over many seeds** for variance reduction.

### Is Vending-Bench multi-agent?
No — it's single-agent (the "world" is scripted/sub-LLM, not a collaborator).
But a business sim naturally **extends** into multi-agent: competition (rival firms)
or collaboration (a firm of specialized agents).

---

## 4. FINAL PLAN — Collaborative Agent-Firm environment

**Decisions:** scope = *collaborative agent-firm*; deliverable = *env + eval/leaderboard*
(no training run required for the demo).

### Pitch (one-liner)
> TheAgentCompany showed a *single* agent can do isolated tasks. The open question is
> whether a *team* of agents can run an operation **together**. We built the first
> verifiable RL environment for agentic collaboration with a **computable optimal**,
> so coordination quality becomes a clean, low-variance reward — and we measure the
> **"coordination tax"** frontier models pay versus a perfectly-informed planner.

### Capability measured
**Information-sharing and joint planning under partial observability** — precise,
verifiable, and exactly what TheAgentCompany's scripted-NPC setup can't test.

### The simulator (own it, keep it small)
- Retail/vending micro-business, ~30 days, 3 products.
- Each day: `demand = f(price, popularity, seed-fixed noise)`;
  `sales = min(demand, inventory)`;
  `P&L = revenue − COGS − holding cost − order cost − daily fee`.
- Deterministic given a seed, resettable, cheap (pure Python, no LLM in the engine).

### The firm = 3 specialized agents with deliberately split observability
(If any one agent saw everything, coordination wouldn't be tested.)
- **Buyer** — sees supplier catalog, prices, lead times. NOT demand signals.
- **Pricer** — sees demand signals / past sales / competitor prices. NOT inventory/cash.
- **Ops** — sees current inventory + cash. NOT catalog or demand.
- They **must communicate** over a shared append-only message log. Bad coordination
  → over-ordering, stockouts, pricing for stock that never arrives. Firm profit
  reflects coordination quality.

### Reward (what wins)
- **Regret = optimal_profit − firm_profit**, on **paired seeds**. (If exact optimal
  is hard, use a centralized omniscient planner as reference — honest and sufficient.)
- **Coordination tax** (headline chart) =
  `omniscient_single_agent_profit − multi_agent_firm_profit`.
  How much the firm loses purely to the information boundary. This number *is* the result.

### Leaderboard axes
Frontier models (Claude / GPT / Gemini / an open model) × team size ×
communication protocol (free-form chat vs. structured handoffs).

### 24-hour schedule
| Hours | Build |
|-------|-------|
| 0–4 | Sim core: deterministic engine, demand model, P&L, seeding + reference/optimal baseline (LP or small-state DP/brute force). |
| 4–10 | Agent harness: 3 role agents, partial-obs views, shared message bus, round-robin turn loop. Wrap as a **HUD environment** (task spec + evaluator = profit/regret). |
| 10–16 | Run frontier models on fixed seed sets; collect trajectories; compute regret + coordination tax; build leaderboard. |
| 16–22 | Demo polish: leaderboard UI + replay viewer ("watch agents negotiate, then succeed or melt down"); failure-mode gallery. |
| 22–24 | Slides + pitch. |

### Risks & mitigations
- **True optimal is hard** → ship a strong *reference policy* (omniscient centralized
  greedy/LP), report normalized score.
- **Multi-agent orchestration is fiddly** → keep it simple: round-robin turns + one
  append-only shared log. No fancy protocol.
- **LLM cost/time** → short horizon (30 days), 3 products, cache calls, few seeds.
- **Coordination not actually forced** → pressure-test the observability split early;
  if a single agent can solve it, tighten the partition.

### Why it scores well
Verifiable (computable regret) · novel (fills the exact TheAgentCompany gap) · great
live demo (agents negotiating + coordination-tax chart) · clean stretch ("env is
RFT-ready; next we fine-tune for coordination" — one config away, not needed for demo).

### Name options
**FirmBench**, **CoordTax**, **Boardroom**.

---

## 5. Immediate next step
Scaffold the simulator core (the 0–4h block): deterministic engine + P&L + seedable
demand model + omniscient reference baseline. Everything else hangs off it.
