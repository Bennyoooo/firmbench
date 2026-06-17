# FirmBench — HUD Frontier/RSI RL Environments Hackathon

> Working doc capturing the full brainstorm + final plan for the HUD (W25) x YC
> RSI hackathon. Project: a **verifiable RL environment for agentic collaboration** —
> a team of agent-execs runs a **SaaS startup** (artifact-hybrid), delivered as an
> environment + eval/leaderboard.

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
  models), DeepMind, Daytona, Anthropic, Exa, Hillclimb, Protege, Antim Labs.

---

## 1. Background research: TheAgentCompany (jumping-off point)

Studied `TheAgentCompany` (CMU, WebArena team) to understand a good task/eval env.

### What it is
- A **task-completion benchmark**, not a business simulation. The "company" is a
  realistic *backdrop* (tools, data, coworkers), not the thing being measured.
- 175 tasks, each a Docker image. Agent works against 4 real self-hosted services
  (GitLab, ownCloud, Plane, RocketChat) + local `/workspace`.
- Roles: SDE (69), HR (29), PM (28), Admin (15), DS (14), Finance (12), misc (~8).
- 41 tasks involve LLM-backed NPC coworkers (RocketChat, built on Sotopia).

### How it evaluates (key learnings)
- **Result/execution-based grading**: an encrypted `evaluator.py` inspects the
  *actual final state* of services/files after the agent finishes. Not step-matching.
- **Weighted checkpoints with partial credit**: `grade_checkpoints()` returns a
  `Result` of `Checkpoint(total, result)` objects → you see *how far* an agent got.
- Mechanisms: deterministic state checks (RocketChat 80, GitLab 31, ownCloud 25,
  Plane 13), execution-based, LLM-as-judge (48; 6 vision), trajectory-based (last resort).
- Scoring strategies: default sum; `bonus_for_completing_final`; `bonus_for_completing_any`.
- Anti-cheat: evaluator ships encrypted (Fernet), decrypted at grade time. `/utils`
  off-limits to agent. Needs a separate **environment LLM** for NPCs + judges.

### Takeaway / limitation
- Measures **"can an agent be a reliable employee on bounded, assigned tasks?"**
- Does NOT measure business KPIs, long-horizon/cumulative state (tasks are isolated,
  services reset between them), prioritization, or true multi-agent collaboration
  (NPCs are scripted, not collaborators).
- HUD's platform structures environments almost identically (task + evaluator +
  sandbox), so this knowledge transfers directly.

---

## 2. Idea exploration

### Guiding principle
A winning RL env = **cheap, automatic, ground-truth reward** + **tasks frontier
models partially fail** + **buildable/resettable in 24h**.

### Why naive "autonomous business" fails
Real business success is long-horizon, noisy, not automatically verifiable.
**Fix: put a deterministic simulator underneath it** so outcomes are computable.

### Directions considered
1. Autonomous business sim (chosen) — simulator-backed; reward = profit/valuation.
2. Autonomous ML research (RSI) — improve ML code to max a metric. Most on-theme.
3. Multi-agent collaboration — agents coordinate to ship; reward = outcome.
4. Verifiable finance/ops — modeling / OR / scheduling with exact answers.

We merged #1 + #3: a **collaborative agent-firm**.

---

## 3. Vending-Bench reference (the "luck" question)

- **Vending-Bench** (Andon Labs, 2025) is a **simulation** — single LLM agent runs a
  virtual vending business; sales from a demand model, suppliers from a sub-LLM;
  metric = net worth over a long horizon. Distinct from **Project Vend** (Andon +
  Anthropic), the real physical-fridge experiment with the famous meltdowns.
- Finding: models do sub-tasks fine but **collapse over long horizons** (lose track of
  cash/inventory, stop ordering, spiral). Huge variance.
- Capability measured: **long-horizon coherence**, not one-shot reasoning.

### Making reward reflect skill, not luck
- Demand is *learnable* (agent sees its sales) → experimentation is the skill.
- Long horizon + multiple runs average out luck.
- **Regret vs. optimal** (own the sim → compute best achievable given realized demand)
  cancels luck. **Paired seeds (common random numbers)** compare skill on identical
  worlds. (NOTE: regret guarantee weakens under artifact-hybrid — see §4.)

---

## 4. FINAL PLAN — FirmBench: collaborative agent-firm running a SaaS startup

**Decisions:** business = **SaaS startup**; fidelity = **artifact-hybrid throughout**;
deliverable = **env + eval/leaderboard** (no training run needed for the demo).

### Why SaaS (vs vending / DTC)
- Vending = just *operations* (buying/pricing), not a real business.
- SaaS is the most realistic + relatable for a YC/HUD audience, and artifacts span the
  whole stack (specs, code, copy, pricing, investor updates) → "agents doing real work"
  everywhere. Engine is generic, so DTC brand is a later swap with no rewrite.

### Honest tradeoff of artifact-hybrid
Artifact space is open-ended → exact **regret-vs-optimal stops being computable**. We
keep the eval clean by anchoring the headline reward on a **hard deterministic number —
firm valuation** — and treating artifact-quality scores as *inputs* to the sim + as
interpretable sub-metrics, NOT as the reward itself. Headline reframed from "regret" →
**valuation leaderboard + coordination tax**.

### Pitch (one-liner)
> TheAgentCompany showed a *single* agent can do isolated tasks. The open question is
> whether a *team* of agents can run an operation **together**. FirmBench is a
> verifiable RL environment where an agent C-suite runs a SaaS startup — they make
> strategic calls AND produce real work artifacts — and we measure the **coordination
> tax** they pay versus a perfectly-informed planner.

### Capability measured
Multi-agent **strategic coordination + coherent cross-functional execution** — do the
CTO's roadmap, CMO's campaign, and CFO's pricing actually align?

### The collaborating agents (C-suite, partial observability)
- **CEO** — strategy, capital allocation, writes investor update; sees board pressure + runway.
- **CTO** — picks features from backlog, allocates eng capacity, writes feature spec/PRD;
  knows build costs + tech debt.
- **CMO** — sets channel marketing budget, writes ad/landing copy; knows channel CACs + brand.
- **CFO** — sets pricing tiers, manages burn/fundraising, writes financial plan; knows unit economics.

Coordinate via a shared company channel + a structured monthly planning doc. The
observability split forces info-sharing.

### The deterministic engine
Latent state stocks: `product_quality`, `brand_equity`, `users`, `MRR`, `cash`,
`team_size`, `tech_debt`. Monthly tick:
- Eng: features built (capacity − tech_debt) → ↑ product_quality, may ↑ tech_debt.
- Marketing: spend × channel_response × (1 + brand_equity) × copy_quality_score → new users.
- Brand: ↑ with sustained spend + product_quality + copy_quality; decays over time.
- Churn: f(product_quality, price).
- Revenue: users × conversion(price, brand) × ARPU(tier).
- Cash: + revenue + raised − burn(team_size). Bankruptcy if cash < 0.
- Valuation: ARR multiple adjusted for growth rate. Bankruptcy = 0/penalty.

### Artifact-hybrid scoring (core mechanic)
Each artifact gets a 0–1 LLM-rubric score that **multiplies** its sim lever:
- feature spec quality → eng efficiency / quality gain
- marketing copy quality → channel conversion multiplier
- pricing / positioning → conversion
- investor update quality → fundraise amount / terms
Noise control: fixed rubric, average several judge samples, keep artifacts short
(~a paragraph each). Valuation remains the deterministic anchor.

### Eval / leaderboard (deliverable)
- **Primary reward:** firm valuation at horizon (deterministic given decisions + artifact scores).
- **Reference baselines:** omniscient single agent (no coordination boundary) →
  **coordination tax**; heuristic policy with template artifacts → floor.
- **Sub-metrics:** coordination tax, artifact-quality scores, survival rate, capital efficiency.
- **Variance control:** multiple + paired seeds; report distributions.
- **Leaderboard axes:** frontier models (Claude / GPT / Gemini / open) × team size ×
  communication protocol (free-form chat vs structured handoffs).

### 24-hour schedule
| Hours | Build |
|-------|-------|
| 0–5 | Sim engine: ~6 state vars, monthly tick, valuation function, seeding. |
| 5–8 | Artifact rubric judges (LLM scorers, averaged). |
| 8–14 | Agent harness: 4 role agents, partial-obs views, shared channel + planning doc, monthly turn loop. Wrap as a **HUD environment** (task spec + evaluator = valuation). |
| 14–20 | Reference baselines (omniscient + heuristic), run frontier models, leaderboard + replay viewer. |
| 20–22 | Demo polish: failure-mode gallery ("watch the C-suite misalign and run out of runway"). |
| 22–24 | Slides + pitch. |

### Risks & mitigations
- **Artifact-hybrid cost/latency/noise** → short horizon (12 months), caching, averaged
  judges, short artifacts; fall back to hybrid-on-marketing-only (engine unchanged).
- **Optimal undefined** → use omniscient + heuristic reference baselines, report
  normalized score; headline = coordination tax + valuation, not regret.
- **Multi-agent orchestration fiddly** → round-robin monthly turns + one append-only
  shared log + a planning-doc template. No fancy protocol.
- **Coordination not actually forced** → pressure-test the observability split early.

### Why it scores well
Verifiable (valuation anchor) · realistic & novel (full-stack SaaS, agents writing real
artifacts) · best-in-class collaboration story (agent C-suite) · great live demo
(agents negotiating + coordination-tax chart) · RFT-ready stretch (one config from a
fine-tuning run; not needed for the demo).

### Name
**FirmBench** (working name). Alternatives: CoordTax, Boardroom.

---

## 5. Immediate next step
Scaffold the simulator engine (the 0–5h block): deterministic monthly tick over the 6
latent state stocks + valuation function + seeding, with stub hooks for artifact-quality
multipliers. Everything else hangs off it.
