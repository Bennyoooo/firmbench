# FirmBench — Design & Progress

> A verifiable RL environment for **experimental market discovery**. A single agent runs
> a firm: it must reverse-engineer a hidden, persona-structured demand population by
> running experiments, then build the right features, target the right segments on the
> right channel, and price for retention. **Reward = discovery efficiency** (profit ÷
> oracle, clipped to [0,1]). Deployed on **HUD**.
>
> **Now Phase A** — a hybrid persona + subscription/LTV market (see the **Phase A**
> section below). `Config()` is exactly the v1 baseline; `Config.phase_a()` turns the full
> market on. §§1–9 describe the v1 baseline.
>
> **Hackathon:** HUD Frontier/RSI RL Environments Hackathon (HUD W25 × YC).

---

## Progress

### ✅ Completed

- [x] **Design** — single-agent market-discovery POMDP, domain-randomized per
  episode, deterministic numeric funnel, computable oracle.
- [x] **Phase 1 — simulation core** (`sim.py`) — `generate_world`, gym-like `FirmEnv`
  (reset/step), `OraclePolicy`, `NaivePolicy`, `ScriptedExperimenter`.
  **Learnability validated:** naive loses money (−2.1k), scripted earns 75k
  (~54% of 138k oracle). Discovery is the binding skill.
- [x] **Phase 2 — LLM agent harness** (`agent.py`) — single agent driving `FirmEnv`
  via structured-JSON actions over Fireworks OpenAI-compatible API. Falls back to
  scripted baseline offline.
- [x] **Verifier** (`run.py`) — secret held-out users (20%), full-episode replay
  grading, cheat tripwire (flags policies whose visible profit diverges from
  held-out profit). Ported from `rl-experiments` (GAIA2/SIMS-inspired) +
  `autonomous-businesses-template` (hidden-ticket pattern).
- [x] **Head-to-head evaluation** (`run.py`) — all policies graded on same held-out
  worlds, single comparison table. Validated:
  - naive: −700 reward, 15/21 flagged ❌
  - scripted: 21,138 reward, 0/21 flagged ✅
  - RL (REINFORCE): 16,670 reward, 2/21 flagged ✅
  - oracle: 24,750 reward, 6/21 flagged (expected)
- [x] **REINFORCE training loop** (`run.py`) — 2-param probe-vs-exploit policy,
  learns from verifier reward via policy gradient. Proves env is trainable (same
  shape as full RFT). Trains in seconds on CPU.
- [x] **HUD v6 integration** (`hud_env.py`, `hud_tasks.py`) — 6 MCP tools
  (`probe_market`, `build_feature`, `set_price`, `run_campaign`, `get_state`,
  `end_round`), verifier grading on episode end, `@env.template` + FastMCP.
  Ported from `autonomous-businesses-template`.
- [x] **Deployed to HUD platform** — `hud deploy .` succeeded.
  - Environment: `firmbench` v1
  - Dashboard: https://hud.ai/environments/169eeeae-d713-4cc8-b700-74c5c4a8719a
  - Image: `156041433621.dkr.ecr.us-west-2.amazonaws.com/hud/envs/firmbench:v1`
- [x] **Phase A — persona + LTV market** (`sim.py`/`env.py`/`run.py`/`tasks.py`) — hidden
  segments, channels, per-user elasticity, quality bar, and a subscriber-lifecycle
  subscription/churn model, all behind `Config` flags (v1 byte-identical when off).
  **Grading redesigned** to discovery-efficiency (profit ÷ oracle); user holdout dropped.
  `ablation_gate()` validates `naive < scripted < oracle` per latent. 16 tests; see the
  Phase A section. Reward-hacking review: flood edge closed by the lifecycle model.

### 🔲 Next

- [x] **RFT harness** (`rft.py`) — rejection-sampling fine-tuning (expert iteration /
  STaR): rollout → grade with verifier → keep best non-flagged positive trajectory
  per world → export chat JSONL → Fireworks SFT (firectl) → re-eval. Offline
  `--selftest` (mock model, no network) validates the whole machinery and shows the
  curve bend: **−734 → 15,524** mean held-out reward, flags 5/8 → 0/8. The verifier
  *curates the training set* — only profitable, non-cheating episodes become SFT data.
- [x] **Real RL run — GRPO, the curve bends** (`grpo/`, `grpo/RESULTS.md`) — genuine
  reinforcement learning (GRPO, not SFT) on Fireworks managed RFT (eval-protocol).
  Reward = our oracle-normalized profit verifier (execution-based, ungameable),
  8 candidates/prompt, KL-regularized. **qwen3-8b** learning curve over 3 epochs:
  **0.193 → 0.307 → 0.367**; held-out generalization **base 0.147 → tuned 0.529
  (+260% on unseen worlds)**. Journey: glm-5p1 trainer too heavy (crashed) → pivoted to
  qwen3-8b; fixed a zero-variance early-stop by unclipping the reward + temp 1.0 +
  8 candidates. Job `kby4ofja` → `firmbench-qwen3-8b-grpo-v4`.
- [x] **HUD eval of the fine-tuned checkpoint** (`HUD_EVAL.md`) — `hud eval` verified
  driving the Fireworks GRPO checkpoint against `env.py` end-to-end (openai_compatible +
  scale-to-zero LoRA deployment; no Docker).
- [~] **Earlier: rejection-sampling SFT** (`rft.py --run`, `RFT_RUN.md`) — on Fireworks with
  `glm-5p1` (only serverless model that trains+serves LoRA cheaply). Base eval
  **−409.6** mean reward (loses money). 24 real rollouts → **cold start**: zero
  winning trajectories (frontier models score ~0 here), so we bootstrap with 240
  expert (`ScriptedExperimenter`) turns. Dataset uploaded to Fireworks (READY);
  firectl installed + authed; SFT command validated. **Blocked on $50 training
  credits** (new account is Tier 1; glm-5p1 needs B200/B300 quota). Adding credits
  → one command finishes train→serve→eval. See `RFT_RUN.md`.
- [ ] Run `hud eval` with Claude / frontier models on held-out seeds
- [ ] Multi-model leaderboard (Claude / GPT / Gemini / Fireworks open)
- [ ] Phase 3 — NL artifact layer (ad copy + spec → craft translator)
- [~] Multi-agent (Phase D — Builder/Marketer/Pricer/Coordinator + coordination tax) —
  building in a separate branch / fresh clone
- [ ] Polish: replay viewer, failure-mode gallery, pitch deck

---

## Phase A — Persona + LTV market (CURRENT)

Upgraded from the v1 single-attribute population to a **hybrid persona market with
subscription/LTV dynamics**, all behind `Config` flags (`Config()` = exactly v1;
`Config.phase_a()` = full market). §§1–9 below describe the v1 baseline; this is what's live.

**World — new hidden latents (each behind a flag):**
- **Segments** (`use_segments`): ~5,000 users from K hidden personas — correlated pain
  cluster, wtp, elasticity, preferred channel, quality bar, churn rate — plus per-user noise.
- **Channels** (`use_channels`): segments differ in which channel reaches them.
- **Per-user elasticity** (`use_elasticity`): price sensitivity varies by user.
- **Quality bar** (`use_quality_bar`): soft gate — under-quality features don't convert.
- **Subscription / churn** (`use_retention`): per-user lifecycle **prospect → subscriber →
  churned**. Only prospects convert (no re-selling to current subscribers); churned users
  don't return unless `readopt_rate>0`; subscribers pay **recurring** revenue and **churn**
  when price is too high or quality too low (responsive + segment-varied). Optimize LTV.

**Firm state:** + per-user subscriber base (latent); `implementation_quality` now varies
(from NL specs scored by the translator in `scorer.py`).

**Tools:** `probe_market` / `run_campaign` take `channel` + optional `ad_copy`; `build_feature`
takes optional `spec`. Diagnostics add **bounce reasons** (`bounced_quality` vs `bounced_price`)
so quality/price failures are separately observable. Horizon scales (~16) with channels.

**Funnel:** `p_try = craft × channel_fit × resonance`; `p_buy = sigmoid(α·ff + β_u·price_term −
γ) × quality_gate`; acquisition gated by per-user **prospect mass**; recurring + churn each round.

**Grading (changed):** reward = **discovery efficiency = profit ÷ oracle**, clipped to [0,1].
The secret-held-out + tripwire scheme was **dropped** — in an execution-based env the agent
can't fake profit (nothing to verify) and the holdout flagged honest agents under Phase A.
Generalization is measured by **held-out eval seeds** (domain randomization). `env.py` and
`run.py` both grade on disc.eff; `beat_oracle` flags any policy exceeding the reference.

**Design discipline:** every latent is a **matched quad** (latent ↔ action ↔ observation ↔
reward), validated by `ablation_gate()` — `naive < scripted < oracle` holds per latent and for
the full stack. (`+channels`-alone WARNs ~4%: a channel-aware scripted edges the greedy
non-LTV oracle — a mechanism artifact; under `full` the LTV-aware oracle dominates.)

**Eval (10 held-out seeds, full model):** naive **0.021** · scripted **0.066** · oracle
**1.000**. The LTV game (acquire-then-coast, price for retention, don't burn cash on a
saturated finite market) is deep skill — scripted reaches only ~7% of the ceiling, leaving
wide RL headroom.

**Reward-hacking review:** execution-based grading rules out score-faking. The one residual
edge — flooding overlapping campaigns to re-convert a saturated pool — is **closed** by the
subscriber lifecycle (prospect-gating bounds the base by the real population; verified).

---

## 0. What it measures

1. **Experimental market discovery** — can the agent run disciplined experiments to
   infer the hidden pain↔feature mapping, the biggest needs in the demography, and
   price elasticity, then exploit them?
2. **Resource allocation under uncertainty** — explore vs exploit; budget between
   building and marketing; don't go bankrupt.
3. **Artifact craft** (once the NL layer is added) — concretely good ads/specs enter
   the sim as parameters, not as a score.

Primary reward: **profit** over the episode. Primary diagnostic: **discovery
efficiency** (regret vs oracle). *Coordination tax* returns when multi-agent is added.

---

## 1. The hidden world (pre-sampled per episode, invisible to the agent)

- **Pain-point pool** `P` (default 8) and **feature pool** `F` (default 8).
- **Hidden mapping** `solves: P → F` (which feature addresses which pain). Default: a
  bijection. This is the "supply–demand formula" the agent must reverse-engineer.
- **Demography** of N **numeric** users (default 5,000; NOT LLM-backed → free to run).
  Each user `u`:
  - `pains ⊂ P` (1–3), sampled from a **skewed popularity distribution** (some pains
    common = "biggest needs", some rare).
  - `wtp` (willingness to pay, $) ~ lognormal.
- Seeded → deterministic + resettable. **Domain-randomized per episode** (§8).

---

## 2. Firm state (visible, evolves over the episode)

`cash` (starts $6,000), `built_features` (each with `implementation_quality ∈ [0,1]`),
current `price`, `round`, and the agent's own history of campaign results.

---

## 3. The agent — tools (MCP, via HUD)

The agent interacts via **6 MCP tool calls** per round:

| Tool | What it does |
|------|------|
| `probe_market(target_pains, spend)` | Cheap discovery campaign → returns audience, impressions, tries, purchases, revenue |
| `build_feature(feature_id)` | Build a feature ($300 each). Hidden which pain it solves. |
| `set_price(price)` | Set the product price |
| `run_campaign(target_pains, spend)` | Full marketing push (same as probe, higher spend) |
| `get_state()` | Current round, cash, price, built features |
| `end_round()` | Commit actions, advance to next round |

The agent calls tools within a round (probe, build, set price), then calls
`end_round()` to commit. Actions are logged for verifier replay.

---

## 4. Episode loop (default horizon = 10 rounds)

Per round: agent calls tools → `end_round()` commits → market sim runs the funnel →
agent receives diagnostic feedback + reward → cash updates; bankruptcy if cash < 0.

---

## 5. Funnel + formulas (deterministic)

For each campaign (`target_pains`, `spend`), with current `price π`:
1. **Reach:** `impressions = spend × 0.1`. Served to users whose `pains ∩ target ≠ ∅`.
2. **Try:** `p_try(u) = |target ∩ u.pains| / |u.pains|` (resonance).
3. **Purchase:** `fulfilled_frac = Σ [solves(p) built] × quality / |u.pains|`;
   `p_buy = sigmoid(4·fulfilled_frac + 2·(wtp−π)/wtp − 3)`.
4. **Revenue** = `Σ p_try·p_buy × π`; round cost = spend + build cost.

Hidden and must be learned: pain popularity, `solves` mapping, price elasticity.

---

## 6. Observations / feedback (diagnostic — the make-or-break)

Each campaign returns per-target breakdowns: **audience** (how many users match),
impressions, tries, purchases, revenue. This makes experiments informative — e.g.
"ads for pain X get tries but no purchases → the feature solving X isn't built yet."

---

## 7. Verifier (secret held-out users + cheat tripwire) — *v1 baseline, SUPERSEDED*

> **Superseded in Phase A** by discovery-efficiency grading (reward = profit ÷ oracle; no
> user holdout — see the Phase A section). Kept below for history.

- **20% of users are held out** — the agent never gets campaign feedback from them.
- On episode end, the verifier **replays every action** on held-out users and computes
  true profit. This is execution-based grading — can't be gamed.
- **Cheat tripwire:** flags policies whose visible-user profit diverges from held-out
  profit by >15%. Flagged episodes get a reward penalty.
- Reward normalized to [0, 1] for HUD (raw holdout profit / 30,000).

---

## 8. Domain randomization (REQUIRED for training)

Fixed world → agent memorizes one answer key (overfitting). **Randomize per episode**
(new pain distribution, new `solves` mapping, new wtp). The agent must learn the
**meta-skill of discovery**, not world-specific facts.

Eval uses **held-out randomized worlds** → measures generalization, not memorization.

---

## 9. RL formalization

A **single-agent POMDP** with outcome reward.
- **State:** hidden world (demography, `solves`) + firm state.
- **Action:** MCP tool calls (build, price, campaigns, end_round).
- **Observation:** firm state + per-campaign diagnostics (partial: world latent).
- **Transition:** deterministic funnel given seed.
- **Reward:** round profit (dense) + cumulative.
- **Episode:** 10 rounds; resettable via seed.

RL-native: success requires **explore→exploit of hidden latent structure**.

---

## 10. Training vs evaluation

- **Eval / leaderboard (current deliverable):** run frontier models on held-out seeds;
  report profit + discovery efficiency. Pure inference, no training.
- **Training (real, `rft.py`):** rejection-sampling fine-tuning (expert iteration /
  STaR): sample world → N rollouts → grade with the held-out verifier → keep the best
  non-flagged positive trajectory → SFT the model on those turns → re-eval. The verifier
  is the reward *and* the data curator. Runs on Fireworks via `firectl`. Offline
  `--selftest` (mock model) proves the loop end-to-end and bends the curve.
- **Proof-of-concept done:** 2-param REINFORCE (probe-vs-exploit) in `run.py` trains in
  seconds — the toy version that proved the env is trainable before the LLM RFT.

---

## 11. Codebase

| File | What | Needs HUD? |
|------|------|-----------|
| `sim.py` | Deterministic sim core: `generate_world`, `FirmEnv`, oracle/naive/scripted | No |
| `agent.py` | LLM agent harness (Fireworks, structured-JSON) | No |
| `run.py` | Verifier + head-to-head eval + toy REINFORCE | No |
| `rft.py` | Real RFT: rejection-sampling fine-tuning on Fireworks (+ offline selftest) | Run: yes / selftest: no |
| `hud_env.py` | HUD v6 env: MCP tools + verifier grading | Yes |
| `hud_tasks.py` | HUD task definitions (3 seeds) | Yes |
| `Dockerfile.hud` | Deployable container | Yes |
| `pyproject.toml` | Dependencies | — |
| `requirements.txt` | Minimal deps for standalone (openai only) | — |

---

## 12. How we got here (design evolution)

1. Started from **TheAgentCompany** research — understood the task-completion eval
   pattern (175 tasks, deterministic + LLM-judge grading, encrypted evaluators).
   Key insight: it tests *employee task completion*, not business operation.
2. Explored **autonomous business** directions for the RSI hackathon. Considered:
   retail/vending sim, SaaS startup, autonomous ML research, multi-agent collaboration.
3. Studied **Vending-Bench** (Andon Labs) — learned the "luck vs skill" problem.
   Key fix: reward = regret-vs-optimal, paired seeds.
4. Evolved from scalar artifact judges → **judge-as-translator** → **simulated user
   population** (the current design). The judge doesn't grade the artifact; it converts
   it to parameters that drive a deterministic numeric market.
5. Simplified from 4-agent C-suite → **single agent** (multi-agent = stretch).
6. Merged patterns from three codebases:
   - **FirmBench** sim.py → the market-discovery environment
   - **rl-experiments/ml-research-rl** → verifier (secret held-out, tripwires, REINFORCE)
   - **autonomous-businesses-template** → HUD v6 integration (MCP, @env.template)
7. **Deployed to HUD platform** as `firmbench` v1.

---

## 13. Risks & mitigations

1. **Identifiability / observation design** (#1) → per-target diagnostic feedback;
   validated with scripted experimenter before any LLM.
2. **Balancing for discoverability** → tuned: impressions_per_dollar=0.1 (reach is
   expensive, wrong targeting loses money). Validated: naive loses money, experimenter
   earns 54% of oracle.
3. **Oracle tractability** → small pools (|F|=8) make brute force trivial.
4. **Verifier gaming** → secret held-out users + replay grading + cheat tripwire.
5. **Reward-hacking the craft judge** (NL phase, future) → deterministic targeting;
   craft is only one input among many.

---

## 14. Stretch goals

- **Multi-agent + coordination tax:** split into Builder / Marketer / Pricer /
  Coordinator with partial observability. Engine unchanged.
- **NL artifact layer:** ad copy + feature specs → LLM translator → craft multiplier.
- **Real RFT run:** Fireworks, small open model, compounding curve demo.
- **Richer sim:** more products, competitor agents, seasonal demand, supply chains.

---

## 15. Sponsor integration

| Sponsor | Usage | Status |
|---------|-------|--------|
| **HUD** (host) | Environment + verifier + leaderboard | ✅ Deployed |
| **Fireworks** | Agent inference (cheap open models) + RFT stretch | ✅ Wired |
| **Anthropic** | Claude as frontier leaderboard entry + env LLM judge (NL phase) | 🔲 |
| **Modal / Daytona** | Parallel rollouts for leaderboard + GPU for RFT | 🔲 |
| **DeepMind / MiniMax** | Gemini / MiniMax as leaderboard entries | 🔲 |
| **Exa / Protege** | Ground the sim in real market data (stretch) | 🔲 |

---

## 16. One-line pitch
> FirmBench drops an agent into a market whose demand structure is hidden in a
> 10,000-user simulation. To make money it must *experiment* to discover what people
> need, *build* it, *market* it to the right people, and *price* it right — and we
> measure profit and how close it gets to the oracle.
