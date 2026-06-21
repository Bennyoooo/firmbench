# Phase D — Multi-Agent FirmBench: Build Spec & Full Context

> **Purpose.** Self-contained spec for a **fresh session** (no prior conversation) to build
> the multi-agent layer (Phase D) of FirmBench. Clone the repo, branch, and execute. Read
> the existing code first (`sim.py`, `env.py`, `run.py`, `rft.py`, `rft_hud.py`, `scorer.py`,
> `tasks.py`) — this spec assumes Phase A (below) is already on `main`.
>
> **Golden rule (learned the hard way):** the codebase is edited in parallel — **re-read a
> file right before you edit it**; don't trust stale reads.

---

## 0. TL;DR of the ask

Split the single firm-agent into cooperating **role-agents** (Builder / Marketer / Pricer /
Coordinator) with **partial observability** and a **coordination tax**, and adapt the RL
training pipeline. Cover: **sim, env, tasks, eval/agent, and the RL fine-tuning pipeline.**

**"Do we train a checkpoint per agent?"** → **No.** Train **one shared checkpoint**; each
role is that same model conditioned on a role-specific system prompt + role-sliced
observation (parameter sharing). All role-turns in an episode share the **team reward**.
Details in §3.

---

## 1. Current state — Phase A (what already exists on `main`)

FirmBench is a single-agent, deterministic, domain-randomized **market-discovery POMDP**,
now upgraded to a **persona + subscription/LTV market**. Everything is gated behind `Config`
flags: `Config()` == exact v1; `Config.phase_a()` == full market.

### Files
| File | Role |
|---|---|
| `sim.py` | Deterministic sim core: `Config`, `User`, `Segment`, `World`, `generate_world`, `FirmEnv`, policies (`NaivePolicy`, `ScriptedExperimenter`, `OraclePolicy`), `run_episode`, `subworld`, `replay_profit`, `ablation_gate`, `best_price_for`, `_best_channel`. Pure stdlib. |
| `env.py` | HUD v6 env (MCP tools + `@env.template`), `_CFG = Config.phase_a()`, discovery-efficiency grading. |
| `run.py` | `Verifier` (disc.eff grade), `evaluate_policy`, `run_episode_detailed`, toy REINFORCE. |
| `agent.py` | LLM agent harness (Fireworks, structured-JSON actions). |
| `rft.py` | Rejection-sampling SFT (expert iteration) + offline `--selftest`. |
| `rft_hud.py` | HUD-native on-policy RL (GRPO via `hud.train`) + offline `--selftest`. |
| `scorer.py` | NL artifact translator: `score_ad_copy → {craft, target_pains}`, `score_feature_spec → {feature_id, quality}`; `fast_mode` returns 1.0 (no LLM). |
| `renderer.py` | Renders ad copy / specs to HTML. |
| `tasks.py` | HUD tasks (seeds 42/123/7) + `SYSTEM_PROMPT`. |

### Config (key fields)
v1: `n_pains=8, n_features=8, n_users=5000, horizon=10, starting_cash=6000, build_cost=300,
impressions_per_dollar=0.2, alpha=4, beta=2, gamma=3, wtp_mu=3.9, wtp_sigma=0.5,
price_grid=range(20,121,10)`.
Phase A flags (default **False**): `use_segments, use_channels, use_elasticity,
use_quality_bar, use_retention`; params `n_segments=5, n_channels=3, channel_fit_off=0.25,
elasticity_mu=2.0, elasticity_sigma=0.5, quality_bar_mu=0.3, quality_bar_sigma=0.1,
quality_gate_k=8.0, subscription=True, churn_base=0.10, churn_price_coef=0.30,
churn_quality_coef=0.30, readopt_rate=0.0`.
`Config.phase_a(scale_budget=True, **overrides)` → all flags on; `horizon=10+2*n_channels=16`,
`starting_cash=6000*n_channels=18000`.

### World / users
`Segment(pain_affinity, wtp_mu, wtp_sigma, elasticity_mu, channel_pref, quality_bar,
churn_base, weight)`. `User(pains, wtp, segment_id, elasticity, channel_pref, quality_bar)`.
When `use_segments`, users are resampled from K hidden personas (correlated pains + shared
economics + per-user noise). `World` carries `segments`, `pain_names/keywords`,
`feature_names/keywords`.

### Funnel (`FirmEnv._run_campaign`, expectation-based, deterministic)
Per reached user (`reached = pool[:impressions]`, `impressions = int(spend*ipd)`, pool =
users sharing ≥1 target pain, ordered by index):
- `p_try = craft * channel_fit * resonance`, `resonance = |target∩pains|/|pains|`,
  `channel_fit = 1.0 if u.channel_pref==channel else channel_fit_off` (1.0 if channels off).
- `p_buy = sigmoid(alpha*ff + beta_u*(wtp-price)/wtp - gamma) * quality_gate`,
  `ff = fulfilled fraction` (built features' quality / |pains|), `beta_u = u.elasticity`
  (else `beta`), `quality_gate = sigmoid(quality_gate_k*(ff - quality_bar))` (1.0 if off).
- **Retention:** `prospect = 1 - sub[idx] - churned[idx]`; `contrib = prospect*p_try*p_buy`;
  if `commit`: `sub[idx] += contrib`. (Only prospects convert → base bounded by population.)
- Diagnostics returned: `target, channel, audience, impressions, tries, purchases,
  bounced_quality, bounced_price, revenue, spend`.

### Subscriber lifecycle (LTV) — `FirmEnv.step`, `_apply_churn`
Per-user dicts `self.sub` / `self.churned` (None when retention off). Each round:
`recurring = sum(sub.values())*price` → campaigns acquire (commit) → `_apply_churn`
(per-segment `churn_eff = clamp(churn_base + price_pressure + quality_gap, .,.95)`, applied
per subscribed user; churned don't return unless `readopt_rate>0`). `profit = revenue −
spend − build_cost`. Bankruptcy if cash<0; done at `horizon`.

### Grading (NO holdout — important)
Reward = **discovery efficiency = profit / oracle**, clipped [0,1]. `env.py._grade_episode`
runs `OraclePolicy` for the seed and divides. `run.py.Verifier.grade` same (`flagged` kept
False for back-compat; `beat_oracle` flags disc.eff>1). The old secret-held-out + tripwire
scheme was **removed** (execution-based env → nothing to fake; holdout flagged honest
agents). Generalization = held-out eval **seeds** (rft trains seeds 1–16, evals 100+).

### Policies (references)
- `NaivePolicy` — random build/target/channel (floor).
- `ScriptedExperimenter` — round 0 probes pains × channels (learns best channel per pain
  from `tries`); builds + tests to discover `solves`; exploits solved pains on their best
  channel. (Disciplined discovery; **not** LTV-savvy → ~7% disc.eff under full.)
- `OraclePolicy` — omniscient, **LTV-aware**: builds top-popularity features, sizes spend to
  remaining prospects (`env.sub`/`env.churned`) — *acquire-then-coast* — caps price near
  median wtp to limit churn. This is the ceiling.

### Validation tools
- `ablation_gate(seeds)` — runs naive/scripted/oracle under v1, each single latent, and full;
  prints `naive < scripted < oracle` PASS / WARN(`sc>orc`) / FAIL. **Use this after every
  change.** Currently all PASS except `+channels`-alone (WARN ~4%, a soft-channel-mechanism
  artifact; under `full` the LTV oracle dominates).
- `replay_profit(world, user_indices, action_log, spend_scale)` — execution replay on a user
  subset through the real funnel (exact vs live at scale 1.0). `subworld(world, idx, cfg)`.
- Tests: `tests/test_phase_a.py` (run `python3 tests/test_phase_a.py` from repo root; the
  file inserts repo root on `sys.path`). `python3 sim.py` prints the learnability table.

### Current eval (10 held-out seeds, full model)
naive **0.021** · scripted **0.066** · oracle **1.000** (disc.eff). The LTV game
(acquire-then-coast, price for retention) is the deep skill; huge RL headroom.

---

## 2. Multi-agent design (Phase D)

Split the firm into 4 **role-agents** that share one firm but see only their slice and must
**communicate** to coordinate. Build it as an additive wrapper — **the single-agent env must
stay unchanged** (multi-agent is opt-in).

### Roles & action ownership
| Role | Owns | Sees (partial obs) |
|---|---|---|
| **Coordinator** | per-round **budget** for the Marketer + a directive message; commits the round | firm summary: round/horizon, cash, built features, last-round aggregate profit & churn; all role messages |
| **Builder** | `build` (feature id + optional `spec`/quality) | built features, `bounced_quality` signal, Coordinator directive |
| **Pricer** | `price` | conversion-vs-price signals (`bounced_price`, recent purchases/churn), Coordinator directive |
| **Marketer** | `campaigns` (target pains + `channel` + `spend` within budget + `ad_copy`) | per-(pain,channel) campaign diagnostics (audience/tries/purchases/bounce), Builder's "what's built" message, budget |

**Partial observability is the point:** the Marketer must learn *from the Builder's message*
which feature was built (so it targets the solved pain) — otherwise it promotes an unbuilt
feature and burns budget. That waste is the **coordination tax**.

### Communication: a shared blackboard
A per-round list of structured messages `{role, text}` (free-form text, capped length).
Each role reads the blackboard slice it's allowed to see and may append one message. This is
the channel the RL agents learn to use. Keep it simple (a list of short strings).

### Round protocol (turn-based, deterministic order)
1. **Coordinator** reads firm summary → sets `budget` + directive message.
2. **Builder** acts (build?) → message ("built feature X" — note: the Builder doesn't know
   which *pain* X solves unless discovered; it reports what it built/tested).
3. **Pricer** acts (price) → message.
4. **Marketer** acts (campaigns within `budget`, using messages) → message.
5. **Commit**: assemble the four role actions into ONE `FirmEnv` action dict
   `{build, price, campaigns:[{target,spend,channel,craft}]}` and call `FirmEnv.step`.
6. Distribute results back: each role gets its sliced observation for next round.

### Coordination tax (the headline metric)
`coordination_tax = oracle_profit − team_profit` (or `1 − team_disc_eff`). The oracle is the
**single-agent full-info** `OraclePolicy` (no partial-obs barrier). A perfectly coordinated
team approaches the single-agent oracle; a poorly coordinated one leaves a large gap. Report
it alongside disc.eff. A **single-agent baseline** (the existing `ScriptedExperimenter`) and a
**scripted team** that reproduces it across roles + messaging bound the tax from both sides.

### Matched-quad discipline (keep it learnable)
Every coordination mechanism must be a matched quad: a **latent/decision** (e.g. which
feature is built), an **action** (Builder builds; Marketer targets), an **observation/message**
(Builder's note on the blackboard), and a **reward consequence** (team profit). If a role
can't observe what it needs (directly or via a message), it can't coordinate — that's the
intended difficulty, but verify a scripted team CAN coordinate (see gates).

---

## 3. RL training for multi-agent (the checkpoint question, in depth)

### Recommended: ONE shared, role-conditioned checkpoint (parameter sharing)
- **One policy** (one fine-tuned model). Each role-agent = the same model with a
  **role-specific system prompt** + its **role-sliced observation** (incl. the blackboard).
- A rollout of one team-episode produces **role-turns**: e.g. (Coordinator turn, Builder
  turn, Pricer turn, Marketer turn) × rounds. Each turn is a `(prompt, completion)` pair,
  exactly like the single-agent harness records today (`agent.py` records `messages`).
- **Reward = the team's episode disc.eff** (profit/oracle). Every role-turn in the episode
  gets the **same** episode-level advantage (cooperative, shared reward).
- **Credit assignment:** simplest and robust is shared advantage across all role-turns; GRPO
  normalizes advantage **within a world's rollout group** (already how `rft_hud.py` groups).
  No per-role critic needed.

### How it adapts the existing pipelines (minimal change)
- **`rft.py` (rejection-sampling SFT):** roll out team-episodes; grade each by team disc.eff;
  keep the best non-degenerate episode per world; **flatten ALL its role-turns** into the SFT
  JSONL (each role-turn is one chat example, with the role system prompt). SFT one model on
  the pooled role-turns → it learns all four roles. The `MockModel`/`--selftest` pattern
  carries over: a mock "team" whose skill rises each iter, driving rollout→filter→dataset→eval.
- **`rft_hud.py` (on-policy GRPO):** each `hud.Run` = one team-episode through the gateway;
  its reward = team disc.eff; group runs by world; one grouped policy-gradient step over all
  role-turns of the grouped runs; `optim_step` promotes the shared checkpoint. Add an offline
  `--selftest` mock team like the single-agent one.

### Alternative (document, don't build): per-agent checkpoints (MAPPO)
Each role = its own model + a centralized critic (CTDE). More expressive, but **4× the
fine-tuning cost** and heavy infra — not worth it for cooperative LLM roles. Mention it as the
"if you really need role specialization" path; default to shared-policy.

---

## 4. Implementation plan (file by file)

Build incrementally, **test + `ablation_gate()` after each step**, keep single-agent
byte-identical. Suggested new files keep the multi-agent layer separable.

### 4.1 `multiagent.py` (new) — the env wrapper + scripted team
- `class Blackboard`: list of `{role, text}`; `post(role, text)`, `read(roles)`, `clear()`.
- `class MultiAgentFirmEnv`: wraps a `FirmEnv` (built from a `phase_a` world).
  - `reset()` → dict of per-role observations + empty blackboard.
  - `role_obs(role)` → the sliced observation for a role (see §2 table) from the underlying
    `FirmEnv` state + last `per_campaign` + blackboard.
  - `submit(role, action, message=None)` → stash the role's partial action + post message.
  - `commit()` → assemble `{build, price, campaigns:[{target,spend,channel,craft}]}` from the
    stashed role actions (clamp Marketer spend to Coordinator budget), call
    `self.env.step(...)`, return new per-role obs + round profit; clear stash + blackboard.
  - Track `total_profit` (mirror `FirmEnv`).
- `def coordination_tax(world)`: `oracle = run_episode(world, OraclePolicy(world))`; run the
  scripted team; return `oracle - team_profit` and `team_profit/oracle`.
- `class ScriptedTeam`: 4 scripted role-policies that **reproduce `ScriptedExperimenter`**
  split across roles, communicating via the blackboard (Builder posts what it built so the
  Marketer targets correctly). This proves the protocol is coordinatable and is the team
  baseline. Also a `NaiveTeam` (roles act without reading messages) to show the tax.

### 4.2 `env_multiagent.py` (new) OR extend `env.py` — HUD serving
Two viable patterns (pick one; document the choice):
- **(A) Coordinator-dispatches:** the single HUD agent *is* the Coordinator and gets tools
  `delegate_build(spec)`, `delegate_price(...)`, `delegate_campaigns(...)` that invoke the
  other roles (LLM sub-calls inside the env, or just structured sub-actions). Simplest for HUD
  (still one HUD "agent").
- **(B) Native multi-agent:** if the HUD/runtime supports multiple agents per episode, serve
  role-specific tool sets. Heavier; check `hud` capabilities first. **Recommendation: ship
  (A) first**; (B) is a stretch. Keep grading = team disc.eff (reuse `_grade_episode`).

### 4.3 `tasks.py` — add a multi-agent task
A `multiagent_market_discovery` template + role system prompts (one per role, describing its
slice, tools, and that it must read/post blackboard messages). Reuse seeds.

### 4.4 `run.py` — multi-agent eval
Add a head-to-head: single-agent scripted vs `ScriptedTeam` vs `NaiveTeam` vs oracle, on
held-out seeds, reporting **disc.eff + coordination tax**. Reuse `Verifier`/`evaluate_policy`
shape.

### 4.5 `rft.py` / `rft_hud.py` — shared-policy role-conditioned training
Per §3. Add a `play_team_episode(world, team_factory)` that returns the flattened role-turn
records + the team action_log + team profit; grade with disc.eff; reuse the
rollout→filter→dataset→eval (`rft.py`) and the GRPO group step (`rft_hud.py`). Add `--selftest`
mock teams.

### 4.6 `tests/test_multiagent.py` (new)
- single-agent env unchanged (import + a v1 `ablation_gate` row still PASS).
- `MultiAgentFirmEnv.commit()` produces the same `FirmEnv.step` result as the equivalent
  single-agent action (round-trip equivalence).
- `ScriptedTeam` (with messaging) beats `NaiveTeam` (no messaging) → coordination tax > 0 and
  the scripted team's tax < naive team's.
- blackboard slicing: a role can't see another role's private obs.
- no information leak (segment ids etc.) to any role.

---

## 5. Build discipline (mirror Phase A)

1. **Single-agent stays byte-identical.** Multi-agent is additive; `python3 sim.py` and
   `ablation_gate()` must be unchanged. Run them after every step.
2. **Coordination gate** (the Phase-D analog of the learnability gate): `NaiveTeam < ScriptedTeam < oracle`,
   and `ScriptedTeam`'s coordination tax should be modest (it coordinates) while `NaiveTeam`'s
   is large (it doesn't). If the scripted team can't beat naive, a role lacks the
   observation/message it needs — fix the slice, don't hand-wave.
3. **Deterministic.** No `Date.now`/`random` outside seeded RNG in `sim.py`; keep the funnel
   expectation-based.
4. **Offline RL selftest** (mock team) must bend the curve before any real run, like
   `rft.py --selftest`.
5. Commit per component with tests green; keep the multi-agent code in its own modules.

---

## 6. Open decisions / recommendations

- **Comms model:** free-form text blackboard (recommended, what RL learns to use) vs
  structured fields. Start free-form, cap length.
- **Coordination tax:** implicit (budget contention + partial obs, recommended) vs explicit
  per-message cost (a later dial).
- **HUD serving:** pattern (A) coordinator-dispatch first; (B) native multi-agent is a stretch
  — verify `hud` supports it before committing.
- **Role count:** 4 is a good start; could collapse Pricer into Coordinator if the gate shows
  Pricer adds little signal.
- **Reward shaping:** start with pure team disc.eff (shared). Only add per-role shaping if the
  coordination tax won't shrink — and measure it, don't assume.

---

## 7. Gotchas (paid for in Phase A)

- **Re-read files before editing** — parallel edits make stale reads dangerous.
- **disc.eff clips at 1.0**; if any policy beats oracle, the `beat_oracle` flag / gate WARN
  fires — that means the oracle (reference) needs strengthening, not that the agent "won."
- **Flags-off reproducibility:** any new RNG draw must be gated so `Config()` worlds stay
  byte-identical; new tool params must default to v1-neutral.
- **Channels are a soft conversion multiplier** (`channel_fit_off=0.25`), not audience-gating
  — re-targeting doesn't expand reach. The `+channels`-alone gate WARN is expected.
- **Subscriber lifecycle bounds the base by population** (prospect-gating). Don't reintroduce
  an aggregate base that can exceed the population (that was a reward-hacking hole).
- **Grading has no user holdout** — don't reintroduce it; generalization is seed-level.
- **`tests/test_phase_a.py` inserts repo root on `sys.path`** — keep that pattern for new
  test files (run from repo root).

---

## 8. First session checklist

1. `git clone <repo>` fresh; `git checkout -b phase-d-multiagent`.
2. `python3 sim.py` and `python3 -c "from sim import ablation_gate; ablation_gate()"` — record
   the baseline (must match §1).
3. Build `multiagent.py` (env wrapper + Blackboard + ScriptedTeam + NaiveTeam +
   coordination_tax) + `tests/test_multiagent.py`; get the coordination gate green.
4. Add the multi-agent eval to `run.py`; report disc.eff + coordination tax.
5. Add the HUD serving (pattern A) + `tasks.py` multi-agent task + role prompts.
6. Add shared-policy role-conditioned training to `rft.py` (+ `--selftest`); then `rft_hud.py`.
7. Update `PLAN.md` (mark Phase D done) + `README.md`.
