# FirmBench — Market-Discovery Firm (single-agent, chosen design)

> A verifiable RL environment for **experimental market discovery**. A single agent
> runs a firm: it must reverse-engineer a hidden demand structure in a numeric user
> population by running experiments, then build the right product, market it to the
> right people, and price it right. **Reward = profit.** Primary diagnostic:
> *discovery efficiency* = regret vs a computable oracle.
> Multi-agent + *coordination tax* = future stretch.

The key move: the LLM judge is a **translator, not a grader** — it turns open-ended
artifacts (ad copy, feature spec) into structured parameters that drive a deterministic
numeric market. Reward emerges from market outcomes, not from the judge. We build the
**deterministic numeric core first** (structured actions), validate learnability, then
layer the natural-language artifact + translator on top.

---

## 0. What it measures

1. **Experimental market discovery** — can the agent run disciplined experiments to infer
   the hidden pain↔feature mapping, the biggest needs in the demography, and price
   elasticity, then exploit them?
2. **Resource allocation under uncertainty** — explore vs exploit; budget between
   building and marketing; don't go bankrupt.
3. **Artifact craft** (once the NL layer is added) — concretely good ads/specs enter the
   sim as parameters, not as a score.

Primary reward: **profit** over the episode. Primary diagnostic: **discovery efficiency**
(regret vs oracle). *Coordination tax* returns when multi-agent is added (stretch).

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
  - interests/keywords derived from `pains` (used later for NL targeting; structured
    phase targets pains directly).
- Seeded → deterministic + resettable. **Domain-randomized per episode** (see §10).

---

## 2. Firm state (visible, evolves over the episode)

`cash`, `built_features` (each with `implementation_quality ∈ [0,1]`), current `price`,
`round`, and the agent's own history of campaign results.

---

## 3. The agent (single) — action & observation schema

One agent ("the operator") issues a **structured action per round**:
```
action = {
  build:     feature_id | None,          # build one feature this round (costs build_cost)
  price:     float,                      # set product price
  campaigns: [ { target_pains: set[int], spend: float } ]   # >=0 ad campaigns
}
```
(Structured phase first. NL phase later: `build` carries a spec string, `campaigns`
carry copy strings; a translator maps them to feature_id/quality and target_pains/craft.)

**Observation each round** (must be diagnostic — see §6):
```
obs = {
  round, cash, price, built_features,
  per_campaign: [ { target_pains,
                    impressions, tries, purchases, revenue } ],   # the discovery signal
}
```

---

## 4. Episode loop (default horizon = 10 rounds)

Per round: agent picks action → market sim runs the funnel → agent receives diagnostic
obs + reward (round profit) → cash updates; bankruptcy if cash < 0. Cumulative profit at
horizon = total return.

---

## 5. Funnel + formulas (deterministic / expectation-based → low variance)

For each campaign (`target_pains`, `spend`), with current `price π`:
1. **Reach:** `impressions = spend × impressions_per_dollar`. Served to the pool of users
   whose `pains ∩ target_pains ≠ ∅` (targeting concentrates spend; broad targeting dilutes).
   Reached count = `min(|matching pool|, impressions)`.
2. **Try:** per reached user, `p_try(u) = craft × resonance`, where
   `resonance = |target_pains ∩ u.pains| / |u.pains|` and `craft = 1.0` in the structured
   phase (set by the copy translator in the NL phase).
3. **Purchase ("API call to the product"):**
   `fulfilled = Σ_{p ∈ u.pains} [solves(p) ∈ built_features] × implementation_quality`
   `fulfilled_fraction = fulfilled / |u.pains|`
   `p_buy(u) = sigmoid(α·fulfilled_fraction + β·(u.wtp − π)/u.wtp)`
4. **Outcome (expected, deterministic):**
   `purchases = Σ_reached p_try(u)·p_buy(u)`; `revenue = purchases × π`;
   round cost = `spend + (build_cost if built)`.

Hidden and must be learned via experiments: the pain popularity distribution, the
`solves` mapping, and price elasticity (wtp).

---

## 6. Observations / feedback — THE make-or-break decision

The agent can only discover structure if feedback is **diagnostic**, not a lone revenue
number. Each campaign returns **per-target breakdowns** (impressions, try-rate,
purchase-rate, revenue) so an experiment is informative (e.g. "ads for pain X get tries
but no purchases → the feature solving X isn't built yet"). Build & validate this first.

---

## 7. Reward + metrics (all computable — world is numeric)

- **Reward (primary): profit** per round (dense) + cumulative at horizon. Bankruptcy penalty.
- **Discovery efficiency (primary diagnostic):** `oracle_profit − agent_profit` (regret).
  **Oracle** knows the world; brute-forces best feature subset (under build budget) ×
  best price × perfect targeting × optimal spend → max achievable profit. Tractable
  (2^|F| × price-grid × cheap funnel). Regret guarantee holds because the world is numeric.
- **Coordination tax:** N/A in single-agent (returns with the multi-agent stretch).
- Variance control: paired seeds; report distributions. Leaderboard: frontier models ×
  held-out seeds.

---

## 8. Judges / translators (NL phase; deferred after numeric core)

- **Ad copy** → `craft ∈ [0,1]` (cheap LLM, averaged) + `target_pains` (deterministic
  keyword/embedding match → near-zero noise/cost).
- **Feature spec** → feature id(s) + `implementation_quality`.
- Structured phase uses no LLM at all → cheap, deterministic, fast to iterate.

---

## 9. Cost

User funnel = pure Python/numpy, free. LLM spend = a few artifact translations/round on a
cheap model + the single agent's calls. Leaderboard (inference on held-out seeds) is
cheap. Keep agent context compact (structured state, not full transcript).

---

## 10. What is being learned? (domain randomization — REQUIRED)

**Trap:** fixed world → the agent just memorizes one answer key (overfitting).
**Fix:** **randomize the hidden world every episode** (new pain distribution, new
`solves` mapping, new wtp). The agent must then learn the **meta-skill of discovery**:
how to figure out *any* unknown market. Eval on **held-out randomized worlds** → measures
generalization, not memorization.

Behaviors reinforced: efficient experimentation / active learning; explore→exploit
budgeting; belief updating from diagnostic feedback; (NL) artifact craft that transfers.
Net capability: **running a data-driven business under uncertainty.**

Honest caveats: transfer to real business is unproven (claim "discovery under
uncertainty, measured by generalization"); guard against reward-hacking the craft judge
(deterministic targeting; craft is only one input).

---

## 11. RL formalization

A **single-agent partially-observable MDP (POMDP)** with outcome reward.
- **State:** hidden world (demography, `solves`) + firm state (built, price, cash, history).
- **Action:** structured `{build, price, campaigns}` (text in the NL phase) — tool calls.
- **Observation:** firm state + per-campaign diagnostic breakdowns (partial: world latent).
- **Transition:** deterministic funnel given seed (NL craft-translator is the one
  stochastic piece — cache it).
- **Reward:** round profit (dense) + cumulative.
- **Episode:** ~10 rounds; resettable via seed.
RL-native because success requires **explore→exploit of hidden latent structure** — a
structured POMDP, unlike one-shot task benchmarks.

---

## 12. Training (stretch) vs evaluation (deliverable)

- **Eval / leaderboard (deliverable, NO training):** run each frontier model on N held-out
  seeds (~20–50); report profit + discovery-efficiency regret. Pure inference.
- **Training (stretch):** re-seed **+ policy update**, interleaved (GRPO-style):
  ```
  repeat: sample world → K rollouts of current policy → reward=profit → update toward best
  ```
  ~1000 episodes shows a learning *signal* (not SOTA); small open model (Fireworks RFT);
  short horizon.

---

## 13. Build order (de-risked) + 24h plan

**Phase 1 — deterministic numeric core (do first):** world gen, demography sampler, funnel,
`env.reset/step` (gym-like), oracle, plus a **scripted experimenter** + naive baseline to
**prove learnability** (smart > naive, smart → oracle). No LLM.
**Phase 2 — agent harness:** single LLM agent, tool-calling loop over structured actions →
run frontier models → leaderboard (profit + regret).
**Phase 3 — NL artifact layer:** copy/spec translators (craft + deterministic targeting) →
adds realism + craft dimension.
**Phase 4 — stretch:** multi-agent + coordination tax; RFT training curve.

| Hours | Build |
|-------|-------|
| 0–5 | Phase 1: numeric world + funnel + env + oracle + scripted/naive baselines; validate learnability. |
| 5–11 | Phase 2: single-agent harness (tool-calling), run frontier models, leaderboard. |
| 11–17 | Phase 3: NL translators (craft LLM + deterministic targeting; spec→feature). |
| 17–22 | Polish: leaderboard + replay viewer; failure-mode gallery; (optional) tiny RFT curve. |
| 22–24 | Slides + pitch. |

---

## 14. Risks & mitigations

1. **Identifiability / observation design** (#1) → per-target diagnostic feedback;
   validate with the scripted experimenter before any LLM.
2. **Balancing for discoverability** → tune pools/skew/costs/horizon so structure is
   inferable within the round budget; not trivial, not impossible.
3. **Oracle tractability** → small pools (|F|=8) make brute force trivial.
4. **Reward-hacking the craft judge** (NL phase) → deterministic targeting; craft one input.

---

## 15. Stretch: multi-agent + coordination tax

Split the single agent into Builder / Marketer / Pricer / Coordinator with partial
observability and a shared log. Adds the **coordination tax** metric
(`omniscient single planner − multi-agent firm`). Engine and world unchanged.

---

## 16. One-line pitch
> FirmBench drops an agent into a market whose demand structure is hidden in a numeric
> user population. To make money it must *experiment* to discover what people need,
> *build* it, *market* it to the right people, and *price* it right — and we measure
> profit and how close it gets to the oracle (discovery efficiency).
