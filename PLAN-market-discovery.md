# FirmBench — Market-Discovery Firm (chosen design)

> A verifiable RL environment for **agentic collaboration + experimental market
> discovery**. A small firm (4 agents) must reverse-engineer a hidden demand
> structure in a numeric user population, build the right product, market it well,
> and price it right. **Reward = profit.** Two computable diagnostic metrics:
> *discovery efficiency* and *coordination tax*.

This supersedes the earlier scalar artifact-hybrid sketch. The key move: **the LLM
judge is a *translator*, not a grader** — it converts open-ended artifacts (ad copy,
feature spec) into structured parameters that drive a deterministic numeric market.
Reward emerges from market outcomes, not from the judge.

---

## 0. What it measures

1. **Experimental market discovery** — can the firm run disciplined experiments to
   infer the hidden pain↔feature mapping, the biggest needs in the demography, and
   price elasticity, then exploit them?
2. **Cross-functional coordination** — do the ads target pains the firm actually built
   features for, at a price the demography will pay?
3. **Artifact craft** — are the ads/specs concretely good (enters the sim as a
   parameter, not a score).

Primary reward: **profit** over the episode. Diagnostics: discovery efficiency
(regret vs oracle) and coordination tax.

---

## 1. The hidden world (pre-sampled, invisible to agents)

- **Pain-point pool** `P` (e.g. 12) and **feature pool** `F` (e.g. 12).
- **Hidden mapping** `solves: P → F` (which feature addresses which pain). This is the
  "supply–demand formula" agents must reverse-engineer.
- **Demography** of ~10,000 **numeric** users (NOT LLM-backed — simulated with formulas,
  so the population is free to run). Each user `u`:
  - `pain_points ⊂ P` (1–3), sampled so some pains are common ("biggest needs") and
    some rare — a skewed distribution agents must discover.
  - `interests` / keywords (used for ad targeting match).
  - `willingness_to_pay` (wtp, $), `price_sensitivity`.
- Everything is seeded → deterministic and resettable. Paired seeds across models.

---

## 2. Firm state (visible, evolves over the episode)

- `built_features ⊂ F`, each with `implementation_quality ∈ [0,1]`.
- `price` (current), `cash`, `budget` allocation.
- History of campaign results (the firm's own observations).

---

## 3. The four agents (partial observability)

| Agent | Decides | Private info |
|---|---|---|
| **Builder** | which features to build from `F`, writes the **feature spec** | build costs, current tech state |
| **Marketer** | **ad copy** + targeting, channel | past campaign engagement stats |
| **Pricer** | `price` tiers + **budget split** (build vs marketing) | unit economics, cash/runway |
| **Coordinator** | strategy, which **experiments** to run, aligns the three | board/goal, summary state |

They coordinate via a shared planning doc + message log. The split forces info-sharing:
the Marketer must know what the Builder shipped; the Pricer must know expected demand;
the Coordinator sequences explore→exploit.

---

## 4. Episode loop

Per round (e.g. ~12 rounds = "months"):
1. Coordinator proposes the round's plan/experiment.
2. Builder (maybe) builds a feature + writes spec; Marketer writes ad(s) + targeting;
   Pricer sets price + spend.
3. **Market simulation** runs the funnel (below) over the demography.
4. Firm receives **diagnostic feedback** (see §6). Cash updates; bankruptcy if cash<0.
5. Repeat. Profit accrues. Valuation/profit at horizon = reward.

---

## 5. The funnel + formulas (judge → params → market outcome)

For each ad campaign (copy `C`, spend `S`, price `π`):
1. **Translate** `C` → `craft ∈ [0,1]` (cheap LLM judge) + `targeting` vector over
   pains/keywords (mostly **deterministic** keyword/embedding match → low noise).
2. **Reach:** `impressions = S × k`; audience drawn weighted by
   `match(targeting, u.interests)` (good targeting concentrates spend → lower CPA).
3. **Try:** `p_try(u) = craft × resonance(targeting, u.pain_points)` — the ad converts
   only if it speaks to a pain the user actually has.
4. **Purchase ("API call to the product"):** trying user checks fulfillment:
   `fulfilled = Σ_{p ∈ u.pain_points} [solves(p) ∈ built_features] × implementation_quality`
   `fulfilled_fraction = fulfilled / |u.pain_points|`
   `p_buy(u) = sigmoid(α·fulfilled_fraction + β·(u.wtp − π)/u.wtp)`
5. **Outcome:** `revenue += purchases × π`; `cost += S + build_costs`; `profit = revenue − cost`.

Feature spec is translated similarly: spec → which feature(s) it implements + an
`implementation_quality` (partial fulfillment if the spec is vague).

**What's hidden and must be learned via experiments:** the pain distribution (biggest
needs), the `solves` mapping (build the right features), and price elasticity (wtp).

---

## 6. Observations / feedback — THE make-or-break design decision

Agents can only discover the hidden structure if feedback is **diagnostic**, not just a
single revenue number (that would be a high-variance bandit → nothing learnable).
Each campaign returns:
- impressions, **try-rate**, **purchase-rate**, revenue, spend;
- ideally **per-segment breakdowns** (by targeted keyword / pain) so an experiment is
  informative (e.g. "ads targeting pain X convert; feature for X isn't built yet").

Design and validate this FIRST — it determines whether the env is learnable at all.

---

## 7. Reward + diagnostic metrics (all computable — world is numeric)

- **Reward (primary): profit** at horizon. Bankruptcy = 0/penalty.
- **Discovery efficiency:** `oracle_profit − firm_profit` (regret). Oracle knows the
  demography → picks best features + targeting + price. Computable because the world is
  numbers → the regret guarantee from the original plan is RESTORED.
- **Coordination tax:** `omniscient_single_planner_profit − multi_agent_firm_profit`.
- Variance control: multiple + paired seeds; report distributions.
- Leaderboard axes: frontier models × (single vs multi-agent) × communication protocol.

---

## 8. Judges (translators, cheap)

- **Ad copy** → `craft` (cheap LLM, averaged 1–3 samples) + `targeting` (deterministic
  keyword/embedding match → near-zero noise/cost).
- **Feature spec** → feature id(s) + `implementation_quality` (cheap LLM or rubric).
- Mock judges during development (rule-based) so the harness is built for free; swap in
  real translators only for final leaderboard runs.

---

## 9. Cost (cheap)

- 10k-user funnel = pure numpy, free per round.
- LLM spend = a few artifact translations/round on a cheap model + the 4 agent calls.
- A full leaderboard easily within sponsor credits. Judges are NOT the cost driver;
  agent context growth is — pass compact structured state, not full transcripts.

---

## 10. 24-hour build plan

| Hours | Build |
|-------|-------|
| 0–4 | Numeric world: pools, hidden `solves` mapping, demography sampler, user object. Seeding. |
| 4–8 | Funnel + formulas (reach/try/buy), **diagnostic feedback**, profit accounting. Validate learnability with a scripted experimenter. |
| 8–10 | Oracle + omniscient-planner baselines (for the two diagnostic metrics). |
| 10–16 | Agent harness: 4 roles, partial-obs views, shared plan/log, round loop. Translators (ad craft + deterministic targeting; spec→feature). Wrap as a HUD environment. |
| 16–21 | Run frontier models (paired seeds); leaderboard (profit + discovery efficiency + coordination tax); replay viewer. |
| 21–24 | Demo polish + pitch. |

---

## 11. Risks & mitigations

1. **Identifiability / observation design** (#1 risk) → ship per-segment diagnostic
   feedback; validate with a scripted experimenter agent before plugging in LLMs.
2. **Balancing for discoverability** → tune so the hidden structure is inferable within
   the round budget; not trivial, not impossible.
3. **Keeping coordination meaningful** → ads must target built-for pains at a payable
   price; coordination tax metric makes misalignment visible.
4. **Judge noise** → keep targeting deterministic; only `craft` is LLM; average samples.

---

## 12. Is this an RL environment? (formalization)

Yes — a **partially-observable, multi-agent environment (Dec-POMDP)** with an outcome reward.

- **State** `s`: hidden world (demography, `solves` mapping) + firm state (built features,
  price, cash, history).
- **Action** `a` (per agent, per round): Builder→build+spec, Marketer→copy+targeting,
  Pricer→price+budget, Coordinator→plan. Structured/text actions (tool calls); joint
  action = all roles.
- **Observation** `o`: partial, role-specific + diagnostic campaign feedback (§6).
- **Transition** `T`: deterministic funnel given seed; the LLM craft-translator is the
  one stochastic piece (cache it for reproducibility).
- **Reward** `r`: per-round profit (dense) + terminal valuation. This is the trainable signal.
- **Episode**: ~12 rounds; **resettable** via seed.

**Why it's RL-native (not a static benchmark):** success requires **explore→exploit of
hidden latent structure** (the demand formula) — a structured POMDP/contextual-bandit.
One-shot task benchmarks (e.g. TheAgentCompany) lack this.

**Eval vs. training:** the scoped deliverable (env + leaderboard) is an RL environment
*used as an eval*. It becomes a full RL training loop by wiring profit into RFT
(e.g. GRPO on Fireworks) — the stretch goal. HUD wants both.

**Trainability notes:** keep reward **dense** (per-round profit) and **fast/cheap/
deterministic** (deterministic targeting + cached craft judge). MARL is harder to train;
simplest path is one shared/centralized policy emitting all four role actions.

## 13. What is being learned? (domain randomization — REQUIRED for training)

**Trap:** if the hidden world is *fixed* across episodes, training just memorizes one
answer key (build feature 7, target X, price $42) → overfitting, and the discovery
mechanic becomes pointless. The user's instinct ("just fitting the hidden state") is
correct *for a fixed world*.

**Fix:** **randomize the hidden world every episode** (procedural generation — new pain
distribution, new `solves` mapping, new wtp distribution per rollout). The agent then
can't memorize a specific answer; to earn reward across many worlds it must learn the
**meta-skill of discovery itself**. Eval uses **held-out randomized worlds** → the
leaderboard measures generalization, not memorization.

**Transferable behaviors reinforced (under randomization):**
1. Efficient experimentation / active learning (campaigns chosen to maximize info).
2. Explore→exploit budgeting (when to stop probing and scale, without bankruptcy).
3. Belief updating from diagnostic feedback (near-Bayesian over observations).
4. Cross-functional coordination (align build ↔ targeting ↔ price; share private info).
5. Artifact craft that transfers (quality generalizes; targeting adapts to discovered demand).

Net: training teaches **"running a data-driven business under uncertainty,"** a meta-
policy (an experimentation strategy), not world-specific facts.

**Honest caveats:**
- *Transfer to real business is unproven* — claim "discovery + coordination under
  uncertainty, measured by generalization to unseen markets," not "better at real companies."
- *Reward-hacking the judge* — a policy may learn copy that games the craft-translator.
  Mitigate: deterministic targeting, craft is only one input (right feature+price+targeting
  still required to sell), robust rubric.

## 14. One-line pitch
> FirmBench drops a team of agents into a market whose demand structure is hidden in a
> 10,000-user simulation. To make money they must *experiment* to discover what people
> need, *build* it, *market* it to the right people, and *price* it right — and we
> measure profit, how close they get to the oracle (discovery efficiency), and how much
> they lose to poor coordination (coordination tax).
