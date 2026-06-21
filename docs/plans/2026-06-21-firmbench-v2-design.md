# FirmBench v2 — Integrated Expansion Design

> **Status:** Phase A locked (validated via brainstorming, 2026-06-21).
> Phases B/C/D specified; key forks marked **[OPEN]**.
> **Date:** 2026-06-21

## Goal

Expand FirmBench across all four dimensions the team identified — **customer
realism + LTV**, **harder discovery**, **NL artifacts + LLM judge**, and
**multi-agent** — done as **layers on one keystone change**, not four parallel
features. Each phase ships independently and is gated by the existing learnability
check (`naive < scripted < oracle`).

## Design principle — the matched quad

Add every new dimension as a **matched quad**:

1. a **latent** in the world,
2. an **action** lever to exploit it,
3. an **observation** channel to discover it, and
4. a **reward** term to grade it.

Drop any leg and the dimension is dead weight: no observation → unlearnable; no
action → inert; no reward term → unguided. Re-validate the learnability gate after
every phase.

---

## v1 baseline — dimensional inventory

### Entities

| Character | Dimensions today | Currently fixed / trivial |
|---|---|---|
| **Market / World** | `n_pains`=8, `n_features`=8; hidden `solves: P→F` **bijection**; Zipf pain popularity; re-randomized per seed | bijection over tiny pools (oracle brute-forceable); static; no competitors |
| **Customers / Users** | 5,000 users; per user `pains` (1–3, weighted) + `wtp` (lognormal ~$49); 20% held out | only 2 attributes; independent sampling; **memoryless** (re-buy every round) |
| **Firm** | `cash` ($6k), `built_features` w/ `implementation_quality` (**=1.0**), one global `price`, `round` | quality dim dormant; one product; one-shot revenue |
| **Founder / Agent** | 3 real levers: `build` (1/rd), `set_price` (global), `campaign(target_pains, spend)`; target **by pain only** | `probe_market` ≡ `run_campaign`; no channel/quality/segment levers |
| **Product / Features** | `feature_id`; flat $300; instant; binary; solves 1 pain | quality unused; no NL layer |
| **Competitors** | — *absent* — | n/a |

### Systems

| System | Today | 
|---|---|
| **Funnel** | reach = `spend×0.1` (first-N by index); try = resonance; buy = `sigmoid(4·ff + 2·(wtp−π)/wtp − 3)`; deterministic; craft=1.0 |
| **Verifier** | **deterministic, no LLM**; secret 20% holdout + replay → profit; cheat tripwire (gap>15%); reward = profit/30k ∈ [0,1] |
| **Episode / RL** | single-agent POMDP; H=10; dense profit; train vs disjoint eval seeds; domain-randomized |

**Note:** there is no LLM judge anywhere in v1 — grading is 100% deterministic.

---

## Keystone

- **Customer:** `{pains, wtp}` → **hybrid persona model** (segment backbone +
  per-user noise).
- **Firm:** one-shot revenue → **stateful LTV base** (subscribers persist, pay
  recurring, churn).

With these two in place, the other three "centers" become layers on top:
difficulty is a re-parameterization of the same model; NL+judge attaches to the
action artifacts; multi-agent splits the (larger) action space.

---

## Phase A — Persona + LTV core  **[LOCKED]**

### Decisions

| Decision | Choice |
|---|---|
| Customer model | **Hybrid** — personas set backbone (correlated pains, channel, churn); per-user noise on `wtp`, `elasticity` |
| Churn model | **Responsive + segment-varied** — `churn_eff` responds to price & quality; coefficients differ per segment |
| Revenue model | **Subscription-only** in Phase A (one-time vs subscription *choice* deferred to a Phase B dial, e.g. "B2B segments only buy via subscription") |

### Data model

```text
Segment (hidden persona) — K≈5, the new thing to discover:
  pain_affinity   : weights over pains  → correlated pain clusters
  wtp_mu, wtp_sig : willingness-to-pay distribution
  elasticity_mu   : price sensitivity (per-user noise added on top)
  channel_pref    : which channel reaches them best
  quality_bar     : min fulfilled/quality to convert
  churn_base      : baseline churn; plus segment-specific price/quality sensitivities
  weight          : share of population (the new "popularity" to discover)

User: segment_id (hidden) + pains (∼affinity) + wtp + elasticity (segment + noise)
World: + segments[], channels[] (cost/reach profile), solves (graded fit-ready)
Firm:  + base: dict[segment_id → expected_subscribers]   (LTV state, internal)
       + pricing_model = subscription;  implementation_quality now varies
Action:+ campaign = (target_pains, channel, spend); optional per-tier price
Obs:   + per-channel diagnostic breakdown; revenue split new vs recurring;
         churn count; active subscriber count (aggregate — segment stays latent)
```

### Funnel math (expectation-based, deterministic)

Campaign = `(target_pains, channel, spend)`:

- **Reach (channel = 2nd targeting axis):**
  `reach_weight(u) = 1{pains∩target≠∅} × channel_fit(channel,u)`,
  `channel_fit = 1.0` on the user's preferred channel else `ρ≈0.25`.
  The `spend × ipd` impressions go to highest-reach-weight users. Audience-by-
  channel is reported → discoverable.
- **Try:** `p_try = |target∩pains| / |pains|` (× craft multiplier in Phase C).
- **Buy:** `p_buy = sigmoid(α·ff + β_u·(wtp_u−price)/wtp_u − γ) × gate(ff ≥ quality_bar_u)`,
  where `β_u` = per-user elasticity, `ff` uses real `implementation_quality`.

### LTV dynamics (per round)

```text
recurring     = base × price
churn_eff(seg)= churn_base(seg) + price_pressure(price, wtp, seg) + quality_gap(ff, bar, seg)
base          ×= (1 − churn_eff)            # applied per segment
base          += new_subscribers            # this round's conversions, by segment
profit        = recurring + new_sales − spend − build_cost
```

### Verifier (stateful)

Holdout replay carries the 20% holdout `base` across rounds (acquire → churn →
recur), profit computed on holdout only. Tripwire unchanged; **rescale the
normalizer** (LTV profits run larger than one-shot).

### Matched quads (nothing dangles)

| Latent | Action | Observation | Reward |
|---|---|---|---|
| segment weights | target big segments | audience per (pain, channel) | profit from big segments |
| channel ↔ segment | pick channel per campaign | try/audience varies by channel | wasted spend on wrong channel |
| per-user elasticity | set price (per tier) | purchases-vs-price curve | revenue at optimal price |
| quality_bar | invest in quality / build | tries-but-no-buys when ff<bar | conversion only above bar |
| churn (seg-varied) | subscription + quality + price restraint | churn / recurring per round | LTV ≫ one-shot |

### Gate

Re-validate `naive < scripted < oracle`. Update `ScriptedExperimenter` to also
discover segments/channels; update `OraclePolicy` to know segments + optimal
retention. Confirm discovery is still the binding skill.

---

## Phase B — Difficulty dials + curriculum

Re-parameterizes the Phase A model — cheap once A exists.

- Pools configurable (8 → 16–20).
- `solves` bijection → **graded many-to-many** `fit(p,f) ∈ [0,1]`.
  `ff` becomes a sum/max of fits over a user's pains.
- **Non-stationarity:** segment weights drift over rounds / one demand shock per
  episode / optional seasonality.
- **Curriculum for RFT:** easy (bijection, stationary) → hard (dense fit,
  non-stationary). Difficulty as explicit `Config` dials.
- One-time-vs-subscription *segment split* lands here.

**Oracle tractability:** dense fit + segments can make an exact oracle expensive →
keep a greedy/approximate oracle; regret-vs-oracle stays meaningful.

**[OPEN B1]** fit density — **sparse** (each feature partially solves a few pains)
vs **dense** matrix. Sparse keeps discovery tractable + oracle cheap (recommended).
**[OPEN B2]** oracle — exact vs approximate/greedy.

---

## Phase C — NL artifacts + LLM judge-as-translator

The first LLM in the stack. Judge **translates artifacts → sim parameters**, never
emits reward directly (anti-reward-hacking; PLAN §13.5).

- **Artifact 1 — ad creative:** agent writes ad copy → judge scores per-segment
  **message-fit** → **craft multiplier ∈ [0.5, 1.3]** on `p_try`. (Turns on the
  craft=1.0 hook.)
- **Artifact 2 — feature spec:** agent writes a short spec → judge →
  **`implementation_quality ∈ [0,1]`**. (Turns on the dormant quality dim.)
- **Model:** Claude (Anthropic sponsor entry).
- **Rigor / determinism:** rubric + few-shot + **ensemble** (k judges, median) +
  **cache** keyed on `(artifact_text, segment)` → reproducible, cost-bounded.
  Numeric population still drives scale; the judge runs **per-artifact, not
  per-user**.
- **Optional diagnostic:** an LLM **process-judge** (was experimentation
  disciplined?) reported alongside reward, **not in reward**.

**Matched quad:** latent = per-segment message/quality response; action = write
copy/spec; obs = try/conversion lift from better craft; reward = profit via the
multipliers.

### Judge role — **translator + evaluator [LOCKED]**

The judge both translates artifacts → params (Job 1, always in reward via the
funnel) **and** emits a qualitative **process/strategy score that is folded into
reward** (Job 2). Engineered so the gameable term is bounded above by execution
truth:

```
reward = w_p · profit_norm  +  w_q · quality_norm     with  w_p ≫ w_q  (e.g. 0.8 / 0.2)
```

- **Quality term measures process** (experimentation discipline, strategic
  coherence) — **not** artifact craft (already rewarded via translation →
  conversions; scoring it again double-counts). Captures good process under bad luck.
- **Guardrails:**
  1. **Rubric-anchored + ensemble** (k=3, median) + **cache** by artifact/trajectory
     hash → determinism, low variance, bounded cost.
  2. **Anti-injection** — agent text is untrusted; sandbox in delimiters, judge
     treats it as data, penalizes injection attempts.
  3. **Judge-honesty tripwire** — quality score must *predict* held-out execution
     outcomes; high quality + ≈0 held-out conversion lift → flag + penalize. Ties
     the gameable term back to ground truth (secret-held-out philosophy applied to
     the judge).
  4. **Calibration monitor** — track corr(quality, held-out lift); decay ⇒ gaming
     ⇒ recalibrate `w_q`/rubric.

---

## Phase D — Multi-agent + coordination tax

Wraps the (now larger) Phase A–C action space.

- **Roles:** Builder (build + spec quality), Marketer (channels + creative +
  spend), Pricer (price/tiers), Coordinator (budget allocation + round commit).
- **Partial observability:** each role sees only its slice → must communicate.
- **Coordination tax:** shared cash budget; uncoordinated actions waste money
  (e.g. Marketer promotes an unbuilt feature). Measured as **regret vs a
  single-agent full-info oracle**.
- **Reward:** shared team profit; coordination tax = the visible gap.

**[OPEN D1]** comms model — free-form scratchpad vs structured messages.
**[OPEN D2]** coordination tax — implicit (budget contention only) vs explicit
(per-message cost).

---

## Sequencing & gates

```
A (keystone)  →  B (dials)  →  C (NL+judge)  →  D (multi-agent)
   gate:           gate:          gate:            gate:
 naive<scripted   curriculum    holdout integrity  regret vs
 <oracle          monotone      + judge cache      single-agent oracle
```

A and C deliver most of the novelty + sponsor story; B is cheap dials; D is the
big one — gate D on A–C landing.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Identifiability explosion** (new latents unobservable) | every dimension is a matched quad; validate scripted experimenter after each phase |
| **Oracle tractability** (dense fit + segments) | sparse fit; greedy/approximate oracle; regret still meaningful |
| **Judge cost / nondeterminism** | per-artifact (not per-user); ensemble + cache keyed on artifact text |
| **Reward-hacking the judge** | judge → params only, never reward; tripwire stays on held-out |
| **Luck vs skill** | keep core deterministic (expectation-based); if noise added, paired seeds (Vending-Bench lesson) |
| **Scope creep** | each phase shippable + gated; D gated on A–C |

---

## Open decisions log

- [x] Customer model = **hybrid**
- [x] Churn = **responsive + segment-varied**
- [x] Phase A revenue = **subscription-only**
- [x] **C1** judge role = **translator + evaluator** (process score folded into
  reward at minority weight; 4 guardrails incl. judge-honesty tripwire)
- [ ] **B1** fit density (default: sparse) · **B2** oracle (default: greedy/approx)
- [ ] **D1** comms model (default: structured) · **D2** coordination tax (default: implicit)
