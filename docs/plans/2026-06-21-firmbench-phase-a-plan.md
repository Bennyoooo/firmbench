# Plan: FirmBench Phase A — Persona + LTV core (full build, ablation-gated)

> **Status:** Revised after plan-review round 1 — CR1–CR4 (test import, dropped
> channel through `step()`, elasticity-test tautology, missing budget dials) and
> S1–S6 addressed. See `firmbench-phase-a-plan-review-1.md`.

**Goal**: Add the Phase A latents (persona segments, channels, per-user elasticity,
quality_bar, subscription/LTV with responsive+segment-varied churn) to the sim,
each behind a `Config` ablation flag, with scaling budget/horizon dials and
separability diagnostics, so the learnability gate (`naive < scripted < oracle`)
becomes a **bisection tool** for the "fix-reactively" loop.

**Architecture**:
- Every new latent is gated by a `Config` boolean (default **OFF** → v1 reproducible
  + clean ablation baseline). `Config.phase_a(**overrides)` turns them all ON.
- New RNG draws happen **after** all v1 draws and **only when the flag is on**, so
  flags-off worlds are byte-identical to today (same trick `generate_world` already
  uses for names/keywords, `sim.py:174-178`).
- Hybrid population: K hidden segments (backbone) + per-user noise on wtp/elasticity.
- Funnel stays **expectation-based / deterministic** (sum probabilities, never
  sample). **Invariant: no term may depend on the *absolute* subscriber count**
  (network effects / caps / integer rounding) — that is what would break the
  verifier's proportional-holdout scaling (confirmed in design review). **Also (S4):
  no campaign may saturate its matching pool asymmetrically across the holdout split**
  (`impressions ≥ pool size` at full scale but not at 0.2 scale, or vice versa) — that
  desyncs reach proportionality. Holds in v1 today; keep it true.
- Diagnostics gain a **bounce-reason breakdown** (`reached / tried / bounced_quality
  / bounced_price / purchased`) so confounded latents become separately observable.
- The craft live/holdout bug is fixed in the same pass (`craft` applied in the live
  funnel, not only in holdout grading).

**Tech Stack**: Python 3 (stdlib only in `sim.py`), plain-assert test scripts run
with `python3` (no pytest dependency), `python3 sim.py` as the learnability gate.

**Commit policy**: commit steps use `git` (this is a git repo, not sl). Per the
working agreement, commits happen only when Benny asks — the `git commit` lines are
provided but should not be run unprompted.

---

## Task dependencies

| Group | Steps | Can parallelize |
|-------|-------|-----------------|
| 1 | Step 1 (Config flags/dials) | No (foundation) |
| 2 | Step 2 (Segment model + world gen) | No (depends on 1) |
| 3 | Steps 3–5 (funnel: channels+craft, elasticity, quality_bar+bounce) | Partially (same function `_run_campaign` / `_p_buy` — serialize 3→4→5) |
| 4 | Step 6 (LTV/base in FirmEnv.step) | Yes (depends on 2; parallel with Group 3) |
| 5 | Steps 7–9 (Naive, Scripted, Oracle policies) | Yes (independent files-of-thought; depends on 3–6) |
| 6 | Step 10 (ablation gate harness) | No (depends on 5) |
| 7 | Steps 11–13 (env.py live-craft fix + channels/diagnostics; env.py stateful holdout; run.py stateful verifier + oracle guard) | Partially (11→12 same file; 13 parallel) |
| 8 | Step 14 (end-to-end verification) | No |

Test scripts live in `tests/test_phase_a.py` (created in Step 1). Run with
`python3 tests/test_phase_a.py` from the repo root.

---

## Step 1: Config flags, dials, and `phase_a()` factory

**File**: `/Users/bennyjiang/Desktop/projects/firmbench/sim.py` (modify `Config`),
`/Users/bennyjiang/Desktop/projects/firmbench/tests/test_phase_a.py` (new)

### 1a. Write failing test
```python
# tests/test_phase_a.py — run from repo root: python3 tests/test_phase_a.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # CR1: put repo root on sys.path
from sim import (Config, generate_world, run_episode, FirmEnv,
                 NaivePolicy, ScriptedExperimenter, OraclePolicy)

def test_flags_default_off_preserves_v1():
    c = Config()
    assert not (c.use_segments or c.use_channels or c.use_elasticity
                or c.use_quality_bar or c.use_retention), "Phase A flags must default OFF"

def test_phase_a_factory_turns_all_on():
    c = Config.phase_a()
    assert (c.use_segments and c.use_channels and c.use_elasticity
            and c.use_quality_bar and c.use_retention)

def test_phase_a_scales_budget():            # CR4: C2 budget/horizon dials must scale
    c = Config.phase_a()
    assert c.horizon > 10 and c.starting_cash > 6000.0

if __name__ == "__main__":
    import sys, traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"ok   {name}")
            except Exception:
                fails += 1; print(f"FAIL {name}"); traceback.print_exc()
    sys.exit(1 if fails else 0)
```

### 1b. Run test to verify it fails
```bash
cd /Users/bennyjiang/Desktop/projects/firmbench && python3 tests/test_phase_a.py
```

### 1c. Write implementation
Add to `Config` in `sim.py` (after `price_grid`):
```python
    # ── Phase A ablation flags (default OFF → v1 reproducible) ──
    use_segments: bool = False
    use_channels: bool = False
    use_elasticity: bool = False
    use_quality_bar: bool = False
    use_retention: bool = False
    # population / channel structure
    n_segments: int = 5
    n_channels: int = 3
    channel_fit_off: float = 0.25      # reach multiplier on the wrong channel
    # elasticity: per-segment beta mean + per-user noise
    elasticity_mu: float = 2.0
    elasticity_sigma: float = 0.5
    # quality_bar: per-segment min fulfilled-fraction to convert (soft gate width)
    quality_bar_mu: float = 0.3
    quality_bar_sigma: float = 0.1
    quality_gate_k: float = 8.0        # softness of the bar (higher = harder gate)
    # retention / subscription
    subscription: bool = True
    churn_base: float = 0.10
    churn_price_coef: float = 0.30     # churn rises when price > wtp
    churn_quality_coef: float = 0.30   # churn rises when fulfilled < bar

    @classmethod
    def phase_a(cls, scale_budget=True, **overrides):
        base = dict(use_segments=True, use_channels=True, use_elasticity=True,
                    use_quality_bar=True, use_retention=True)
        if scale_budget:
            # CR4 / review C2: the experiment space grows ~ n_pains × n_channels, so
            # scale the probe budget + horizon with it. These are the dials to tune per
            # the ablation gate (Step 10) — and the lever the reactive loop pulls to
            # tell a *budget* failure apart from an *unobservability* failure.
            nch = overrides.get("n_channels", 3)
            base["horizon"] = 10 + 2 * nch
            base["starting_cash"] = 6000.0 * nch
        base.update(overrides)
        return cls(**base)
```

### 1d. Run test to verify it passes
```bash
cd /Users/bennyjiang/Desktop/projects/firmbench && python3 tests/test_phase_a.py
```
Also confirm v1 is unchanged: `python3 sim.py` still prints the **mean row** naive≈4035, scripted≈69783, oracle≈138570 (S6: per-seed values differ; only the mean row is the regression check).

### 1e. Commit (only if asked)
```bash
git add sim.py tests/test_phase_a.py && git commit -m "Phase A step 1: Config ablation flags + dials + phase_a() factory"
```

---

## Step 2: Segment model + hybrid world generation

**File**: `sim.py` (add `Segment`, extend `World`, branch `generate_world`)

### 2a. Write failing test
```python
def test_segments_present_when_enabled():
    w = generate_world(1, Config.phase_a())
    assert len(w.segments) == 5
    assert all(0 <= u.segment_id < 5 for u in w.users[:50])
    # hybrid: per-user wtp varies within a segment
    seg0 = [u.wtp for u in w.users if u.segment_id == 0][:20]
    assert len(set(round(x,3) for x in seg0)) > 1, "per-user noise expected"

def test_v1_world_unchanged_when_disabled():
    a = generate_world(3, Config())
    assert a.segments is None or a.segments == []   # off → no segments
    # demography identical to pre-Phase-A: spot-check popularity vector is stable
    assert a.pain_popularity == generate_world(3, Config()).pain_popularity
```

### 2b. Run test (fails — `segments`/`segment_id` don't exist yet)
```bash
python3 tests/test_phase_a.py
```

### 2c. Write implementation
- Add `Segment` dataclass: `pain_affinity: list`, `wtp_mu`, `wtp_sigma`,
  `elasticity_mu`, `channel_pref: int`, `quality_bar: float`, `churn_base: float`,
  `weight: float`.
- Add to `User`: `segment_id: int = -1`, `elasticity: float = None`,
  `channel_pref: int = 0`, `quality_bar: float = 0.0` (all default to v1-neutral).
- Add to `World`: `segments: list = None`.
- In `generate_world`, after the v1 `pain_popularity` line (S3: gating keeps flags-off
byte-identical regardless of where this goes; place it consistently right after
`pain_popularity`, before `_sample_names`), add:
```python
    segments = None
    if cfg.use_segments:
        segments = _make_segments(rng, cfg, weight_by_pain)
        users = _resample_users_from_segments(rng, cfg, segments)
        # rebuild users_by_pain / pain_popularity from the new users
        users_by_pain = {p: [] for p in pains}
        for idx, u in enumerate(users):
            for p in u.pains:
                users_by_pain[p].append(idx)
        pain_popularity = [len(users_by_pain[p]) for p in pains]
```
- `_make_segments`: K segments; `pain_affinity` = a skewed weight vector per segment
  (reuse the zipf idea, shuffled per segment → correlated pain clusters);
  `channel_pref` = `rng.randrange(cfg.n_channels)`; `quality_bar` ~
  `clamp(normal(quality_bar_mu, quality_bar_sigma), 0, 0.9)`; `elasticity_mu` ~
  `normal(cfg.elasticity_mu, 0.4)`; `churn_base` ~ `clamp(normal(cfg.churn_base,0.04),0.02,0.4)`;
  `weight` = zipf over segment rank (some segments dominate → "which segments matter").
- `_resample_users_from_segments`: for each user pick a segment by `weight`; draw
  `pains` from that segment's `pain_affinity` (1–3, weighted-without-replacement);
  `wtp` ~ lognormal(seg.wtp_mu, seg.wtp_sigma); `elasticity` =
  `seg.elasticity_mu + normal(0, cfg.elasticity_sigma)` (per-user noise = the hybrid);
  `channel_pref = seg.channel_pref`; `quality_bar = seg.quality_bar`.
- Return `World(..., segments=segments)`.

### 2d. Run test to verify it passes
```bash
python3 tests/test_phase_a.py && python3 sim.py   # v1 numbers still unchanged (flags off)
```

### 2e. Commit (only if asked)
```bash
git commit -am "Phase A step 2: hybrid persona segments + segment-based world gen (gated)"
```

---

## Step 3: Channels + craft in the funnel (and fix the live-craft bug)

**File**: `sim.py` (`FirmEnv._run_campaign`)

### 3a. Write failing test
```python
def _popular_pain(w, cfg):
    return max(range(cfg.n_pains), key=lambda x: w.pain_popularity[x])

def test_channel_reaches_preferred_segment():
    cfg = Config.phase_a()
    w = generate_world(1, cfg); env = FirmEnv(w)
    # a campaign on the right channel should reach more matching users than wrong channel
    p = _popular_pain(w, cfg)
    pref = w.users[w.users_by_pain[p][0]].channel_pref
    r_right = env._run_campaign({p}, 200.0, channel=pref)
    r_wrong = env._run_campaign({p}, 200.0, channel=(pref + 1) % cfg.n_channels)
    assert r_right["tries"] >= r_wrong["tries"]

def test_craft_scales_tries_live():
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    p = _popular_pain(w, cfg)                 # S2: use a populated pain, not literal {0}
    base = env._run_campaign({p}, 200.0, channel=0, craft=1.0)["tries"]
    half = env._run_campaign({p}, 200.0, channel=0, craft=0.5)["tries"]
    assert half < base

def test_channel_forwarded_through_step():    # CR2: channel must survive env.step(), not just direct calls
    cfg = Config.phase_a(); w = generate_world(1, cfg)
    p = _popular_pain(w, cfg); pref = w.users[w.users_by_pain[p][0]].channel_pref
    e1 = FirmEnv(w); o1, _, _, _ = e1.step({"build": None, "price": 50.0,
        "campaigns": [{"target": {p}, "spend": 200.0, "channel": pref}]})
    e2 = FirmEnv(w); o2, _, _, _ = e2.step({"build": None, "price": 50.0,
        "campaigns": [{"target": {p}, "spend": 200.0, "channel": (pref + 1) % cfg.n_channels}]})
    assert o1["per_campaign"][0]["tries"] >= o2["per_campaign"][0]["tries"]
```

### 3b. Run test (fails — `channel`/`craft` params don't exist)
```bash
python3 tests/test_phase_a.py
```

### 3c. Write implementation
Change signature: `def _run_campaign(self, target, spend, channel=0, craft=1.0):`
- Build the matching pool as today. When `cfg.use_channels`, compute a per-user
  reach weight `channel_fit = 1.0 if u.channel_pref == channel else cfg.channel_fit_off`
  and **order the pool by (channel_fit desc, idx)** before slicing `[:impressions]`
  (right channel reaches your segment first; wrong channel mostly off-segment).
- `p_try = craft * resonance` (× `channel_fit` when channels on). This fixes the
  bug where craft only affected holdout grading.
- Keep the `audience` field = full pool size (free demand signal, unchanged).
- **CR2 — forward the new params through `step()`.** Update the live call site
  `sim.py:282` from `res = self._run_campaign(c.get("target", set()), spend)` to
  `res = self._run_campaign(c.get("target", set()), spend, channel=c.get("channel", 0), craft=c.get("craft", 1.0))`.
  Without this, `step()` silently drops `channel`/`craft`, every campaign runs on
  channel 0, and the `+channels` ablation row is meaningless.
- (Bounce-reason fields added in Step 5.)

### 3d. Run test to verify it passes
```bash
python3 tests/test_phase_a.py && python3 sim.py
```

### 3e. Commit (only if asked)
```bash
git commit -am "Phase A step 3: channel-aware reach + craft in live funnel (gated; fixes craft bug)"
```

---

## Step 4: Per-user elasticity in the purchase model

**File**: `sim.py` (`FirmEnv._p_buy`)

### 4a. Write failing test
```python
def test_elasticity_uses_per_user_beta():    # CR3: must FAIL on v1 (shared cfg.beta), pass after per-user beta
    import sim as S
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    # pick a user whose elasticity is clearly different from the global cfg.beta
    u = max(w.users[:500], key=lambda x: abs(x.elasticity - cfg.beta))
    assert abs(u.elasticity - cfg.beta) > 1e-3, "need a user with non-default elasticity"
    for p in u.pains:
        env.built[w.solves[p]] = 1.0
    env.price = 90.0
    ff = env._fulfilled_fraction(u); pt = (u.wtp - env.price) / u.wtp
    per_user = S.sigmoid(cfg.alpha * ff + u.elasticity * pt - cfg.gamma)   # post-impl formula
    v1_value = S.sigmoid(cfg.alpha * ff + cfg.beta      * pt - cfg.gamma)  # v1 (shared beta)
    got = env._p_buy(u)
    assert abs(got - per_user) < 1e-9    # _p_buy now uses beta_u = user.elasticity
    assert abs(got - v1_value) > 1e-6    # ...and that genuinely differs from v1
```

### 4b. Run test
```bash
python3 tests/test_phase_a.py
```

### 4c. Write implementation
In `_p_buy`, use `beta_u = user.elasticity if (cfg.use_elasticity and user.elasticity is not None) else cfg.beta`:
```python
    return sigmoid(cfg.alpha * ff + beta_u * price_term - cfg.gamma)
```

### 4d–4e. Verify + commit
```bash
python3 tests/test_phase_a.py && python3 sim.py
git commit -am "Phase A step 4: per-user elasticity in purchase model (gated)"
```

---

## Step 5: quality_bar soft gate + bounce-reason diagnostics (separability)

**File**: `sim.py` (`FirmEnv._p_buy`, `_run_campaign`)

### 5a. Write failing test
```python
def test_quality_bar_emits_distinct_bounce_signal():
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    p = max(range(cfg.n_pains), key=lambda x: w.pain_popularity[x])
    pref = w.users[w.users_by_pain[p][0]].channel_pref
    # feature built but LOW quality → bounces should be attributed to quality, not price
    env.built[w.solves[p]] = 0.05; env.price = 25.0
    r = env._run_campaign({p}, 400.0, channel=pref)
    assert r["bounced_quality"] > r["bounced_price"]
    assert r["purchases"] < 1.0
```

### 5b. Run test
```bash
python3 tests/test_phase_a.py
```

### 5c. Write implementation
- Soft gate in `_p_buy` when `cfg.use_quality_bar`:
  `gate = sigmoid(cfg.quality_gate_k * (ff - user.quality_bar))`; `p_buy *= gate`.
  (Soft, not hard — keeps the signal smooth and discoverable; the design's hard gate
  was the main C1 confound.)
- In `_run_campaign`, accumulate per-reached-user buckets so the agent can *separate*
  failure modes:
  - `bounced_quality += p_try * (1 - gate)` (resonated, blocked by quality bar)
  - `bounced_price   += p_try * gate * (1 - price_accept)` where
    `price_accept = sigmoid(beta_u * price_term)` (resonated, passed quality, lost on price)
  - `purchases += p_try * p_buy` (as today)
  Return these in the result dict alongside `tries`, `purchases`, `revenue`.

### 5d–5e. Verify + commit
```bash
python3 tests/test_phase_a.py && python3 sim.py
git commit -am "Phase A step 5: soft quality_bar gate + bounce-reason diagnostics (separability)"
```

---

## Step 6: Subscription base + responsive, segment-varied churn (LTV)

> **S5 — split for granularity:** do this as **6a** (reset `base` + recurring revenue +
> per-segment responsive churn) and **6b** (segment-attributed acquisition from
> campaign purchases), each with its own failing test. Keep each ≤5 min.

**File**: `sim.py` (`FirmEnv.reset`, `FirmEnv.step`)

### 6a. Write failing test
```python
def test_base_persists_and_churns():
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    # build a popular feature, set a fair price, run a big campaign → acquire subscribers
    p = max(range(cfg.n_pains), key=lambda x: w.pain_popularity[x])
    env.step({"build": w.solves[p], "price": 40.0,
              "campaigns": [{"target": {p}, "spend": 1000.0, "channel": w.segments[0].channel_pref}]})
    base_after_acq = sum(env.base.values()); assert base_after_acq > 0
    # a round with no new campaigns → recurring revenue > 0 and base decays by churn
    obs, profit, done, _ = env.step({"build": None, "price": 40.0, "campaigns": []})
    assert profit > 0                      # recurring revenue from retained subscribers
    assert sum(env.base.values()) < base_after_acq   # churn applied
```

### 6b. Run test
```bash
python3 tests/test_phase_a.py
```

### 6c. Write implementation
- `reset()`: `self.base = {}  # segment_id -> expected active subscribers` (only used when `cfg.use_retention`).
- In `step()`, when `cfg.use_retention and cfg.subscription`:
  1. **recurring** = `sum(base.values()) * price` added to revenue (before churn).
  2. **churn** per segment: `churn_eff = clamp(seg.churn_base + churn_price_coef*max(0,(price-seg_wtp_med)/seg_wtp_med) + churn_quality_coef*max(0, avg_quality_gap), 0, 0.95)`; `base[s] *= (1 - churn_eff)`.
     - `churn_price_coef`/`churn_quality_coef` from Config → **responsive**; `seg.churn_base` differs per segment → **segment-varied**.
  3. **acquisition**: campaigns' `purchases` are attributed to segments by the
     composition of reached users (track per-segment purchases inside `_run_campaign`,
     return a `by_segment` dict) and added to `base[s]`.
  - **Invariant guard** (cheap assert in a debug path): churn_eff and per-user terms
    must not reference `len(base)` or rounded counts — only price/quality/seg params.
- One-time mode (`subscription=False`) keeps v1 behavior: purchases are one-shot
  revenue, no base.

### 6d–6e. Verify + commit
```bash
python3 tests/test_phase_a.py && python3 sim.py
git commit -am "Phase A step 6: subscription base + responsive segment-varied churn (LTV, gated)"
```

---

## Steps 7–9: Update policies (Naive, Scripted, Oracle)

**Files**: `sim.py` (`NaivePolicy`, `ScriptedExperimenter`, `OraclePolicy`)

For each: actions must now carry a `channel` per campaign (default 0), and the
experimenter/oracle must use the new signals. Each step has a test asserting the
policy *runs* under `Config.phase_a()` without error and produces valid actions; the
real proof is the gate (Step 10).

- **Step 7 — NaivePolicy**: also pick a random `channel`. Test: action dict has
  `channel` in each campaign; episode runs under `phase_a()`.
- **Step 8 — ScriptedExperimenter** (the important one; **S5: split into 8a channel
  discovery + 8b LTV-valued exploit**, each with its own test):
  - Round 0: probe each pain on **each channel** cheaply → discover channel↔segment
    from per-channel `tries`/`audience`.
  - Build + test: attribute feature→pain via `purchases` delta as today, but read
    `bounced_quality` to tell "wrong feature" from "low quality"; if a built feature
    shows high `bounced_quality`, mark it low-quality (would need a better spec —
    Phase C lever; for now record and prefer features with low bounce).
  - Use `bounced_price` to drive the price search (raise/lower until `bounced_price`
    minimized at the target margin).
  - Exploit: target solved popular pains on their best channel; in subscription mode,
    value targets by **LTV** (`price / churn_eff`) not one-shot revenue.
  - Test: under `phase_a()`, scripted episode profit > naive episode profit on seed 1.
- **Step 9 — OraclePolicy**: knows `segments`, `solves`, channels, churn. Builds for
  the highest-`weight` segments' pains, targets each on the segment's `channel_pref`,
  prices to maximize LTV. Test: under `phase_a()`, oracle ≥ scripted on seed 1.

```bash
# after each: python3 tests/test_phase_a.py
git commit -am "Phase A steps 7-9: channel/LTV-aware naive, scripted, oracle policies"
```

---

## Step 10: Ablation gate harness (the bisection tool)

**File**: `sim.py` (extend `main()` or add `ablation_gate()`),
optionally `/Users/bennyjiang/Desktop/projects/firmbench/ablate.py` (new, thin driver)

### 10a. Write failing test
```python
def test_ablation_gate_reports_each_latent():
    from sim import ablation_gate
    rows = ablation_gate(seeds=[1,2,3])     # returns list of dicts
    keys = {r["config"] for r in rows}
    assert {"v1","+segments","+channels","+elasticity","+quality_bar","+retention","full"} <= keys
    for r in rows:
        assert "naive" in r and "scripted" in r and "oracle" in r
```

### 10b. Run test
```bash
python3 tests/test_phase_a.py
```

### 10c. Write implementation
`ablation_gate(seeds)`: for each config in
`[v1, +each-single-flag, full(phase_a)]`, run naive/scripted/oracle over the seeds,
return mean profits + `disc_eff` + a **PASS/FAIL** on `naive < scripted < oracle` and
a **WARN** if `scripted > oracle` (signals the oracle/reference is no longer a valid
ceiling — review issue C3). Print a table. This is the **reactive-fix driver**: the
first config whose gate FAILS names the latent that broke identifiability.

### 10d. Run + interpret
```bash
cd /Users/bennyjiang/Desktop/projects/firmbench && python3 -c "from sim import ablation_gate; ablation_gate([1,2,3,4,5])"
```
Expected use: read the table. For any `FAIL` row, that latent's observation isn't
separable enough → go back and enrich its diagnostic (e.g., more channels probed,
sharper bounce signal) or reduce its difficulty (smaller K, softer gate). Re-run.

### 10e. Commit (only if asked)
```bash
git commit -am "Phase A step 10: ablation gate harness (bisection tool for reactive identifiability fixes)"
```

---

## Steps 11–13: Wire env.py + run.py to Phase A

- **Step 11 — `env.py` live-craft fix + channels/diagnostics**: pass `craft` into the
  live `_ENV._run_campaign(...)` call (currently omitted, `env.py:234`); add optional
  `channel` param to `probe_market`/`run_campaign`; surface `bounced_quality`/
  `bounced_price` in the tool return. Test: a probe with `ad_copy` (craft<1 via
  fast_mode override in a unit test) yields reduced live `tries` AND matching holdout
  → `gap` stays under the 0.15 tripwire (proves the craft bug is fixed).
- **Step 12 — `env.py` stateful holdout replay**: when `cfg.use_retention`, replicate
  the base/churn/recurring update inside `_replay_on_holdout` so held-out profit is
  LTV-based; keep `spend * _HOLDOUT_FRAC` scaling. Add the absolute-count invariant
  as an assert/comment. Test: walk a 2-round subscription action_log; assert
  `reported * 0.2 ≈ holdout` (gap < 0.15) so honest LTV agents don't self-flag.
- **Step 13 — `run.py` Verifier**: same stateful replay in `Verifier.replay_on_holdout`;
  add a loud guard in `evaluate_policy`/`main` that prints a WARNING if any policy's
  reward exceeds the oracle's (review C3); relabel "oracle" diagnostics as
  "reference ceiling" where the bound isn't proven. **S1:** `run.py`'s replay +
  action_log omit `craft` (consistent for pure Phase A where craft=1.0); add the
  craft term here when Phase C policies start emitting craft, so it isn't forgotten.

```bash
python3 tests/test_phase_a.py
git commit -am "Phase A steps 11-13: env.py craft fix + channels + stateful holdout; run.py stateful verifier + oracle guard"
```

---

## Step 14: End-to-end verification

### 14a. Run the full suite + gate + head-to-head
```bash
cd /Users/bennyjiang/Desktop/projects/firmbench
python3 tests/test_phase_a.py                                   # all unit tests pass
python3 -c "from sim import ablation_gate; ablation_gate([1,2,3,4,5])"   # ablation table
python3 sim.py                                                  # v1 numbers UNCHANGED (flags off)
python3 run.py                                                  # head-to-head incl. stateful verifier
```

### 14b. Acceptance criteria
1. `python3 sim.py` (flags off) prints the **same** v1 **mean row** as today
   (naive≈4035, scripted≈69783, oracle≈138570; per-seed values differ) → no regression.
2. Ablation table: each single-latent row and `full` reports naive/scripted/oracle;
   record which rows **PASS** `naive < scripted < oracle` and which **FAIL/WARN**.
   (Per the chosen "fix reactively" path, FAIL rows are the work-list, not a blocker.)
3. No row triggers the `scripted > oracle` WARN; if it does, strengthen the oracle.
4. `run.py` stateful verifier: honest scripted policy is **not** flagged (gap<0.15)
   under `phase_a()` (proves stateful holdout + craft fix are consistent).

### 14c. Commit (only if asked)
```bash
git commit -am "Phase A step 14: end-to-end verification (ablation gate green/work-list, no v1 regression)"
```

---

## Reactive-fix loop (how to use this plan given the chosen path)

1. Build Steps 1–13 (full Phase A, all latents).
2. Run Step 10's ablation gate.
3. For each `FAIL` row (latent that breaks `naive < scripted`): the culprit is named.
   Enrich that latent's *observation* (more probing axes / sharper bounce signal) or
   ease its *difficulty* (smaller K, softer gate, larger `channel_fit_off`), then
   re-run the gate. Repeat until `full` passes.
4. For any `scripted > oracle` WARN: strengthen `OraclePolicy` or relabel the metric.
5. Only then layer Phase B difficulty dials onto a Phase A that is *proven* learnable.
