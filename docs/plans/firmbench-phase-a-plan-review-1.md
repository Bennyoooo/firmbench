VERDICT: NEEDS_REVISION

## Summary Assessment

The plan's architecture is sound — gated latents preserve v1 reproducibility (verified: flags-off RNG is byte-identical because no `rng.*` call fires when a flag is off), the proportional-holdout scaling provably survives a compounding subscriber base (verified: `reported*0.2 == holdout`, gap 0.0), and the craft live/holdout bug + line references are all accurate. But the plan ships with four execution-blocking defects: (1) the test harness as written **cannot import `sim`** when run as `python3 tests/test_phase_a.py` from the repo root; (2) the Step 3 `_run_campaign` signature change is **never forwarded through the `step()` call site**, so `channel` is silently dropped and Step 6's own test becomes fragile; (3) the Step 4 elasticity test is a **tautology that passes on unmodified v1**; and (4) the C2 prerequisite fix (scaling budget/horizon dials) is **claimed in the goal/Step-1 title but absent from the implementation**.

## Critical Issues (must fix)

### CR1. The test harness cannot import `sim` as written — every test step fails at line 1
**Plan section:** Step 1 (lines 47–48, 60, 86), and the repeated invocation `python3 tests/test_phase_a.py` in Steps 1b/1d, 2b/2d, 3b/3d, 4b, 5b, 6b, 7–9, 10b, 11–13, 14a.

**Problem:** The test file lives at `tests/test_phase_a.py` and begins `from sim import ...` (plan line 60). The plan runs it via `python3 tests/test_phase_a.py` from the repo root (plan lines 47–48, 86). When Python runs a script, it puts the **script's own directory** (`tests/`) on `sys.path[0]`, **not** the current working directory. `sim.py` lives at the repo root, which is therefore not on the path. The import raises `ModuleNotFoundError: No module named 'sim'` before any test runs.

**Evidence (reproduced):**
```
$ cd /tmp/sptest && python3 tests/t.py        # t.py does `from mymod import hello`, mymod.py at root
Traceback (most recent call last):
  File "/private/tmp/sptest/tests/t.py", line 1, in <module>
    from mymod import hello
ModuleNotFoundError: No module named 'mymod'
exit=1
```
Also confirmed `tests/` does not yet exist (`ls tests/` → "NO tests dir"), so this is the first time the path is exercised — the bug will surface immediately at Step 1b.

**Why it breaks execution:** Every single TDD cycle in the plan ("run test to verify it fails" → "...verify it passes") depends on this command. It will fail with an import error, not a meaningful assertion failure, so the "verify it fails for the right reason" gate is meaningless and the executor is blocked at Step 1.

**Concrete fix (pick one, state it in the plan):**
- **Simplest:** keep the test at the repo root as `test_phase_a.py` and run `python3 test_phase_a.py`. `from sim import ...` then resolves because `sys.path[0]` is the repo root.
- **If keeping `tests/`:** prepend a path shim at the top of the file *before* the import:
  ```python
  import os, sys
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from sim import Config, generate_world, ...
  ```
- **Or run as a module from the repo root:** add `tests/__init__.py` and invoke `python3 -m tests.test_phase_a` (note: this requires the package form and changes the run command everywhere in the plan).

Whichever is chosen, update the run command in **all** steps consistently (Steps 1,2,3,4,5,6,7–9,10,11–13,14 all repeat the broken invocation).

### CR2. Step 3's `_run_campaign` signature change is never forwarded through `step()` — `channel` is silently dropped
**Plan section:** Step 3 (lines 236–244) changes the signature to `_run_campaign(self, target, spend, channel=0, craft=1.0)`. Step 6's test (plan lines 345–347) and Steps 7–9 pass `channel` inside campaign dicts to `env.step(...)`.

**Problem:** The live call site is `sim.py:282`:
```python
res = self._run_campaign(c.get("target", set()), spend)
```
It passes only `target` and `spend`. The plan changes the `_run_campaign` *definition* but **never updates this call site** to forward `channel=c.get("channel", 0)` (and `craft` where relevant). So when a policy or test puts `"channel": ...` in a campaign dict, `step()` discards it and the campaign always runs on channel 0.

**Evidence:** `sim.py:278–287` (current call site, verified) passes no `channel`; the plan's Step 3 section never mentions editing `step()`. Grep of the plan for the call-site forward returns nothing relevant.

**Why it breaks execution:** Step 6's acceptance test (plan line 345) explicitly sets `"channel": w.segments[0].channel_pref` and then asserts `base_after_acq > 0`. If `segments[0].channel_pref != 0`, the popular pain's users sit on a non-zero channel, the wrong-channel `channel_fit≈0.25` downweights reach, acquisition drops, and the `> 0` assert can fail intermittently per seed. More broadly, Steps 7–9 (NaivePolicy/Scripted/Oracle all carry a per-campaign `channel`) would be inert — the entire channel latent would be unexploitable through the normal `env.step` path even when `use_channels` is on, which would make the ablation gate's `+channels` row meaningless.

**Concrete fix:** In Step 3c, explicitly add: update `sim.py:282` to
```python
res = self._run_campaign(c.get("target", set()), spend,
                         channel=c.get("channel", 0), craft=c.get("craft", 1.0))
```
and add a test that a campaign dict with a non-default `channel` actually changes `tries` *through `env.step`* (not only through a direct `_run_campaign` call, which is all Step 3a currently tests).

### CR3. The Step 4 elasticity test is a tautology — it passes on unmodified v1 and cannot fail before implementation
**Plan section:** Step 4a (lines 264–270).

**Problem:** The test builds the solving feature, then asserts `_p_buy` at price 30 > `_p_buy` at price 90. v1's `_p_buy` already has `beta=2.0 > 0`, so higher price always lowers buy-prob **regardless of whether per-user elasticity is implemented**. The test passes against today's code.

**Evidence (reproduced on unmodified v1):**
```
v1 (no elasticity): low=0.5127 high=0.1361  low>high=True
=> Step 4a test PASSES on unmodified v1 code -> tautology, cannot fail before impl.
```

**Why it breaks execution:** The plan's own methodology (Step 4b "Run test" expecting failure, and TDD generally) requires the test to fail before the change and pass after. This one passes before and after, so it provides zero verification that per-user `beta_u` was actually wired in. An executor could skip Step 4c entirely and the test stays green.

**Concrete fix:** Make the assertion *specific to per-user elasticity*. E.g. construct (or stub) two users with very different `elasticity` under `Config.phase_a()` and assert that the **more elastic** user's `p_buy` drops *more* for the same price increase than the less elastic user's — a property that is false in v1 (where all users share `cfg.beta`) and only true once `beta_u = user.elasticity` is used. Equivalently, assert `_p_buy` differs from the v1 `cfg.beta` baseline for a user whose `elasticity != cfg.beta`.

### CR4. C2 prerequisite fix (scaling budget/horizon dials) is claimed but not implemented
**Plan section:** Goal (line 5: "scaling budget/horizon dials"), Step 1 title (line 52: "Config flags, **dials**, and `phase_a()` factory"), dependency table (line 38). Step 1c Config block: lines 90–121.

**Problem:** The design review's C2 fix is "make horizon and cash **explicit difficulty dials that scale with pool/channel/segment count**" so the larger experiment space (now pains × channels × price) stays solvable within budget. The plan advertises this in its Goal and Step-1 title, but the actual Step 1c Config additions contain **no `horizon` change, no `starting_cash` change, and no scaling logic**. The only new knobs are counts (`n_segments`, `n_channels`) and behavioral coefficients. `horizon=10` and `starting_cash=6000` (`sim.py:27–28`) are left exactly as v1.

**Evidence:** Grep of Step 1c (plan lines 90–121) for `horizon|cash|budget|spend` returns nothing (">>> NO horizon/cash/budget dial in Step 1c Config block <<<"). The C2 review issue (firmbench-v2-design-review-1.md, "C2. Experiment-space blowup vs an unchanged $6k / H=10 budget") is therefore not addressed.

**Why it matters for the chosen path:** The whole point of "build FULL Phase A, fix identifiability reactively" is that the ablation gate (Step 10) bisects which latent breaks `naive < scripted`. But with channels (×3 axis) and segments added on top of pains while the probe budget stays at $6k/H=10, the `+channels` and `full` rows can FAIL not because a latent is *unobservable* but because the experimenter ran out of money to probe pain×channel separately — a budget failure masquerading as an identifiability failure, which defeats the bisection tool. The reactive loop (plan lines 512–521) lists "ease its difficulty (smaller K, softer gate, larger `channel_fit_off`)" but conspicuously **not** "raise the budget," because the dial doesn't exist.

**Concrete fix:** Add to the Step 1c Config block the explicit dials the Goal already promises, e.g.:
```python
    horizon: int = 10            # already in v1; make it a stated difficulty dial
    starting_cash: float = 6000.0
```
and have `phase_a()` (or a `phase_a(scale_budget=True)` option) scale `horizon`/`starting_cash` with `n_channels`/`n_segments` (e.g. probe budget ∝ n_pains × n_channels). At minimum, add an acceptance check in Step 10/14 that distinguishes "FAIL because unobservable" from "FAIL because under-budget" (e.g. re-run the failing row with 2× cash; if it passes, it was a budget problem, not identifiability).

## Suggestions (nice to have)

- **S1 — `step()` must also forward `craft` for Phase C readiness, and `run.py`'s replay still lacks craft entirely.** `run.py:107` computes `purchases += resonance * p_buy` with no craft term, and `run.py:153–162`'s action_log records only `target`+`spend`. In pure Phase A (no NL artifacts) this is *consistent* (craft=1.0 everywhere, so no tripwire risk — verified), so it is not a blocker for Phase A. But Step 13 frames the run.py work as "stateful replay + oracle guard" and silently inherits the same craft omission env.py had; note explicitly that run.py's replay (and its action_log) will need the craft fix *when Phase C policies emit craft*, so it isn't forgotten later.

- **S2 — Step 3's `test_craft_scales_tries_live` hard-codes pain `{0}`, which may be empty under phase_a resampling.** Step 5's test correctly uses the max-popularity pain; Step 3a uses `{0}`. Under `_resample_users_from_segments`, pain 0 could draw ~0 users in some seed, making `tries==0` for both `craft=1.0` and `craft=0.5`, so `assert half < base` becomes `0 < 0` → False. Use the most-popular pain (as Step 5 does) instead of a literal `{0}`.

- **S3 — Doc inconsistency on the segment-block insertion point.** Step 2c prose says insert "after the v1 demography + names block" (plan line 169), but the Step 2c **code skeleton** (plan lines 169–180) inserts the segment block right after `pain_popularity`, i.e. *before* `_sample_names` (`sim.py:177`). This does not break flags-off reproducibility (verified — gating means no `rng.*` fires when off), but when `use_segments` is ON the resample draws shift the names RNG, so `phase_a()` names differ from v1 names. Harmless for correctness (no test asserts phase_a names == v1 names), but the prose and skeleton contradict each other; pick one location and make them agree.

- **S4 — Pool-exhaustion is the one latent threat to the proportional-holdout invariant; add it to the explicit invariant.** The scaling holds because base/recurring/churn_eff/new_subs are all linear in scale (verified: gap 0.0). The single way it breaks is if a full-scale campaign saturates its matching pool (`impressions >= pool size`) while the 0.2-scale holdout pool is not equally saturated (or vice versa) — then reach stops being proportional. In v1 today this never bites (verified: top-6 union pool ratio = 0.2009, impressions stay well under pool size even at $5k spend). The plan's "no absolute-count nonlinearity" invariant (lines 17–19) is *necessary*; recommend explicitly adding "...and no campaign may saturate the matching pool asymmetrically across the holdout split" so a future high-spend/low-pool config doesn't silently desync the grader.

- **S5 — Step 6 and Step 8 are oversized vs the "2–5 min" rubric; split them.** Step 6 (reset + step + recurring + per-segment churn + per-segment acquisition attribution + invariant guard + one-time-mode fallback) is ~6 distinct edits; Step 8 (round-0 multi-channel probe + feature attribution via bounce + price search via bounce_price + LTV-valued exploit) is effectively 4 sub-features in one. Recommend splitting Step 6 into 6a (base + recurring + churn) and 6b (segment-attributed acquisition), and Step 8 into 8a (channel discovery) and 8b (LTV exploit), each with its own failing test. As written they are not 2–5 min steps and bundle multiple things that can independently break the gate.

- **S6 — Per-seed vs mean wording in acceptance criteria.** Steps 1d/14b assert `python3 sim.py` "prints naive≈4035, scripted≈69783, oracle≈138570." Verified those are the **mean** row only; per-seed values differ widely (e.g. seed 1 oracle=140486, seed 2=140021). The criterion is correct if read as "the mean row is unchanged," but state "mean row" explicitly so an executor doesn't think per-seed must match those three numbers.

## Verified Claims (confirmed correct against the code)

- **Flags-off reproducibility / RNG preservation (plan Architecture lines 11–14, Step 2d).** Confirmed: with all Phase A flags off, `generate_world(3, Config())` is byte-identical across calls and contains no `segments` attribute. Because every new draw is behind `if cfg.use_segments:` (etc.), no `rng.*` call fires on the flags-off path, so inserting gated draws *anywhere* (start, middle, or end) cannot shift any v1 draw. The plan's claim that this reuses the same trick as `sim.py:174–178` is accurate — that comment and code exist exactly as cited.

- **Stateful-verifier proportional scaling survives compounding (plan Steps 12–13, Architecture line 19).** Walked a 3-round subscription example with `recurring = base*price`, `base = base*(1-churn_eff) + new_subs`, `new_subs ∝ int(spend*ipd)`: full profit 53000, holdout profit 10600, `reported*0.2 = 10600`, ratio 1.000000, gap 0.000000 (tripwire threshold 0.15). Because `churn_eff` depends only on price/quality (scale-free) and base/recurring/new_subs are linear in population scale, `reported*0.2 == holdout` exactly. The plan's "no absolute-count nonlinearity" invariant is sufficient (modulo S4's pool-saturation caveat).

- **The craft live/holdout bug and its line references.** Confirmed: `sim.py:252` live funnel hard-codes `p_try = resonance` (no craft), while `env.py:104` holdout replay uses `p_try = craft * resonance` — so an honest NL agent's live numbers undercount vs the craft-aware holdout, tripping the gap tripwire. The plan's fix (apply craft in the live funnel, Step 3c) targets the right line.

- **Cited line references are accurate.** `env.py:234` is exactly `result = _ENV._run_campaign(target, float(spend))` (the live call omitting craft, plan Step 11). `sim.py:174–178` is the names/keywords-after-demography RNG-preservation block (plan line 14). `sim.py:233` is the `_run_campaign` signature; `sim.py:228–231` is `_p_buy`; `run.py:54` is `replay_on_holdout` with proportional `spend * self.holdout_frac` at `run.py:85` and `purchases += resonance * p_buy` at `run.py:107`. All match.

- **v1 baseline numbers (plan lines 127, 496–499).** `python3 sim.py` prints the mean row naive **4035**, scripted **69783**, oracle **138570**, disc.eff **50%** — exactly as the plan claims (as means).

- **Dependency-table conflict serialization is correct.** Steps 3 & 5 both edit `_run_campaign`; Steps 4 & 5 both edit `_p_buy`. The plan serializes 3→4→5 (table line 40), which correctly avoids the same-function edit conflict.

- **Phase C is deferred, not silently dropped.** The plan references the Phase C lever once (line ~398, "would need a better spec — Phase C lever; for now record and prefer features with low bounce"), confirming Job 2 / judge-determinism are consciously out of scope for Phase A rather than forgotten. Consistent with the design review's C4/C5 deferral.

- **Soft quality gate fixes the C1 confound direction.** The plan replaces the design's hard quality gate with `gate = sigmoid(quality_gate_k * (ff - quality_bar))` plus separate `bounced_quality` / `bounced_price` buckets (Step 5c). This is the right shape to make the quality_bar latent separately observable from price/elasticity — the core C1 ask. (Whether the magnitudes actually clear the gate is exactly what Step 10's ablation gate is designed to discover, consistent with the chosen "fix reactively" path.)
