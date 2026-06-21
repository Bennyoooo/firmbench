# Phase D — Multi-Agent Run Log & Status

The multi-agent layer (Builder / Marketer / Pricer / Coordinator with partial observability
+ a shared blackboard + a **coordination tax**) and its shared-policy role-conditioned RL.
This file records what ran — mirrors `RFT_RUN.md` for the single-agent pipeline.

Branch: `phase-d-multiagent`. Built from `docs/plans/phase-d-multiagent-build-spec.md`.

---

## What ran offline (real, no keys — the validated proxy)

Per the build discipline (single-agent byte-identical; gates + selftests green before any
real run), everything below runs with **pure stdlib + the local sim** — no network, no keys.

| Check | Command | Result |
|---|---|---|
| Single-agent **byte-identical** | `python3 sim.py` · `ablation_gate()` | unchanged (v1 disc.eff ~50%; `+channels` WARN; full PASS) |
| **Coordination gate** | `python3 multiagent.py` | `naive_team < scripted_team < oracle`, scripted tax < naive tax → **PASS** |
| Team head-to-head (10 seeds) | `python3 run.py --multiagent` | naive_team **0.016** · scripted_team **0.047** · single-agent scripted **0.066** · oracle **1.000**; messages buy **+3.1%** of the oracle |
| `OracleTeam` == oracle | (test) | team disc.eff **1.000** — the protocol reaches the ceiling when perfectly coordinated |
| **Team RFT selftest** (SFT) | `python3 rft.py --multiagent --selftest --iterations 3` | curve bends **0.45 → 1.00**; 512 role-turns/iter, balanced 128/role |
| **Team GRPO selftest** (on-policy) | `python3 rft_hud.py --multiagent --selftest` | curve bends **0.44 → 1.00** toward the oracle ceiling |
| HUD env wiring (offline) | `python3 env_multiagent.py` | delegate-tool round protocol grades a team episode (seed42 reward 0.075) |
| Tests | `python3 tests/test_multiagent.py` · `tests/test_rft_hud.py` · `tests/test_phase_a.py` | **36 pass** (10 multiagent + 8 rft_hud incl. 3 team + 18 phase_a) |

**Coordination tax (the headline).** `tax = oracle − team_profit`. `ScriptedTeam` (reads the
blackboard) and `NaiveTeam` (ignores it) share identical role policies, so the gap is purely
the value of the messages. Full market, 10 held-out seeds:

```
policy                       mean profit  disc.eff   coord tax
oracle (full-info)               691176     1.000       0.0%
single-agent scripted             46105     0.066      93.4%
scripted-team (comms)             31836     0.047      95.3%
naive-team (no comms)             10676     0.016      98.4%
VERDICT: coordination (blackboard messages) buys +3.1% of the oracle. gate -> PASS
```

---

## Real HUD eval — multi-agent task (Coordinator-dispatch, via the HUD Gateway)

`HUD_API_KEY` is configured (`~/.hud/.env`), so the team task runs through the HUD gateway —
no provider key needed. The env is served locally (MCP) and the agent drives the four roles
via the delegate tools.

```bash
hud eval env_multiagent.py claude \
  --task-ids multiagent_market_discovery_seed42 \
  -y --auto-respond --max-steps 100 --gateway -v
```

**Run (claude-sonnet-4-6, 3 seeds, via the HUD gateway).** The MCP env served locally;
Claude correctly executed the round protocol every round — `get_team_state →
coordinator_set_budget → delegate_build → delegate_price → delegate_campaigns → end_round` —
through all 16 rounds of each episode (it even self-summarized "🏆 GAME COMPLETE").

| Seed | Claude (team disc.eff) | naive_team | scripted_team | oracle |
|---|---|---|---|---|
| 42 | **0.036** | 0.005 | 0.075 | 1.000 |
| 123 | **0.003** | 0.009 | 0.043 | 1.000 |
| 7 | **0.009** | 0.005 | 0.023 | 1.000 |
| **mean** | **0.016** | 0.006 | 0.047 | 1.000 |

Jobs: [seed 42](https://hud.ai/jobs/51adff8e87b24f568f81bc13cee18b40) ·
[seeds 123+7](https://hud.ai/jobs/f73408e4e2de4816a11849af7276e96e). (Per-seed 123/7 read in
task-id order from the HUD details table; the 3-seed mean is order-independent.)

**Read:** an untrained frontier model, driving the four roles via the delegate tools, lands
at the **no-coordination floor** (≈ the naive-team 0.016) — well below the disciplined
scripted team (0.047) and far below the LTV oracle (1.000). The multi-agent LTV game
(coordinate who-builds-what, acquire-then-coast, price for retention) is exactly the skill
RL has room to teach. This run also validates the **real HUD pipeline end-to-end** (gateway
auth → local MCP env → tool round-protocol → team disc.eff grade).

(A fixed env-resolution bug first surfaced here: a task-only file resolves its env to the
sibling `env.py` — the single-agent env — so the multi-agent env is served by pointing
`hud eval` at the self-contained `env_multiagent.py`.)

---

## Real training — exact launch commands (credential-gated)

### A. Shared-policy team RFT on Fireworks (`rft.py --multiagent --run`)

Blocked on the **same** Tier-1 training-credit gate as the single-agent run (see `RFT_RUN.md`:
glm-5p1 needs B200/B300 quota; add $50 to unlock). The pipeline is wired end-to-end —
role-conditioned rollouts → grade by team disc.eff → flatten **all** role-turns of the
winning episode per world → SFT one shared model on the pooled role-turns → re-eval.

```bash
export FIREWORKS_API_KEY=...        # + FIREWORKS_ACCOUNT, firectl on PATH
python3 rft.py --multiagent --run --iterations 2 --rollouts 4 \
  --train-seeds 16 --eval-seeds 8 --model accounts/fireworks/models/glm-5p1
# offline proof (no key): python3 rft.py --multiagent --selftest --iterations 3
```

### B. Shared-policy team RL through HUD (`rft_hud.py --multiagent --run`)

Needs a **forked trainable model** (one-time): `hud models fork <base-model>` → a gateway
slug whose weights advance in place. Each `hud.Run` is one team-episode (the Coordinator
agent driving the roles); reward = team disc.eff; grouped GRPO step promotes the shared
checkpoint.

```bash
hud login                                   # HUD_API_KEY already set here
hud models fork <base-model>                # -> accounts/<team>/models/<forked-trainable>
python3 rft_hud.py --multiagent --run \
  --model accounts/<team>/models/<forked-trainable> \
  --steps 5 --group-size 8 --train-seeds 8 --eval-seeds 8
# offline proof (no fork): python3 rft_hud.py --multiagent --selftest
```

---

## The checkpoint question (why ONE model, not four)

Train **one shared, role-conditioned checkpoint** (parameter sharing): each role-agent is the
same model + a role system prompt (`tasks.ROLE_PROMPTS`) + its role-sliced observation. A
team-episode yields role-turns (Coord/Builder/Pricer/Marketer × rounds), each a
`(prompt, completion)`; **every role-turn shares the team's episode disc.eff** (cooperative
reward; GRPO normalizes within a world's group). Per-role checkpoints (MAPPO/CTDE) are 4× the
fine-tuning cost for cooperative LLM roles — documented as the stretch path, not built.

---

## REAL runs executed (2026-06-21, with the provided Fireworks key)

### glm-5p1 SFT on Fireworks — FINISHED ✅
Job `e1ek8liv` (base `glm-5p1`, dataset `firmbench-expert-v1` = 240 expert turns, 3 epochs,
LoRA-16, max_ctx 8192) → output model **`accounts/bennyjxh/models/firmbench-rft-glm-v1`**,
state **READY**. This is the run `RFT_RUN.md` was blocked on (the Tier-1 training-credit gate
is resolved). One validation fix was needed: glm-5p1 requires `--max-context-length`
divisible by 16 (the unset default isn't).

**Serving caveat (corrected):** the trained LoRA addon returns HTTP 404 ("not deployed") on
serverless inference — on this account LoRA addons are NOT serverless-served (only base
models are), so evaluating the tuned model needs a **dedicated** deployment
(`firectl deploy <addon> --deployment <dep>`; glm-5p1 = B200, paid — not spun up unprompted).
Base glm-5p1 evaluates to disc.eff **0.000** (real: loses money on held-out seeds); tuned is
**not evaluated** pending a deployment. The SFT half of the loop is proven; serving isn't
serverless here. See `RFT_RUN.md` for the deploy commands.

### HUD on-policy RL — PIPELINE VALIDATED end-to-end ✅ (cheap model: Qwen3-8B)
Forked a cheap trainable model `firmbench-team-rl-q8b` (Qwen/Qwen3-8B, $0.13/$0.40 per Mtok)
and ran both backends through the **real** HUD gateway + `hud.train`:

| Run | steps | result | checkpoints |
|---|---|---|---|
| team (`--multiagent`, env_multiagent) | 2 | disc.eff 0.000 → 0.000 | promoted (e3afe255, b179e74a) |
| single-agent (env.py) | 2 | disc.eff 0.000 → 0.000 | promoted (c439d9ac, 45f2ab3f) |

The loop runs end-to-end on real infra: **rollout → `forward_backward` → `optim_step`
(promotes a new checkpoint) → re-eval**. The curve is flat at 0 because Qwen3-8B scores ~0
on this hard Phase A market (even Claude Sonnet 4.6 gets only 0.016 team / ~0 single) — no
reward signal → GRPO advantages are all zero → nothing to learn in 2 steps. This is a
model-strength × env-difficulty result, not a pipeline bug; the offline selftest (mock
imitating `OracleTeam`) proves the same loop **bends 0.44 → 1.00** when reward signal exists.

**Bug fixed to make `--run` work at all:** `HudBackend._agent` now passes
`extra_body={"return_token_ids": True}` so the openai_compatible agent records per-turn token
ids + sampling logprobs. Without it, `forward_backward` 400s with *"no trainable turns"* — the
real `--run` path had never been exercised before (only `--selftest`).

**To get a visible real bend:** warm-start RL from a competent base (e.g. the expert-SFT'd
glm above, or an expert-SFT'd small model) so rollouts score > 0, then RL has signal; or run
many more steps with a stronger base. The harness is ready (`--run` works); it just needs a
policy that gets traction on the env.

### Replay / visualization
`env_multiagent.py` now records the per-round **blackboard** (role messages), Coordinator
budget, and campaign results into `artifacts_multiagent/<seed>/manifest.json`.
**`replay_multiagent.html`** is a team replay viewer: step through rounds to watch the
blackboard (role-colored Coordinator/Builder/Pricer/Marketer messages), role actions,
campaign results, per-round profit, and a coordination-tax (team vs oracle) bar. Serve with
`python3 -m http.server 8000` and open `replay_multiagent.html` (run `python3 env_multiagent.py`
first to generate manifests).
