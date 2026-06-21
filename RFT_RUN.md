# Real RFT Run — Log & Status

Rejection-sampling fine-tuning (expert iteration / STaR) of a small open model on
Fireworks, driven by `rft.py`. This file records the real run on `glm-5p1`.

## What ran (real, on Fireworks)

| Step | Result |
|------|--------|
| Model selection | `accounts/fireworks/models/glm-5p1` — the **only** current serverless model that supports the full cheap loop: serverless inference + supervised LoRA tuning + serverless LoRA serving. (deepseek-v4-flash/nemotron aren't V1-tunable; kimi/glm need B200; only glm-5p1 serves LoRA serverlessly.) |
| Base eval (4 held-out worlds) | **mean reward −409.6**, flagged 1/4. The base model loses money — the floor we want to bend up. |
| Rollouts (3 × 8 worlds, temp 0.7, parallel) | 24 episodes graded by the held-out verifier. **Every rollout was negative** (best per world −240…−356, `pos=0`). |
| Rejection sampling | Cold start: no winning trajectories to learn from (consistent with the README leaderboard — frontier models score ~0 on this env). |
| Expert bootstrap dataset | 240 chat turns from `ScriptedExperimenter` (earns ~21k, runs locally/free) → `rft_out/sft_expert.jsonl`. Standard RFT cold-start fix: SFT on demonstrations first, then later iterations rejection-sample the model's own (now profitable) rollouts. |
| Dataset upload (`firectl create dataset`) | ✅ `firmbench-expert-v1` — **READY**, 710 KiB. |
| SFT job (`firectl create supervised-fine-tuning-job`) | ✅ **FINISHED 2026-06-21** (was blocked on billing; now unblocked). |

## ⚠️ SFT vs RL — read this first

This file documents **rejection-sampling SFT** (expert-bootstrap supervised fine-tuning).
When asked to "finish a training on glm" I first completed that *supervised* job (below) —
but the actual ask was **RL**. The correct RL run is a **GRPO reinforcement fine-tuning**
job, launched 2026-06-21 and mirroring the qwen3-8b GRPO setup:

- **Job `veu077gh`** → output **`accounts/bennyjxh/models/firmbench-glm-grpo-v1`**,
  **Loss: GRPO**, base glm-5p1, 3 epochs, LoRA-8, group size 8 (response candidates),
  evaluator `test-firmbench-grpo` + dataset `…-20260621070244` (copied from the qwen3-8b
  GRPO job `kby4ofja`). Launch (native firectl, avoids the eval-protocol 0.3.31 CLI bug
  where `--loss-config-method` rejects every value):
  ```bash
  firectl create reinforcement-fine-tuning-job \
    --base-model accounts/fireworks/models/glm-5p1 \
    --dataset test-firmbench-grpo-test-firmbench-grpo-dataset-20260621070244 \
    --evaluator accounts/bennyjxh/evaluators/test-firmbench-grpo-test-firmbench-grpo \
    --output-model firmbench-glm-grpo-v1 --rl-loss-method grpo \
    --epochs 3 --lora-rank 8 --max-context-length 8192
  ```
  Monitor: `firectl get reinforcement-fine-tuning-job veu077gh`. The GRPO reward curve (in
  the job status / Fireworks dashboard) is the RL signal — no serving needed to see it.
- The evaluator reward is currently the **single-turn round-0 profit** (`ep_firmbench.py`,
  the cost-probe); scaling to the full multi-turn episode reward is the next step.

The supervised run below still completed and is kept for the record.

## ✅ FINISHED (2026-06-21) — the SFT run completed

Training credits were added, so the SFT job ran to completion:

- **Job** `e1ek8liv` → output model **`accounts/bennyjxh/models/firmbench-rft-glm-v1`**,
  state **READY** (serverless LoRA serving). Base `glm-5p1`, dataset `firmbench-expert-v1`,
  3 epochs, LoRA-16.
- **One fix needed:** glm-5p1's SFT requires `--max-context-length` divisible by 16
  (the unset default is not) — pass e.g. `--max-context-length 8192`.

### Serving gotcha — LoRA addons are NOT serverless on this account
The eval first returned base **0.000** → tuned **0.000**, but the tuned number was an
artifact: calling `accounts/bennyjxh/models/firmbench-rft-glm-v1` (and the earlier
`probe-deepseek-r1-distill-qwen-1p5b`) returns **HTTP 404 "Model not found, inaccessible,
and/or not deployed"**. So `RFT_RUN.md`'s original premise — *glm-5p1 supports serverless
LoRA serving* — does **not** hold here: only the **base** model serves serverlessly; the
fine-tuned **LoRA addon** must be loaded onto a **dedicated deployment** to be callable
(`firectl deploy <addon> --deployment <dep>`; glm-5p1 = B200/B300 dedicated, paid). The
agent's API errors are caught → no-op action → $0 profit → disc.eff 0, which is why the
tuned eval read 0.000.

- **base glm-5p1: 0.000** is a *real* number (the base serves; it loses money on held-out
  seeds, consistent with the earlier −409.6 raw reward → clips to 0).
- **tuned: not evaluated** — needs a dedicated deployment (not spun up here: a glm-5p1 B200
  deployment is costly and was not authorized).

**To actually eval the tuned model** (incurs dedicated-GPU cost):
```bash
firectl create deployment --model accounts/fireworks/models/glm-5p1 --accelerator-type NVIDIA_B200 ...
firectl deploy firmbench-rft-glm-v1 --deployment <deployment-id> --wait
# then re-run rft_out/glm_eval.json's eval against the deployed addon
```

The training half of the cheap loop (data → SFT → READY adapter) is proven end-to-end; the
serving half needs a dedicated deployment (serverless addons aren't available on this tier).

The historical blocker write-up below is kept for the record.

## The one blocker: training credits (HISTORICAL — resolved)

```
ResourceExhausted: B200/B300 training requires a Tier 2 account or higher.
Add $50 in credits to unlock training quota automatically.
```

The account (`bennyjxh`, created 2026-06-20) is Tier 1. Quota check confirms
`training-b200-count = 0`, `training-b300-count = 0` — and glm-5p1 (a large MoE)
requires B200/B300. The H100/H200 training quota we *do* have doesn't match these
large models, and the only serverless-LoRA-servable model is glm-5p1.

**Unblock (≈2 min, ~$50):** add $50 credits at
https://app.fireworks.ai/account/billing → training quota unlocks automatically.

## Finish the run after credits are added

```bash
export FIREWORKS_API_KEY=...      # the account key
export FIREWORKS_ACCOUNT=bennyjxh
export PATH="$HOME/.local/bin:$PATH"   # firectl

# Dataset is already uploaded (firmbench-expert-v1, READY). Just launch + serve + eval:
firectl create supervised-fine-tuning-job \
  --base-model accounts/fireworks/models/glm-5p1 \
  --dataset firmbench-expert-v1 --output-model firmbench-rft-v1 \
  --epochs 3 --lora-rank 16 \
  --api-key $FIREWORKS_API_KEY --account-id bennyjxh

# When firectl get model firmbench-rft-v1 shows READY, eval the tuned model vs base:
FIREWORKS_MODEL=accounts/bennyjxh/models/firmbench-rft-v1 \
  python3 -c "from sim import Config; from run import Verifier; from rft import make_llm_agent, evaluate; \
  cfg=Config(); print(evaluate(list(range(100,104)), make_llm_agent('accounts/bennyjxh/models/firmbench-rft-v1',0.0), cfg, Verifier()))"
# Compare to base mean reward −409.6 → the curve bends.
```

Or end-to-end (rollout → SFT → eval, multi-iteration) once quota is live:
`python3 rft.py --run --iterations 2 --rollouts 4`

## Verified offline (no credits needed)

`python3 rft.py --selftest --iterations 3` exercises the entire machinery
(rollout → verifier-grade → reject → dataset → eval) with a mock model and bends
the curve: **−734 → 15,524** mean held-out reward, flags 5/8 → 0/8.
