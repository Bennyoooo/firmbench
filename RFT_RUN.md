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
| SFT job (`firectl create supervised-fine-tuning-job`) | ❌ **Blocked on billing** (see below). |

## The one blocker: training credits

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
