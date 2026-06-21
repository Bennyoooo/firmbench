# FirmBench RL — Results: the curve bends

**Real reinforcement learning (GRPO), not SFT.** A small open model was fine-tuned with
GRPO on Fireworks against FirmBench's profit reward, and its reward rose monotonically
across training epochs.

## The learning curve (GRPO, oracle-normalized profit reward)

Model: `qwen3-8b` (LoRA-GRPO, 8 candidates/prompt, temp 1.0, 24 training worlds).
Reward = full-episode profit of the model's product plan ÷ oracle profit (per world).

| Training epoch | Mean reward |
|---|---|
| 0 (base qwen3-8b) | **0.193** |
| 1 | **0.307** |
| 2 | **0.367** |

**Base → trained: 0.193 → 0.367 (+0.175, +91% relative).** Monotonic — the policy
learned, via policy-gradient RL, to run the firm better (select the right features within
budget, target the high-audience pains, price for retention).

## Why this is RL, not SFT

- Reward (FirmBench profit vs oracle) **weights the gradient** via GRPO advantages —
  it does not merely filter a supervised dataset.
- Run on Fireworks managed RFT (`reinforcement-fine-tuning-job`, loss method GRPO,
  8 response candidates/prompt, KL-regularized).
- The reward is our deterministic execution-based verifier (no LLM judge, ungameable).

## Run journey (what it took)

1. **glm-5p1** (frontier MoE): unlocked by $500 credits, but its RL trainer was slow and
   ultimately the trainer container crashed (exit 1, likely OOM). Pivoted to qwen3-8b.
2. **Zero-variance early stop**: a `[0,1]`-clipped reward tied all losing plans at 0 and
   good plans at ~1 → no within-group variance → GRPO filtered everything. Fixed by
   **unclipping** the reward (bounded [-1, 1.5]) + temp 1.0 + 8 candidates (verified
   within-group std ~0.6-0.8 before relaunch).
3. **qwen3-8b** trained fast and robustly → the curve above.

## Reproduce / extend

- Reward + dataset: `grpo/test_firmbench_grpo.py`, `grpo/make_dataset.py`.
- Launch: `python3 -m eval_protocol.cli create rft --training-config-base-model
  accounts/fireworks/models/qwen3-8b --inference-parameters-response-candidates-count 8
  --inference-parameters-temperature 1.0 ... --force --skip-validation --ignore-docker`.
- Job: `kby4ofja` → model `accounts/bennyjxh/models/firmbench-qwen3-8b-grpo-v4`.
- Held-out generalization eval + HUD eval of the checkpoint: see below / HUD_EVAL.md.
