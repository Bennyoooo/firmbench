# FirmBench GRPO run — live notes

## Account / auth
- Key: bennyjxh account (fw_74b...gM5V), $500 credits (unlocks B200 training).
- firectl: `~/.local/bin/firectl`, auth via `--api-key <key> --account-id bennyjxh`.

## Task (eval-protocol, isolated in grpo/)
- Single-turn STRATEGY reward: given market intelligence (audiences + solves mapping +
  economics), model outputs a product plan {build:[feats], price, target:[pains]}.
  Reward = oracle-normalized full-episode profit of executing that plan.
- Economy: `Config(starting_cash=3500, build_cost=500)` → build-all bankrupts; must
  select top-audience features within budget + price well.
- Datasets: `firmbench_prompts.jsonl` (train seeds 1-24), `firmbench_eval.jsonl`
  (held-out seeds 100-115).
- Base glm-5p1 reward (train, K=4): **0.653** mean, std 0.38, intra-prompt std 0.32.

## Training job
- RFT job id: **emjcrkg5** (GRPO, glm-5p1, lora_rank 16, epochs 4, max_ctx 8192,
  inference max_output_tokens 4096, temp 0.7).
- Output model: **accounts/bennyjxh/models/firmbench-glm5p1-grpo-v1**
- Evaluator: accounts/bennyjxh/evaluators/test-firmbench-grpo-test-firmbench-grpo
- Launched 2026-06-21 04:55 UTC.

## Monitoring
- `firectl get reinforcement-fine-tuning-job emjcrkg5 --api-key <k> --account-id bennyjxh`
- Eval a model: `python3 base_eval.py <model_id> <K> firmbench_eval.jsonl`

## Curve (held-out mean reward)
- base glm-5p1 (held-out, K=4): **0.533** (std 0.43, intra 0.34); train: 0.653
- trained: (after job completes)
