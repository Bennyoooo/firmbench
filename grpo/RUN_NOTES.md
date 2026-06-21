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

## Progress observations (05:22 UTC)
- Job healthy, slow: epoch-0 baseline eval running 8 parallel runs x 24 rollouts
  (group size 8), ~7-15s/rollout (glm-5p1 reasoning at 4096 tokens). ~30 min/epoch-eval
  + training rollouts -> full 4-epoch run likely 3-5 hrs.
- jobProgress.percent (6%) and token counters lag (don't reflect eval phase) — true
  progress is in the per-epoch streamlogs / epoch_to_evaluation_output.
- monitor.sh (PID logged) polls every 10 min -> monitor.log; auto-runs held-out eval
  on the output model when training completes. job_final.json saved on completion.
- The per-epoch eval rewards (epoch 0..3) form the learning curve; base held-out 0.533.

## RESTART -> leaner v2 (05:37 UTC)
- v1 (emjcrkg5) killed: pathologically slow (110min, epoch-0 baseline eval not done,
  individual rollouts spiking to ~15min). Heavy: group 8 + 4096 tokens + eval-every-epoch.
- v2 job: **qxcfrbww**, output **accounts/bennyjxh/models/firmbench-glm5p1-grpo-v2**
  GRPO, lora16, epochs 3, candidates 4 (was 8), max_output_tokens 3072 (was 4096),
  max_concurrent_rollouts 16 (was 8). Expect ~3-4x faster.

## v3 = the good run (06:34 UTC)  job ymrer252 -> firmbench-glm5p1-grpo-v3
- v2 (qxcfrbww) EARLY_STOPPED: "zero variance in scores" — clipped [0,1] reward tied
  losing plans at 0 / good at ~1; 4 candidates too few. Fix: UNCLIP reward (bounded
  [-1,1.5]) + temp 1.0 + 8 candidates. Verified locally: within-group std 0.6-0.8.
- Also fixed: pytest discovery hung importing monitor.py (module-level while True) and
  base_eval.py (module-level exec). Guarded base_eval; moved monitor out of grpo/
  (-> ../grpo_monitor.py). Committed reward change so eval-protocol re-uploaded (--force
  only re-uploads on a new git hash).
- v3 PASSED the variance check -> training (Epoch 1/3). curves field will fill with
  per-epoch Score = the learning curve. Monitor: ../grpo_monitor.py (PID logged).
