# Running `hud eval` with the Fireworks fine-tuned checkpoint

The FirmBench HUD environment (`env.py` / `tasks.py`) can be evaluated by **any
Fireworks-served model** — base or our GRPO fine-tune — through HUD's
`openai_compatible` agent pointed at the Fireworks inference API. **Verified working**:
`hud eval` runs the MCP env locally and drives the Fireworks model, which calls the
tools (no Docker needed).

## One-time setup

```bash
python3 -m pip install hud-python                 # hud 0.6.6
export OPENAI_API_KEY=$FIREWORKS_API_KEY          # HUD's openai_compatible agent reads this
export OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
```

## Eval the BASE model

```bash
python3 -m hud eval tasks.py openai_compatible \
  --model accounts/fireworks/models/glm-5p1 \
  --config base_url=https://api.fireworks.ai/inference/v1 \
  --task-ids market_discovery_seed42 --max-steps 80 -y
```

## Eval the FINE-TUNED checkpoint (GRPO model)

The trained model is `accounts/bennyjxh/models/firmbench-qwen3-8b-grpo-v4` (LoRA-GRPO on
qwen3-8b). qwen3-8b is **not serverless**, so it is served via a dedicated LoRA
deployment (`f81fmqll`, scale-to-zero) and addressed with the `#deployment` routing
suffix. **Verified**: `hud eval` drives this checkpoint against `env.py` end-to-end.

```bash
TUNED="accounts/bennyjxh/models/firmbench-qwen3-8b-grpo-v4#accounts/bennyjxh/deployments/f81fmqll"
python3 -m hud eval tasks.py openai_compatible \
  --model "$TUNED" \
  --config base_url=https://api.fireworks.ai/inference/v1 \
  --task-ids market_discovery_seed42 --max-steps 80 -y
```

The deployment is scale-to-zero: the first request after idle has a ~1-2 min cold start,
then runs; it costs nothing while idle.

> Note: this checkpoint was trained on the **single-turn strategy** task (see
> `grpo/RESULTS.md`), where its reward improved base 0.147 → 0.529 on held-out worlds.
> `tasks.py` is the **multi-turn** discovery env, so HUD eval here is an
> out-of-distribution transfer check (the integration/plumbing is the deliverable; the
> learning evidence is the strategy-task curve in RESULTS.md).

## Notes

- `tasks.py` defines the **multi-turn** market-discovery task (6 MCP tools). The GRPO
  fine-tune in `grpo/` was trained on the **single-turn strategy** formulation, so its
  *learning curve* is measured by `grpo/base_eval.py` on held-out seeds (the training
  objective). HUD eval here is the platform/leaderboard surface and an
  out-of-distribution transfer check for the checkpoint.
- `.hud_eval.toml` carries these defaults so flags can be omitted.
- Results appear at the printed `hud.ai/jobs/<id>` link.
