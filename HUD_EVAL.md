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

## Eval the FINE-TUNED checkpoint

Identical command, just swap `--model` to the GRPO output model:

```bash
python3 -m hud eval tasks.py openai_compatible \
  --model accounts/bennyjxh/models/firmbench-glm5p1-grpo-v1 \
  --config base_url=https://api.fireworks.ai/inference/v1 \
  --task-ids market_discovery_seed42,market_discovery_seed123,market_discovery_seed7 \
  --max-steps 80 --group-size 1 -y
```

(glm-5p1 supports serverless LoRA serving, so the fine-tuned adapter is callable by id
with no dedicated deployment.)

## Notes

- `tasks.py` defines the **multi-turn** market-discovery task (6 MCP tools). The GRPO
  fine-tune in `grpo/` was trained on the **single-turn strategy** formulation, so its
  *learning curve* is measured by `grpo/base_eval.py` on held-out seeds (the training
  objective). HUD eval here is the platform/leaderboard surface and an
  out-of-distribution transfer check for the checkpoint.
- `.hud_eval.toml` carries these defaults so flags can be omitted.
- Results appear at the printed `hud.ai/jobs/<id>` link.
