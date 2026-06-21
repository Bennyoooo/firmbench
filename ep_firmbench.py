"""FirmBench RL reward for Fireworks GRPO via eval-protocol (managed RFT).

Quick-test / cost-probe version: SINGLE-TURN. Each dataset row is a round-0 market
observation; the model emits one JSON action; the reward is the normalized one-step
profit of that action run through the deterministic sim. This is a real FirmBench
signal with a minimal surface, used to gauge qwen3-8b GRPO cost/quota before scaling
to the full multi-turn env (and to qwen3-32b).

Launch (dry-run prints the planned job, no charge):
    export FIREWORKS_API_KEY=...  FIREWORKS_ACCOUNT_ID=bennyjxh
    python3 -m eval_protocol.cli create rft --entry ep_firmbench.py::firmbench_round_profit \
        --training-config-base-model accounts/fireworks/models/qwen3-8b \
        --training-config-lora-rank 8 --loss-config-method grpo \
        --training-config-epochs 1 --max-concurrent-rollouts 4 --dry-run --skip-validation
"""

from eval_protocol.models import EvaluationRow, EvaluateResult, Message
from eval_protocol.pytest import evaluation_test

from sim import Config, generate_world, FirmEnv
from agent import system_prompt, format_obs, extract_json, validate_action

cfg = Config()
SEEDS = list(range(1, 9))  # 8 training worlds


def _dataset():
    rows = []
    for s in SEEDS:
        world = generate_world(s, cfg)
        env = FirmEnv(world)
        obs = env.reset()
        rows.append(EvaluationRow(
            messages=[
                Message(role="system", content=system_prompt(cfg)),
                Message(role="user", content=format_obs(obs, cfg, [])),
            ],
            ground_truth={"seed": s},
        ))
    return rows


@evaluation_test(
    input_rows=[_dataset()],
    completion_params=[{
        "model": "accounts/fireworks/models/qwen3-8b",
        "temperature": 0.7,
        "max_tokens": 2048,
    }],
    mode="pointwise",
)
def firmbench_round_profit(row: EvaluationRow) -> EvaluationRow:
    """Reward = normalized one-step profit of the model's action on its world."""
    seed = (row.ground_truth or {}).get("seed", 1)
    world = generate_world(seed, cfg)
    env = FirmEnv(world)
    env.reset()
    text = (row.messages[-1].content or "") if row.messages else ""
    action = validate_action(extract_json(text), cfg)
    _obs, profit, _done, _ = env.step(action)
    # round-0 profit lives roughly in [-600, +600]; map to [0, 1]
    score = max(0.0, min(1.0, (profit + 600.0) / 1200.0))
    row.evaluation_result = EvaluateResult(score=score, reason=f"round-0 profit ${profit:.0f}")
    return row
