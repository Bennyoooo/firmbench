"""FirmBench RL reward for Fireworks GRPO via eval-protocol (managed RFT).

TWO rewards here:

1. firmbench_episode  (PRIMARY, multi-turn) — the real FirmBench objective. Each rollout is a
   FULL interactive episode through the MCP-Gym env (firmbench_mcp/): the model calls
   `firm_round` each round, sees per-campaign diagnostics, and adapts (probe -> discover ->
   exploit). Reward = disc.eff = profit / oracle, delivered densely as per-round
   profit/oracle (sums to disc.eff). A single completion CAN'T express FirmBench — round 0
   has no demand signal — so the reward must be interactive/multi-turn.

2. firmbench_round_profit  (LEGACY, single-turn) — the original round-0 cost-probe (one
   action, one-step profit). Kept for reference / cheap quota checks. This is what the first
   qwen3-8b and glm GRPO runs used; firmbench_episode is the upgrade.

Local check (env + reward, no model):
    python3 firmbench_mcp/firmbench_adapter.py is wrapped; see the validation in the repo.

Launch multi-turn RFT (GRPO) — provide the MCP-Gym server so Fireworks drives the episode:
    export FIREWORKS_API_KEY=...  FIREWORKS_ACCOUNT_ID=bennyjxh
    python3 -m eval_protocol.cli create rft --entry ep_firmbench.py::firmbench_episode \
        --training-config-base-model accounts/fireworks/models/glm-5p1 \
        --training-config-max-context-length 8192 --training-config-lora-rank 8 \
        --training-config-epochs 1 --max-concurrent-rollouts 4
  (or, native firectl with --mcp-server firmbench_mcp/server.py once the evaluator is uploaded;
   see PHASE_D_RUN.md for the firectl reinforcement-fine-tuning-job recipe.)
"""

import os

from eval_protocol.models import EvaluationRow, EvaluateResult, InputMetadata, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor

from sim import Config, generate_world, FirmEnv
from agent import system_prompt, format_obs, extract_json, validate_action

cfg = Config()
SEEDS = list(range(1, 1 + int(os.environ.get("EP_SEEDS", "8"))))  # training worlds (held-out evals use 100+)
# Local rollout model (litellm). Override with EP_MODEL for a local smoke test; the Fireworks
# RFT base model is set separately via --training-config-base-model and is unaffected by this.
_LOCAL_MODEL = os.environ.get("EP_MODEL", "fireworks_ai/accounts/fireworks/models/qwen3-8b")

# Multi-turn system prompt: the model drives the firm via the firm_round MCP tool.
_MCP_SYSTEM_PROMPT = system_prompt(cfg) + """

YOU INTERACT VIA ONE TOOL: firm_round(action_json). Call it once per round. `action_json` is a
JSON object: {"build": feature_id or null, "price": number, "campaigns": [{"target": [pain_ids],
"spend": dollars, "channel": 0-2}]}. The tool returns the next observation (per-campaign
audience / tries / purchases / revenue). Round 0: probe pains cheaply ($10 each) to read demand.
Then build for the biggest-audience pains, find which pain each feature solves (purchases>0), and
exploit. Keep calling firm_round until the episode ends (cash must stay >= 0)."""


# ----------------------------- PRIMARY: multi-turn episode reward -----------------------------

def _episode_dataset():
    """One row per training world; the MCP-Gym server is seeded from environment_context."""
    rows = []
    for s in SEEDS:
        rows.append(EvaluationRow(
            messages=[Message(role="system", content=_MCP_SYSTEM_PROMPT)],
            input_metadata=InputMetadata(
                row_id=f"firmbench-seed-{s}",
                dataset_info={
                    "environment_context": {"game": "FirmBench", "seed": s},
                    # formatted as template.format(observation=..., **environment_context),
                    # so only {observation}/{game}/{seed} are valid keys (NOT {round}).
                    "user_prompt_template": (
                        "Latest observation:\n{observation}\n"
                        "Call firm_round(action_json) with your next action."
                    ),
                },
            ),
        ))
    return rows


@evaluation_test(
    input_rows=[_episode_dataset()],
    completion_params=[{
        # local rollout routes via litellm -> needs the provider prefix; the Fireworks RFT
        # base model is set separately via --training-config-base-model.
        "model": _LOCAL_MODEL,
        "temperature": 0.7,
        "max_tokens": 2048,
    }],
    rollout_processor=MCPGymRolloutProcessor(),
    server_script_path="firmbench_mcp/server.py",
    mode="pointwise",
    max_concurrent_rollouts=4,
)
def firmbench_episode(row: EvaluationRow) -> EvaluationRow:
    """Reward = full-episode disc.eff. The MCP-Gym env returns per-round profit/oracle; their
    sum (row.get_total_reward()) is disc.eff. Clip to [0,1] for a clean GRPO reward."""
    disc_eff = max(0.0, min(1.0, row.get_total_reward()))
    row.evaluation_result = EvaluateResult(
        score=disc_eff, reason=f"episode disc.eff = {disc_eff:.3f} (profit/oracle)")
    return row


# ----------------------------- LEGACY: single-turn round-0 cost-probe -----------------------------

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
    """LEGACY cost-probe: reward = normalized one-step profit of the model's round-0 action."""
    seed = (row.ground_truth or {}).get("seed", 1)
    world = generate_world(seed, cfg)
    env = FirmEnv(world)
    env.reset()
    text = (row.messages[-1].content or "") if row.messages else ""
    action = validate_action(extract_json(text), cfg)
    _obs, profit, _done, _ = env.step(action)
    score = max(0.0, min(1.0, (profit + 600.0) / 1200.0))
    row.evaluation_result = EvaluateResult(score=score, reason=f"round-0 profit ${profit:.0f}")
    return row
