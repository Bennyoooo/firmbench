"""FirmBench EnvironmentAdapter for eval-protocol MCP-Gym (multi-turn RL reward).

Wraps the deterministic `FirmEnv` as a gym-style environment so eval-protocol's `McpGym`
runs a FULL interactive episode (probe -> discover -> build -> exploit) per rollout. The
reward is the real FirmBench objective: **disc.eff = profit / oracle**, delivered DENSELY as
per-round `round_profit / oracle_profit` (summed across the episode == disc.eff). This
replaces the single-turn round-0 cost-probe in ep_firmbench.py — a single completion can't
express FirmBench's discovery game (round 0 has no demand signal), so the reward must be
multi-turn and interactive.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

# repo root on path so `sim`/`agent` import when this file runs as an MCP server subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_protocol.mcp.adapter import EnvironmentAdapter

from sim import Config, generate_world, FirmEnv, OraclePolicy, run_episode, theoretical_max
from agent import extract_json, validate_action

# v1 market (matches the existing ep_firmbench dataset/evaluator + the running GRPO job).
# Flip to Config.phase_a() for the full persona/LTV market (longer episodes, ~16 rounds).
CFG = Config.phase_a() if os.environ.get("FIRMBENCH_PHASE_A") else Config()


class FirmBenchAdapter(EnvironmentAdapter):
    """Adapts FirmEnv to eval-protocol's EnvironmentAdapter interface."""

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> FirmEnv:
        config = config or {}
        seed = int(config.get("seed", 0))
        world = generate_world(seed, CFG)
        env = FirmEnv(world)
        # theoretical_max(world) is the per-world optimistic CEILING (matches env.py's reward
        # normalizer). Per-round reward = round_profit / theoretical_max, so the episode total
        # equals profit / theoretical_max -> honest [0,1].
        env._tmax = theoretical_max(world) or 1.0
        env._seed = seed
        return env

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[FirmEnv, Any, Dict[str, Any]]:
        cfg = {**(config or {})}
        if seed is not None:
            cfg["seed"] = seed
        env = self.create_environment(cfg)
        obs = env.reset()
        return env, obs, {}

    def reset_environment(self, env: FirmEnv, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        return env.reset(), {}

    def step_environment(self, env: FirmEnv, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, profit, done, info = env.step(action)
        tmax = getattr(env, "_tmax", 1.0) or 1.0
        reward = profit / tmax                   # dense; sum over the episode == profit/theoretical_max
        return obs, reward, bool(done), False, info

    def close_environment(self, env: FirmEnv) -> None:
        pass

    def parse_action(self, action_str: str) -> Dict[str, Any]:
        """Parse the model's JSON action (same format as agent.py) into a FirmEnv action."""
        return validate_action(extract_json(action_str or ""), CFG)

    def format_observation(self, observation: Any) -> Any:
        return observation        # FirmEnv obs dict is already JSON-safe

    def get_default_config(self) -> Dict[str, Any]:
        return {"seed": 0}
