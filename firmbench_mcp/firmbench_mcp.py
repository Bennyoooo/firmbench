"""FirmBench MCP-Gym environment (eval-protocol multi-turn RL).

One `firm_round` tool call == one market round (build + price + campaigns committed
together). The model sees per-campaign diagnostics in the next observation and adapts —
interactive discovery. Reward (control plane, server-side) = per-round profit/oracle,
summing across the episode to the real disc.eff. Mirrors the FrozenLake MCP-Gym example.
"""

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from mcp.server.fastmcp import Context
from eval_protocol.mcp import McpGym

from firmbench_adapter import FirmBenchAdapter, CFG

_TOOL_DESC = (
    "Commit this round's firm action and advance one round. `action_json` is ONE JSON object: "
    '{"build": feature_id (0-7) or null, "price": number, '
    '"campaigns": [{"target": [pain_ids], "spend": dollars, "channel": 0-2}]}. '
    "Returns the next observation: per-campaign audience (demand size), tries, purchases, "
    "revenue, and bounce reasons. Probe pains cheaply ($10) in early rounds to learn demand, "
    "build features for the biggest-audience pains, discover which pain each solves "
    "(purchases>0), then exploit. Maximize cumulative profit over the episode."
)


class FirmBenchMcp(McpGym):
    """FirmBench as an MCP-Gym environment."""

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__("FirmBench-v1", FirmBenchAdapter(), seed, **kwargs)

    def _register_tools(self):
        @self.mcp.tool(name="firm_round", description=_TOOL_DESC)
        def firm_round(action_json: str, ctx: Context) -> Dict[str, Any]:
            action = self.adapter.parse_action(action_json)
            session_id = self._get_session_id(ctx)
            self._get_or_create_session(ctx)
            obs = self._execute_session_environment_step(session_id, action)
            return obs

    def format_observation(self, obs: Any, env: Any) -> Dict[str, Any]:
        """Data-plane observation the model sees each round (no reward/control-plane data)."""
        if isinstance(obs, dict):
            return {
                "round": obs.get("round"),
                "horizon": CFG.horizon,
                "cash": obs.get("cash"),
                "price": obs.get("price"),
                "built_features": obs.get("built_features"),
                "per_campaign": obs.get("per_campaign", []),
                "done": obs.get("done"),
            }
        return {"observation": obs}
