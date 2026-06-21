"""FirmBench — HUD v6 environment.

A market-discovery RL environment where an agent runs a firm: it must experiment
to discover a hidden demand structure, build the right features, target the right
customers, and price correctly. Reward = profit, graded on secret held-out users.

Tools are served over MCP. The agent interacts via tool calls each round:
  probe_market(target_pains, spend)  — cheap discovery campaign, returns diagnostics
  build_feature(feature_id)          — build a feature (costs cash)
  set_price(price)                   — set the product price
  run_campaign(target_pains, spend)  — full marketing campaign, returns revenue
  get_state()                        — current firm state
  end_round()                        — advance to the next round

Grading is deterministic: the verifier replays the agent's actions on a secret
20% of the user population that the agent never saw feedback from.
"""

import asyncio
import contextlib
import json
import logging
import socket
import sys
from collections.abc import AsyncGenerator
from typing import Any

from hud import Environment
from hud.capabilities import Capability
from hud.graders import EvaluationResult

from sim import Config, generate_world, FirmEnv, sigmoid

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[%(levelname)s] %(name)s | %(message)s")
for noisy in ("FastMCP", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("firmbench")

env = Environment(name="firmbench")

# ── per-episode state (reset at the top of each template run) ────────

_CFG = Config()
_WORLD = None
_ENV = None
_ROUND_ACTIONS = []          # per-round action log for verifier replay
_CURRENT_ROUND_ACTION = {}   # accumulates within a round
_HOLDOUT_FRAC = 0.2
_EPISODE_SEED = 42

# ── verifier (replays on held-out users) ─────────────────────────────

def _split_users(world):
    n = len(world.users)
    cutoff = int(n * (1 - _HOLDOUT_FRAC))
    return list(range(cutoff)), list(range(cutoff, n))


def _replay_on_holdout(world, holdout_indices, action_log):
    cfg = world.cfg
    ho_by_pain = {p: [] for p in range(cfg.n_pains)}
    for idx in holdout_indices:
        for p in world.users[idx].pains:
            ho_by_pain[p].append(idx)

    built = {}
    total_profit = 0.0
    for entry in action_log:
        f = entry.get("build")
        if f is not None:
            built[f] = 1.0
        price = entry["price"]
        round_revenue = 0.0
        round_spend = 0.0
        for camp in entry.get("campaigns", []):
            target = set(camp["target"])
            spend = camp["spend"] * _HOLDOUT_FRAC
            round_spend += spend
            pool = set()
            for p in target:
                pool.update(ho_by_pain.get(p, []))
            pool = sorted(pool)
            impressions = int(spend * cfg.impressions_per_dollar)
            reached = pool[:impressions]
            purchases = 0.0
            for idx in reached:
                u = world.users[idx]
                resonance = len(target & u.pains) / len(u.pains) if u.pains else 0.0
                ff = 0.0
                if u.pains:
                    for p in u.pains:
                        fe = world.solves[p]
                        if fe in built:
                            ff += built[fe]
                    ff /= len(u.pains)
                p_buy = sigmoid(cfg.alpha * ff + cfg.beta * (u.wtp - price) / u.wtp - cfg.gamma)
                purchases += resonance * p_buy
            round_revenue += purchases * price
        build_cost = cfg.build_cost * _HOLDOUT_FRAC if f is not None else 0.0
        total_profit += round_revenue - round_spend - build_cost
    return total_profit


def _grade_episode():
    _, holdout = _split_users(_WORLD)
    holdout_profit = _replay_on_holdout(_WORLD, holdout, _ROUND_ACTIONS)
    reported = _ENV.total_profit
    reported_scaled = reported * _HOLDOUT_FRAC
    gap = abs(reported_scaled - holdout_profit) / (abs(holdout_profit) + 100.0)
    flagged = gap > 0.15
    reward = holdout_profit
    if flagged:
        reward -= 0.3 * abs(holdout_profit)
    # normalize to [0, 1] roughly (oracle holdout profit ~ 25k)
    normalized = max(0.0, min(1.0, reward / 30000.0))
    return normalized, {
        "holdout_profit": round(holdout_profit, 2),
        "reported_profit": round(reported, 2),
        "gap": round(gap, 4),
        "flagged": flagged,
        "raw_reward": round(reward, 2),
    }


# ── tools (served over MCP) ─────────────────────────────────────────

def _ensure_round_action():
    global _CURRENT_ROUND_ACTION
    if not _CURRENT_ROUND_ACTION:
        _CURRENT_ROUND_ACTION = {"build": None, "price": _ENV.price, "campaigns": []}


async def probe_market(target_pains: list[int], spend: float) -> dict[str, Any]:
    """Run a cheap discovery campaign to learn about demand. Returns audience size,
    impressions, tries, purchases, and revenue for the targeted pain points.

    Args:
        target_pains: list of pain point IDs (0-7) to target.
        spend: dollars to spend on this probe.
    """
    _ensure_round_action()
    target = set(target_pains)
    _CURRENT_ROUND_ACTION["campaigns"].append({"target": sorted(target), "spend": float(spend)})
    result = _ENV._run_campaign(target, float(spend))
    return {k: v for k, v in result.items() if k != "spend"}


async def build_feature(feature_id: int) -> dict[str, Any]:
    """Build a product feature (costs $300). Each feature potentially solves one
    customer pain point, but the mapping is hidden — you must experiment to discover it.

    Args:
        feature_id: which feature to build (0-7).
    """
    _ensure_round_action()
    cfg = _CFG
    if feature_id in _ENV.built:
        return {"success": False, "reason": "already built"}
    if _ENV.cash < cfg.build_cost:
        return {"success": False, "reason": f"not enough cash (have ${_ENV.cash:.0f}, need ${cfg.build_cost:.0f})"}
    _CURRENT_ROUND_ACTION["build"] = int(feature_id)
    return {"success": True, "feature_id": feature_id, "cost": cfg.build_cost}


async def set_price(price: float) -> dict[str, Any]:
    """Set your product's price. Customers buy based on willingness-to-pay vs your price.

    Args:
        price: the price in dollars (must be > 0).
    """
    _ensure_round_action()
    price = max(1.0, min(500.0, float(price)))
    _CURRENT_ROUND_ACTION["price"] = price
    return {"price_set": price}


async def run_campaign(target_pains: list[int], spend: float) -> dict[str, Any]:
    """Run a full marketing campaign. Like probe_market but intended for exploitation
    (higher spend). Returns audience, impressions, tries, purchases, and revenue.

    Args:
        target_pains: list of pain point IDs to target.
        spend: dollars to spend.
    """
    return await probe_market(target_pains, spend)


async def get_state() -> dict[str, Any]:
    """Get your current firm state: round, cash, price, built features."""
    return {
        "round": _ENV.round,
        "horizon": _CFG.horizon,
        "cash": round(_ENV.cash, 2),
        "price": _ENV.price,
        "built_features": sorted(_ENV.built.keys()),
        "done": _ENV.done,
    }


async def end_round() -> dict[str, Any]:
    """Commit your actions and advance to the next round. All builds, price changes,
    and campaigns you called this round are executed together. Returns the new state
    and round profit.
    """
    global _CURRENT_ROUND_ACTION
    _ensure_round_action()
    action = _CURRENT_ROUND_ACTION
    _ROUND_ACTIONS.append(action)

    obs, profit, done, _ = _ENV.step({
        "build": action.get("build"),
        "price": action.get("price", _ENV.price),
        "campaigns": [{"target": set(c["target"]), "spend": c["spend"]}
                      for c in action.get("campaigns", [])],
    })
    _CURRENT_ROUND_ACTION = {}
    return {
        "round": obs["round"],
        "cash": obs["cash"],
        "price": obs["price"],
        "built_features": obs["built_features"],
        "per_campaign": obs["per_campaign"],
        "round_profit": round(profit, 2),
        "done": obs["done"],
    }


# ── MCP lifecycle ───────────────────────────────────────────────────

_HOST = "127.0.0.1"
_MCP_PORT = None
_MCP_SERVER_TASK = None


def _free_port():
    with socket.socket() as s:
        s.bind((_HOST, 0))
        return int(s.getsockname()[1])


async def _listening(port, timeout=15.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((_HOST, port), timeout=0.5).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"MCP server never came up on {_HOST}:{port}")


@env.initialize
async def _up():
    from fastmcp import FastMCP

    global _MCP_PORT, _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is None:
        server = FastMCP(name="firmbench-tools")
        for tool in (probe_market, build_feature, set_price, run_campaign,
                     get_state, end_round):
            server.tool(tool)
        _MCP_PORT = _free_port()
        _MCP_SERVER_TASK = asyncio.create_task(
            server.run_async(transport="http", host=_HOST, port=_MCP_PORT, show_banner=False)
        )
        await _listening(_MCP_PORT)
    env.add_capability(Capability.mcp(name="firmbench", url=f"http://{_HOST}:{_MCP_PORT}/mcp"))


@env.shutdown
async def _down():
    global _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is not None:
        _MCP_SERVER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MCP_SERVER_TASK
        _MCP_SERVER_TASK = None


# ── task template ───────────────────────────────────────────────────

@env.template()
async def market_discovery(prompt: str, seed: int = 42) -> AsyncGenerator[Any, Any]:
    """Run one episode of market discovery: experiment, build, market, price."""
    global _WORLD, _ENV, _ROUND_ACTIONS, _CURRENT_ROUND_ACTION, _EPISODE_SEED

    _EPISODE_SEED = seed
    _WORLD = generate_world(seed, _CFG)
    _ENV = FirmEnv(_WORLD)
    _ENV.reset()
    _ROUND_ACTIONS = []
    _CURRENT_ROUND_ACTION = {}

    yield prompt

    # if the agent didn't finish all rounds, flush any pending action
    if _CURRENT_ROUND_ACTION and not _ENV.done:
        await end_round()

    reward, info = _grade_episode()
    logger.info("market_discovery seed=%d reward=%.3f (%s)", seed, reward, info)
    yield EvaluationResult(reward=reward, content=f"reward={reward:.3f}", info=info)
