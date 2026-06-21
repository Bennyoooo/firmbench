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

All campaign/build tools accept optional NL artifact params (ad_copy, spec).
When provided, artifacts are scored via the translator and rendered as HTML.
When omitted, the structured fast path runs (craft=1.0, quality=1.0).

Grading is deterministic: the verifier replays the agent's actions on a secret
20% of the user population that the agent never saw feedback from.
"""

import asyncio
import contextlib
import json
import logging
import os
import socket
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from hud import Environment
from hud.capabilities import Capability
from hud.graders import EvaluationResult

from sim import Config, generate_world, FirmEnv, sigmoid, OraclePolicy, run_episode
from scorer import ScoreConfig, score_ad_copy, score_feature_spec
from renderer import render_ad_card, render_feature_page

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[%(levelname)s] %(name)s | %(message)s")
for noisy in ("FastMCP", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("firmbench")

env = Environment(name="firmbench")

# ── per-episode state (reset at the top of each template run) ────────

_CFG = Config.phase_a()      # deployed env runs the full Phase A market
_WORLD = None
_ENV = None
_ROUND_ACTIONS = []          # per-round action log for verifier replay
_CURRENT_ROUND_ACTION = {}   # accumulates within a round
_HOLDOUT_FRAC = 0.2
_EPISODE_SEED = 42
_ARTIFACTS_DIR = None        # set per-episode: artifacts/{seed}/
_SCORE_CFG = None            # scorer config (fast_mode unless env vars set)

# ── grading: discovery efficiency = profit / oracle (no user holdout) ─────────
# In an execution-based env the agent can't fake its numbers, and the latent market
# structure is shared across users — so a user-level holdout adds nothing here.
# Generalization is measured at the SEED level (domain randomization + held-out eval
# seeds). Reward = profit / oracle clipped to [0,1] (the design's "regret vs oracle").

def _grade_episode():
    profit = _ENV.total_profit
    oracle_profit = run_episode(_WORLD, OraclePolicy(_WORLD))
    disc_eff = profit / oracle_profit if oracle_profit > 0 else 0.0
    reward = max(0.0, min(1.0, disc_eff))
    beat_oracle = disc_eff > 1.0          # C3 guard: reference no longer a valid ceiling

    if _ARTIFACTS_DIR:
        manifest = {
            "seed": _EPISODE_SEED,
            "rounds": len(_ROUND_ACTIONS),
            "final_reward": round(reward, 4),
            "profit": round(profit, 2),
            "oracle_profit": round(oracle_profit, 2),
            "disc_eff": round(disc_eff, 3),
            "pain_names": _WORLD.pain_names,
            "feature_names": _WORLD.feature_names,
            "actions": _ROUND_ACTIONS,
        }
        manifest_path = os.path.join(_ARTIFACTS_DIR, "manifest.json")
        os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2, default=_json_default),
                                        encoding="utf-8")
        logger.info("Manifest written to %s", manifest_path)

    return reward, {
        "profit": round(profit, 2),
        "oracle_profit": round(oracle_profit, 2),
        "disc_eff": round(disc_eff, 3),
        "beat_oracle": beat_oracle,
    }


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── tools (served over MCP) ─────────────────────────────────────────

def _ensure_round_action():
    global _CURRENT_ROUND_ACTION
    if not _CURRENT_ROUND_ACTION:
        _CURRENT_ROUND_ACTION = {"build": None, "price": _ENV.price, "campaigns": []}


def _campaign_artifact_path(campaign_idx: int) -> str:
    if not _ARTIFACTS_DIR:
        return None
    return os.path.join(_ARTIFACTS_DIR,
                        f"round_{_ENV.round:02d}",
                        f"campaign_{campaign_idx}",
                        "ad_card.html")


def _feature_artifact_path(feature_id: int) -> str:
    if not _ARTIFACTS_DIR:
        return None
    return os.path.join(_ARTIFACTS_DIR,
                        f"round_{_ENV.round:02d}",
                        f"feature_{feature_id}",
                        "feature_page.html")


async def probe_market(target_pains: list[int], spend: float,
                       ad_copy: str = None, channel: int = 0) -> dict[str, Any]:
    """Run a cheap discovery campaign to learn about demand. Returns audience size,
    impressions, tries, purchases, and revenue for the targeted pain points.

    Args:
        target_pains: list of pain point IDs (0-7) to target.
        spend: dollars to spend on this probe.
        ad_copy: optional ad copy text (headline | body | CTA). When provided,
                 the copy is scored for quality and targeting is extracted from it.
        channel: marketing channel id (0-2); segments differ in which channel reaches them.
    """
    _ensure_round_action()

    craft = 1.0
    camp_entry = {"spend": float(spend), "channel": int(channel)}

    if ad_copy and _WORLD:
        scored = score_ad_copy(ad_copy, _WORLD, _SCORE_CFG)
        craft = scored["craft"]
        nl_targets = scored["target_pains"]
        # NL-extracted targeting supplements the explicit target_pains
        target = set(target_pains) | nl_targets if target_pains else nl_targets
        camp_entry["ad_copy"] = ad_copy
        camp_entry["craft"] = craft
        camp_entry["nl_targets"] = sorted(nl_targets)

        # render the ad card
        camp_idx = len(_CURRENT_ROUND_ACTION.get("campaigns", []))
        art_path = _campaign_artifact_path(camp_idx)
        if art_path and _WORLD.pain_names:
            parts = ad_copy.split("|", 2)
            headline = parts[0].strip() if len(parts) > 0 else ad_copy[:60]
            body = parts[1].strip() if len(parts) > 1 else ""
            cta = parts[2].strip() if len(parts) > 2 else "Learn More"
            render_ad_card(headline, body, cta, target,
                           _WORLD.pain_names, craft, float(spend), art_path)
            camp_entry["artifact_path"] = art_path
    else:
        target = set(target_pains)

    camp_entry["target"] = sorted(target)
    _CURRENT_ROUND_ACTION["campaigns"].append(camp_entry)
    result = _ENV._run_campaign(target, float(spend), channel=int(channel), craft=craft)
    # Store campaign results in the action log for replay viewer
    camp_entry["audience"] = result.get("audience", 0)
    camp_entry["impressions"] = result.get("impressions", 0)
    camp_entry["tries"] = result.get("tries", 0)
    camp_entry["purchases"] = result.get("purchases", 0)
    camp_entry["revenue"] = result.get("revenue", 0)
    out = {k: v for k, v in result.items() if not k.startswith("_") and k != "spend"}
    if craft != 1.0:
        out["craft_score"] = craft
    return out


async def build_feature(feature_id: int = None,
                        spec: str = None) -> dict[str, Any]:
    """Build a product feature (costs $300). Each feature potentially solves one
    customer pain point, but the mapping is hidden — you must experiment to discover it.

    Args:
        feature_id: which feature to build (0-7). If spec is provided and feature_id
                    is omitted, the feature is identified from the spec text.
        spec: optional feature specification text. When provided, the spec is scored
              for quality which affects how well the feature works.
    """
    _ensure_round_action()
    cfg = _CFG
    quality = 1.0

    if spec and _WORLD:
        scored = score_feature_spec(spec, _WORLD, _SCORE_CFG)
        if feature_id is None:
            feature_id = scored["feature_id"]
        quality = scored["quality"]
        _CURRENT_ROUND_ACTION["spec"] = spec
        _CURRENT_ROUND_ACTION["quality"] = quality

        # render the feature page
        art_path = _feature_artifact_path(feature_id)
        if art_path and _WORLD.feature_names:
            fname = _WORLD.feature_names[feature_id] if feature_id < len(_WORLD.feature_names) else f"Feature {feature_id}"
            lines = spec.strip().split("\n")
            tagline = lines[0][:100] if lines else fname
            description = "\n".join(lines[1:])[:500] if len(lines) > 1 else spec[:500]
            benefits = [l.strip("- •").strip() for l in lines if l.strip().startswith(("-", "•", "*"))][:5]
            render_feature_page(fname, tagline, description,
                                benefits or [spec[:80]], quality, feature_id, art_path)
            _CURRENT_ROUND_ACTION["feature_artifact_path"] = art_path

    if feature_id is None:
        return {"success": False, "reason": "feature_id is required (or provide a spec)"}
    if feature_id in _ENV.built:
        return {"success": False, "reason": "already built"}
    if _ENV.cash < cfg.build_cost:
        return {"success": False, "reason": f"not enough cash (have ${_ENV.cash:.0f}, need ${cfg.build_cost:.0f})"}

    _CURRENT_ROUND_ACTION["build"] = int(feature_id)
    if quality != 1.0:
        _CURRENT_ROUND_ACTION["quality"] = quality

    result = {"success": True, "feature_id": feature_id, "cost": cfg.build_cost}
    if _WORLD and feature_id < len(_WORLD.feature_names):
        result["feature_name"] = _WORLD.feature_names[feature_id]
    if quality != 1.0:
        result["quality_score"] = quality
    return result


async def set_price(price: float) -> dict[str, Any]:
    """Set your product's price. Customers buy based on willingness-to-pay vs your price.

    Args:
        price: the price in dollars (must be > 0).
    """
    _ensure_round_action()
    price = max(1.0, min(500.0, float(price)))
    _CURRENT_ROUND_ACTION["price"] = price
    return {"price_set": price}


async def run_campaign(target_pains: list[int], spend: float,
                       ad_copy: str = None, channel: int = 0) -> dict[str, Any]:
    """Run a full marketing campaign. Like probe_market but intended for exploitation
    (higher spend). Returns audience, impressions, tries, purchases, and revenue.

    Args:
        target_pains: list of pain point IDs to target.
        spend: dollars to spend.
        ad_copy: optional ad copy text (headline | body | CTA).
    """
    return await probe_market(target_pains, spend, ad_copy=ad_copy, channel=channel)


async def get_state() -> dict[str, Any]:
    """Get your current firm state: round, cash, price, built features, and
    the names of all pains and features in this market."""
    state = {
        "round": _ENV.round,
        "horizon": _CFG.horizon,
        "cash": round(_ENV.cash, 2),
        "price": _ENV.price,
        "built_features": sorted(_ENV.built.keys()),
        "done": _ENV.done,
    }
    if _WORLD and _WORLD.pain_names:
        state["pain_names"] = {i: n for i, n in enumerate(_WORLD.pain_names)}
        state["feature_names"] = {i: n for i, n in enumerate(_WORLD.feature_names)}
    return state


async def end_round() -> dict[str, Any]:
    """Commit your actions and advance to the next round. All builds, price changes,
    and campaigns you called this round are executed together. Returns the new state
    and round profit.
    """
    global _CURRENT_ROUND_ACTION
    _ensure_round_action()
    action = _CURRENT_ROUND_ACTION
    _ROUND_ACTIONS.append(action)

    # Apply quality to the build if NL mode was used
    quality = action.get("quality", 1.0)

    obs, profit, done, _ = _ENV.step({
        "build": action.get("build"),
        "price": action.get("price", _ENV.price),
        "campaigns": [{"target": set(c["target"]), "spend": c["spend"],
                       "channel": c.get("channel", 0), "craft": c.get("craft", 1.0)}
                      for c in action.get("campaigns", [])],
    })

    # Override the default quality=1.0 if NL scoring set a different value
    if action.get("build") is not None and quality != 1.0:
        _ENV.built[action["build"]] = quality

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
    global _ARTIFACTS_DIR, _SCORE_CFG

    _EPISODE_SEED = seed
    _WORLD = generate_world(seed, _CFG)
    _ENV = FirmEnv(_WORLD)
    _ENV.reset()
    _ROUND_ACTIONS = []
    _CURRENT_ROUND_ACTION = {}

    # set up artifacts dir and scorer config
    _ARTIFACTS_DIR = os.path.join("artifacts", str(seed))
    has_key = bool(os.environ.get("FIREWORKS_API_KEY"))
    _SCORE_CFG = ScoreConfig(fast_mode=not has_key)

    yield prompt

    # if the agent didn't finish all rounds, flush any pending action
    if _CURRENT_ROUND_ACTION and not _ENV.done:
        await end_round()

    reward, info = _grade_episode()
    logger.info("market_discovery seed=%d reward=%.3f (%s)", seed, reward, info)
    yield EvaluationResult(reward=reward, content=f"reward={reward:.3f}", info=info)
