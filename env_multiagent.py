"""FirmBench — Phase D HUD environment (multi-agent, pattern A: Coordinator-dispatch).

The single HUD agent IS the Coordinator. Each round it drives the four-role protocol via
delegate tools that act on the underlying `MultiAgentFirmEnv`:

    coordinator_set_budget(budget, directive)  — set the Marketer's budget + a directive
    delegate_build(feature_id?, spec?, note)   — the Builder builds; posts a note
    delegate_price(price, note)                — the Pricer sets the price; posts a note
    delegate_campaigns(campaigns, note)        — the Marketer stages campaigns within budget
    get_team_state()                           — firm summary + the shared blackboard
    end_round()                                — commit the round; returns the Marketer's
                                                 per-campaign diagnostics + new state

Each delegate tool returns ONLY that role's sliced view (the Builder doesn't see the
Marketer's diagnostics, etc.), so the single agent must thread information through the
blackboard exactly as a real team would. Grading = TEAM discovery efficiency
(team_profit / oracle), identical in spirit to env.py:_grade_episode.

Pattern B (native per-role agents) is the documented stretch goal; the role-conditioned
parameter-sharing it implies is realized in the RL pipeline (rft.py / rft_hud.py team mode).

Run `python3 env_multiagent.py` for an OFFLINE selftest that drives the tools with a
scripted Coordinator and prints the team grade — validates the wiring with no HUD/keys.

Serve over HUD (this file is self-contained: env + tools + template + task rows):
    hud eval env_multiagent.py claude \\
        --task-ids multiagent_market_discovery_seed42 -y --gateway --max-steps 100
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

from sim import Config, generate_world, OraclePolicy, run_episode
from multiagent import MultiAgentFirmEnv, ROLES
from scorer import ScoreConfig, score_ad_copy, score_feature_spec

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[%(levelname)s] %(name)s | %(message)s")
for noisy in ("FastMCP", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("firmbench.multiagent")

# ── per-episode state (reset at the top of each template run) ────────
_CFG = Config.phase_a()
_WORLD = None
_MENV = None              # MultiAgentFirmEnv
_EPISODE_SEED = 42
_SCORE_CFG = None
_ROUND_LOG = []          # committed actions per round (for the manifest / replay)
_ARTIFACTS_DIR = None


# ── grading: team discovery efficiency = team_profit / oracle ───────────────────────
def _grade_episode():
    profit = _MENV.total_profit
    oracle_profit = run_episode(_WORLD, OraclePolicy(_WORLD))
    disc_eff = profit / oracle_profit if oracle_profit > 0 else 0.0
    reward = max(0.0, min(1.0, disc_eff))
    coordination_tax = oracle_profit - profit
    info = {
        "team_profit": round(profit, 2),
        "oracle_profit": round(oracle_profit, 2),
        "disc_eff": round(disc_eff, 3),
        "coordination_tax": round(coordination_tax, 2),
        "beat_oracle": disc_eff > 1.0,
        "rounds": len(_ROUND_LOG),
    }
    if _ARTIFACTS_DIR:
        manifest = {"seed": _EPISODE_SEED, "final_reward": round(reward, 4),
                    "pain_names": _WORLD.pain_names, "feature_names": _WORLD.feature_names,
                    "actions": _ROUND_LOG, **info}
        os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
        Path(os.path.join(_ARTIFACTS_DIR, "manifest.json")).write_text(
            json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return reward, info


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _setup_episode(seed):
    """Build the world + multi-agent env for a fresh episode. Shared by the HUD template
    and the offline selftest."""
    global _WORLD, _MENV, _EPISODE_SEED, _SCORE_CFG, _ROUND_LOG, _ARTIFACTS_DIR
    _EPISODE_SEED = seed
    _WORLD = generate_world(seed, _CFG)
    _MENV = MultiAgentFirmEnv(_WORLD)
    _MENV.reset()
    _ROUND_LOG = []
    _ARTIFACTS_DIR = os.path.join("artifacts_multiagent", str(seed))
    _SCORE_CFG = ScoreConfig(fast_mode=not bool(os.environ.get("FIREWORKS_API_KEY")))


# ── tools (served over MCP) ─────────────────────────────────────────

async def get_team_state() -> dict[str, Any]:
    """Coordinator view: the firm summary (round, horizon, cash, price, built features,
    last-round profit & churn) plus the shared blackboard (all role messages so far).
    Call this first each round."""
    obs = _MENV.role_obs("coordinator")
    if _WORLD and _WORLD.pain_names:
        obs["pain_names"] = {i: n for i, n in enumerate(_WORLD.pain_names)}
        obs["feature_names"] = {i: n for i, n in enumerate(_WORLD.feature_names)}
    return obs


async def coordinator_set_budget(budget: float, directive: str = "") -> dict[str, Any]:
    """As the Coordinator, set this round's marketing BUDGET (a cap on total Marketer spend)
    and post a DIRECTIVE to the blackboard for the other roles.

    Args:
        budget: dollars the Marketer may spend this round.
        directive: a short instruction the Builder/Pricer/Marketer can read (e.g.
                   "PHASE: discover — build the next feature; test the top pains").
    """
    _MENV.submit("coordinator", {"budget": float(budget)}, message=directive or None)
    return {"budget_set": float(budget), "directive_posted": bool(directive)}


async def delegate_build(feature_id: int = None, spec: str = None,
                         note: str = "") -> dict[str, Any]:
    """Delegate to the BUILDER: build a feature this round ($300). Tell the Marketer what you
    built via `note` (it can't target the solved pain unless it knows the feature exists).

    Args:
        feature_id: which feature to build (0-7). If a spec is given and id omitted, it's
                    identified from the spec.
        spec: optional feature spec text (scored for implementation quality).
        note: a blackboard message for the team (e.g. "BUILT: feature 3").
    """
    quality = 1.0
    if spec and _WORLD:
        scored = score_feature_spec(spec, _WORLD, _SCORE_CFG)
        if feature_id is None:
            feature_id = scored["feature_id"]
        quality = scored["quality"]
    action = {"build": (int(feature_id) if feature_id is not None else None)}
    if quality != 1.0:
        action["quality"] = quality
    _MENV.submit("builder", action, message=note or None)
    view = _MENV.role_obs("builder")
    return {"build_staged": action["build"], "quality": quality,
            "built_features": view["built_features"], "bounced_quality": view["bounced_quality"]}


async def delegate_price(price: float, note: str = "") -> dict[str, Any]:
    """Delegate to the PRICER: set the product price (drives conversion now and churn later).

    Args:
        price: the price in dollars.
        note: optional blackboard message.
    """
    price = max(1.0, min(500.0, float(price)))
    _MENV.submit("pricer", {"price": price}, message=note or None)
    view = _MENV.role_obs("pricer")
    return {"price_set": price, "bounced_price": view["bounced_price"],
            "recent_purchases": view["recent_purchases"], "last_round_churn": view["last_round_churn"]}


async def delegate_campaigns(campaigns: list[dict], note: str = "") -> dict[str, Any]:
    """Delegate to the MARKETER: stage this round's ad campaigns (spend is clamped to the
    Coordinator's budget). Each campaign = {target_pains:[int], spend:float, channel:int,
    ad_copy?:str}. Results arrive after end_round().

    Args:
        campaigns: list of campaign dicts.
        note: blackboard message (e.g. "TARGETS: [0, 2, 5]").
    """
    staged = []
    for c in (campaigns or []):
        tgt = c.get("target_pains", c.get("target", []))
        if isinstance(tgt, int):
            tgt = [tgt]
        craft = 1.0
        ad_copy = c.get("ad_copy")
        if ad_copy and _WORLD:
            scored = score_ad_copy(ad_copy, _WORLD, _SCORE_CFG)
            craft = scored["craft"]
            tgt = sorted(set(tgt) | scored["target_pains"]) if tgt else sorted(scored["target_pains"])
        staged.append({"target": set(int(p) for p in tgt),
                       "spend": max(0.0, float(c.get("spend", 0.0))),
                       "channel": int(c.get("channel", 0)), "craft": craft})
    _MENV.submit("marketer", {"campaigns": staged}, message=note or None)
    view = _MENV.role_obs("marketer")
    return {"campaigns_staged": len(staged), "budget": view["budget"]}


async def end_round() -> dict[str, Any]:
    """Commit the staged role actions: assemble {build, price, campaigns} and step the firm.
    Returns the round profit and the Marketer's per-campaign diagnostics (audience, tries,
    purchases, bounce reasons) — the discovery signal for next round."""
    # capture the round's coordination story BEFORE commit() clears the blackboard/stash
    messages = [dict(m) for m in _MENV.bb.msgs]
    budget = _MENV._budget
    _obs, profit, done, committed = _MENV.commit()
    _ROUND_LOG.append({
        "build": committed["build"],
        "price": committed["price"],
        "budget": (round(budget, 2) if budget is not None else None),
        "messages": messages,                              # blackboard: who said what
        "campaigns": _MENV._last_per_campaign,             # results: audience/tries/purchases/revenue
        "round_profit": round(profit, 2),
    })
    mk = _MENV.role_obs("marketer")
    return {"round": _MENV.round, "round_profit": round(profit, 2),
            "cash": mk["cash"], "price": mk["price"], "built_features": mk["built_features"],
            "per_campaign": mk["per_campaign"], "done": done}


_TOOLS = (get_team_state, coordinator_set_budget, delegate_build, delegate_price,
          delegate_campaigns, end_round)


# ── MCP lifecycle (mirrors env.py) ──────────────────────────────────
try:
    from hud import Environment
    from hud.capabilities import Capability
    from hud.graders import EvaluationResult
    env = Environment(name="firmbench-multiagent")
    _HUD_AVAILABLE = True
except Exception:                                   # offline selftest needs no hud
    env = None
    _HUD_AVAILABLE = False

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


if _HUD_AVAILABLE:
    @env.initialize
    async def _up():
        from fastmcp import FastMCP
        global _MCP_PORT, _MCP_SERVER_TASK
        if _MCP_SERVER_TASK is None:
            server = FastMCP(name="firmbench-multiagent-tools")
            for tool in _TOOLS:
                server.tool(tool)
            _MCP_PORT = _free_port()
            _MCP_SERVER_TASK = asyncio.create_task(
                server.run_async(transport="http", host=_HOST, port=_MCP_PORT, show_banner=False))
            await _listening(_MCP_PORT)
        env.add_capability(Capability.mcp(name="firmbench-multiagent",
                                          url=f"http://{_HOST}:{_MCP_PORT}/mcp"))

    @env.shutdown
    async def _down():
        global _MCP_SERVER_TASK
        if _MCP_SERVER_TASK is not None:
            _MCP_SERVER_TASK.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _MCP_SERVER_TASK
            _MCP_SERVER_TASK = None

    @env.template()
    async def multiagent_market_discovery(prompt: str, seed: int = 42) -> AsyncGenerator[Any, Any]:
        """One episode of multi-agent market discovery (Coordinator-dispatch)."""
        _setup_episode(seed)
        yield prompt
        # flush any half-staged round so a lazy agent still gets graded
        if _MENV._stash and not _MENV.done:
            await end_round()
        reward, info = _grade_episode()
        logger.info("multiagent_market_discovery seed=%d reward=%.3f (%s)",
                    seed, reward, info)
        yield EvaluationResult(reward=reward, content=f"reward={reward:.3f}", info=info)

    # Concrete task rows live in THIS file so `hud eval env_multiagent.py` serves the
    # firmbench-multiagent env directly. (A task-only file like tasks.py resolves its env to
    # the sibling env.py — the single-agent env — which has the wrong name for these tasks.)
    from tasks import MULTIAGENT_SYSTEM_PROMPT  # noqa: E402

    _t1 = multiagent_market_discovery(prompt=MULTIAGENT_SYSTEM_PROMPT, seed=42)
    _t1.slug = "multiagent_market_discovery_seed42"
    _t2 = multiagent_market_discovery(prompt=MULTIAGENT_SYSTEM_PROMPT, seed=123)
    _t2.slug = "multiagent_market_discovery_seed123"
    _t3 = multiagent_market_discovery(prompt=MULTIAGENT_SYSTEM_PROMPT, seed=7)
    _t3.slug = "multiagent_market_discovery_seed7"
    tasks = [_t1, _t2, _t3]   # the single list HUD's loader scans (underscore vars skipped)


# ── offline selftest (no HUD, no keys): drive the tools with a scripted Coordinator ──
async def _selftest_episode(seed):
    """Validate the tool wiring by driving the delegate tools with the ScriptedTeam's brain
    (a stand-in Coordinator). Proves the HUD env grades a team episode correctly offline."""
    from multiagent import _RolePolicies
    _setup_episode(seed)
    brain = _RolePolicies(_WORLD, seed, read_messages=True)
    brain.reset()
    while not _MENV.done:
        # the Coordinator agent threads the protocol through the tools, in order
        co_obs = _MENV.role_obs("coordinator")
        co_act, co_msg = brain.coordinator(co_obs)
        await coordinator_set_budget(co_act["budget"], directive=co_msg or "")

        bu_act, bu_msg = brain.builder(_MENV.role_obs("builder"))
        await delegate_build(feature_id=bu_act.get("build"), note=bu_msg or "")

        pr_act, _ = brain.pricer(_MENV.role_obs("pricer"))
        await delegate_price(pr_act["price"])

        ma_act, ma_msg = brain.marketer(_MENV.role_obs("marketer"))
        camps = [{"target_pains": sorted(c["target"]), "spend": c["spend"],
                  "channel": c.get("channel", 0)} for c in ma_act.get("campaigns", [])]
        await delegate_campaigns(camps, note=ma_msg or "")
        await end_round()
    return _grade_episode()


def _selftest():
    print("=" * 60)
    print("FirmBench — multi-agent HUD env OFFLINE selftest (pattern A)")
    print("=" * 60)
    for seed in (42, 123, 7):
        reward, info = asyncio.run(_selftest_episode(seed))
        print(f"  seed {seed:>3}: reward={reward:.3f}  team_profit={info['team_profit']:.0f}  "
              f"oracle={info['oracle_profit']:.0f}  tax={info['coordination_tax']:.0f}  "
              f"rounds={info['rounds']}")
    print("-" * 60)
    print("OK — the delegate-tool round protocol grades a team episode (disc.eff = team/oracle).")


if __name__ == "__main__":
    _selftest()
