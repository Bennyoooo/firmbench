"""FirmBench tasks for HUD.

    hud eval tasks.py claude --task-ids market_discovery_seed42 -y --max-steps 80
"""

from env import env, market_discovery  # noqa: F401

SYSTEM_PROMPT = """You run a simulated software company. Your goal: maximize PROFIT over 10 rounds.

THE MARKET (hidden — you must discover it by experimenting):
- There are 8 customer pain points (IDs 0-7) and 8 buildable features (IDs 0-7).
- Each pain is solved by exactly ONE feature, but the mapping is hidden.
- Customers have 1-3 pains each; some pains are far more common than others (hidden).
- A customer buys only if your product has the feature solving their pain AND your price
  fits their willingness to pay.

YOUR TOOLS (call them via MCP):
- probe_market(target_pains, spend) — run a cheap campaign to learn demand for specific pains.
  Returns audience size (how many customers have those pains) + purchases (if you have the right feature built).
- build_feature(feature_id) — build a feature ($300). Experiment to find which pain it solves.
- set_price(price) — set your price (customers compare to their willingness-to-pay).
- run_campaign(target_pains, spend) — full marketing push (same as probe but for exploitation).
- get_state() — check your cash, price, built features, round number.
- end_round() — commit your actions and advance. You MUST call this after EVERY round.

IMPORTANT WORKFLOW — follow this pattern each round:
1. Decide what to do this round (build? probe? campaign?)
2. Call the relevant tools (2-4 calls per round is typical)
3. Call end_round() to commit and see results
4. Repeat for next round — you have 10 rounds total

STRATEGY:
Round 1: Probe all 8 pains cheaply ($10 each) to find the biggest audiences. Then end_round().
Rounds 2-5: Build one feature per round. After building, probe each top pain ($60 each) to
  discover which pain the new feature solves (purchases > 0 means it matches). Then end_round().
Rounds 6-10: Exploit — run big campaigns on your best discovered pain-feature combos. Then end_round().

Budget: $6000 starting cash. Building costs $300. Don't go bankrupt.
Be efficient — call end_round() after 2-5 tool calls per round, not more."""

_task1 = market_discovery(prompt=SYSTEM_PROMPT, seed=42)
_task1.slug = "market_discovery_seed42"

_task2 = market_discovery(prompt=SYSTEM_PROMPT, seed=123)
_task2.slug = "market_discovery_seed123"

_task3 = market_discovery(prompt=SYSTEM_PROMPT, seed=7)
_task3.slug = "market_discovery_seed7"

tasks = [_task1, _task2, _task3]
