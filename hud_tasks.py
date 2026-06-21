"""FirmBench tasks for HUD.

    hud eval hud_tasks.py claude --task-ids market_discovery_seed42 -y --max-steps 30
"""

from hud_env import env, market_discovery  # noqa: F401

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
- end_round() — commit your actions and advance. You MUST call this to finish each round.

STRATEGY:
1. Probe: run cheap single-pain campaigns ($10 each) to discover which pains have the most customers.
2. Discover: build features one by one; after building, run campaigns to see which pain it solves (purchases appear).
3. Exploit: once you know the mapping, target your best pains at the right price with big campaigns.
4. Don't go bankrupt — you start with $6000 and building costs $300 each.

You have 10 rounds. Call end_round() after each round's actions to advance."""

_task1 = market_discovery(prompt=SYSTEM_PROMPT, seed=42)
_task1.slug = "market_discovery_seed42"

_task2 = market_discovery(prompt=SYSTEM_PROMPT, seed=123)
_task2.slug = "market_discovery_seed123"

_task3 = market_discovery(prompt=SYSTEM_PROMPT, seed=7)
_task3.slug = "market_discovery_seed7"

tasks = [_task1, _task2, _task3]
