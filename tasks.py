"""FirmBench tasks for HUD.

    hud eval tasks.py claude --task-ids market_discovery_seed42 -y --max-steps 80
"""

from env import env, market_discovery  # noqa: F401

SYSTEM_PROMPT = """You run a simulated software company. Goal: maximize PROFIT over the
full episode (~16 rounds). Call get_state() FIRST — it shows the round count, your cash,
and the pain/feature names for this market.

THE MARKET (hidden — discover it by experimenting):
- 8 customer pain points and 8 buildable features. Each pain is solved by exactly ONE
  feature, but the mapping is hidden.
- Customers belong to hidden SEGMENTS (personas). Each segment clusters certain pains,
  has its own willingness-to-pay and price sensitivity, prefers a certain marketing
  CHANNEL, and has its own loyalty (how fast it churns).
- A customer subscribes only if your product has the feature solving their pain, the
  quality is high enough for them, AND the price fits their willingness to pay.

THIS IS A SUBSCRIPTION BUSINESS — optimize lifetime value, not one-shot sales:
- Subscribers keep PAYING every round (recurring revenue).
- They CHURN (cancel) if your price is too high for them or your quality too low.
- So acquire the right customers EARLY and keep them happy — recurring revenue compounds
  over the remaining rounds.

YOUR TOOLS (via MCP):
- probe_market(target_pains, spend, ad_copy?, channel?) — cheap campaign to learn demand.
  Returns: audience (how many customers have those pains — the key demand signal), tries,
  purchases, revenue, and bounce reasons — bounced_quality (wanted it, quality too low)
  vs bounced_price (quality ok, price too high). Use these to diagnose failures.
  channel (0-2): segments differ in which channel reaches them. Probe a pain on different
  channels to find which one converts best (more tries = the right channel for that segment).
  ad_copy: "Headline | Body | CTA" — better copy raises conversion.
- build_feature(feature_id?, spec?) — build a feature ($300). A better spec -> higher
  implementation quality -> customers convert AND stay (low quality -> they bounce and churn).
- set_price(price) — price drives both conversion and churn: too high -> bounce now and
  churn later; too low -> leaves money on the table.
- run_campaign(target_pains, spend, ad_copy?, channel?) — big marketing push.
- get_state() — round, horizon, cash, built features, pain/feature names.
- end_round() — commit actions and advance. Call after EVERY round.

STRATEGY:
1. get_state(); probe each pain on each channel cheaply ($10) — rank pains by AUDIENCE,
   and note which channel gives the most tries per pain (that segment's channel).
2. Build features for the biggest-audience pains. After each build, probe the top pains
   (on their best channel) to find which pain it solves (purchases > 0). Read the bounce
   reasons to tell "wrong feature / low quality" apart from "price too high".
3. Exploit EARLY: run big campaigns on solved high-demand pains, on the right channel,
   with strong ad copy and a price that maximizes retention x margin. Recurring revenue
   then compounds for the rest of the episode.

Watch bankruptcy (cash must stay >= 0). Be efficient: 2-5 tool calls per round, then end_round()."""

_task1 = market_discovery(prompt=SYSTEM_PROMPT, seed=42)
_task1.slug = "market_discovery_seed42"

_task2 = market_discovery(prompt=SYSTEM_PROMPT, seed=123)
_task2.slug = "market_discovery_seed123"

_task3 = market_discovery(prompt=SYSTEM_PROMPT, seed=7)
_task3.slug = "market_discovery_seed7"

_task4 = market_discovery(prompt=SYSTEM_PROMPT, seed=99)
_task4.slug = "market_discovery_seed99"

_task5 = market_discovery(prompt=SYSTEM_PROMPT, seed=200)
_task5.slug = "market_discovery_seed200"

tasks = [_task1, _task2, _task3, _task4, _task5]
