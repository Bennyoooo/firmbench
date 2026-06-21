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

First call get_state() to see the pain names and feature names for this market.

YOUR TOOLS (call them via MCP):
- probe_market(target_pains, spend, ad_copy?) — run a campaign to learn about demand.
  Returns audience, impressions, tries, purchases, revenue.
  Optional: pass ad_copy as "Headline | Body text | CTA button text" to create a real ad.
  Example: ad_copy="Stop Losing Customers | Our search finds what you need instantly | Try Free"
- build_feature(feature_id?, spec?) — build a feature ($300).
  Pass feature_id (0-7) OR pass spec text and the system infers which feature you mean.
  Optional: write a spec to create a product page. Include a title line, description, and
  bullet points for benefits (prefix with - or •).
  Example: spec="Smart Search Engine\\nFind anything instantly with autocomplete.\\n- Sub-100ms results\\n- Typo tolerance\\n- Faceted filters"
- set_price(price) — set your price.
- run_campaign(target_pains, spend, ad_copy?) — full marketing push (same as probe, higher spend).
- get_state() — check cash, price, built features, round number, AND pain/feature names.
- end_round() — commit actions and advance. Call this after EVERY round.

IMPORTANT WORKFLOW — follow this pattern each round:
1. Decide what to do this round (build? probe? campaign?)
2. Call the relevant tools (2-4 calls per round is typical)
3. Call end_round() to commit and see results
4. Repeat for next round — you have 10 rounds total

STRATEGY:
Round 1: Call get_state() to see pain/feature names. Probe all 8 pains cheaply ($10 each)
  to find the biggest audiences. Write compelling ad copy for each probe. Then end_round().
Rounds 2-5: Build one feature per round — write a spec describing what it does. After
  building, probe each top pain ($60 each) with targeted ad copy to discover which pain
  the new feature solves (purchases > 0 means it matches). Then end_round().
Rounds 6-10: Exploit — run big campaigns with polished ad copy on your best discovered
  pain-feature combos at the optimal price. Then end_round().

AD COPY TIPS: Good ad copy names the specific pain ("tired of slow search?"), states a
concrete benefit ("find results in under 100ms"), and has a clear CTA ("Try it free").
The better your copy, the higher your conversion rate.

Budget: $6000 starting cash. Building costs $300. Don't go bankrupt.
Be efficient — call end_round() after 2-5 tool calls per round, not more."""

_task1 = market_discovery(prompt=SYSTEM_PROMPT, seed=42)
_task1.slug = "market_discovery_seed42"

_task2 = market_discovery(prompt=SYSTEM_PROMPT, seed=123)
_task2.slug = "market_discovery_seed123"

_task3 = market_discovery(prompt=SYSTEM_PROMPT, seed=7)
_task3.slug = "market_discovery_seed7"

tasks = [_task1, _task2, _task3]
