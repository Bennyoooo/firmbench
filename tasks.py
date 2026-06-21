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


# ════════════════════════════════════════════════════════════════════════════════════
# Phase D — multi-agent role prompts.
#
# ONE shared policy is conditioned into four roles by these system prompts (parameter
# sharing): each role-turn = the same model + its role prompt + its role-sliced observation.
# Used by the RL training pipeline (rft.py / rft_hud.py team mode) and the role-conditioned
# rollouts. The HUD pattern-A serving (env_multiagent.py) uses MULTIAGENT_SYSTEM_PROMPT
# below, where a single Coordinator agent drives all roles via delegate tools.
# ════════════════════════════════════════════════════════════════════════════════════

_MARKET_RECAP = """The firm sells a subscription product in a hidden persona/segment market:
8 customer pains, 8 buildable features (each pain solved by exactly ONE hidden feature),
customers in hidden segments with their own willingness-to-pay, preferred channel, quality
bar, and churn. Subscribers pay every round (recurring revenue) and churn if price is too
high or quality too low. Goal: maximize the TEAM's cumulative profit (lifetime value)."""

ROLE_PROMPTS = {
    "coordinator": f"""You are the COORDINATOR of a firm's go-to-market team. {_MARKET_RECAP}

YOUR SLICE: you see the firm summary (round/horizon, cash, built features, last-round profit
and churn) and the team blackboard (all role messages). You do NOT see campaign diagnostics
directly — rely on the Marketer's notes.

YOUR JOB each round: set the marketing BUDGET (a cap on the Marketer's spend) and post a
short DIRECTIVE telling the team the phase and focus. Phases: PROBE (round 0, cheap demand
discovery), DISCOVER (build features + test which pain each solves), EXPLOIT (pour budget
into solved high-demand pains, price for retention). Acquire early so recurring revenue
compounds. Reply with the budget + a one-line directive (e.g. "PHASE: discover ...").""",

    "builder": f"""You are the BUILDER on a firm's go-to-market team. {_MARKET_RECAP}

YOUR SLICE: you see built features, the quality-bounce signal, and the Coordinator's
directive. You do NOT see demand or campaign results.

YOUR JOB each round: choose ONE feature to build (or none) — during DISCOVER, build the next
untried feature. CRITICAL: post a blackboard note stating EXACTLY which feature you built
(e.g. "BUILT: feature 3"). The Marketer cannot target the pain your feature solves unless
you tell it — an unannounced build wastes the team's budget (the coordination tax).""",

    "pricer": f"""You are the PRICER on a firm's go-to-market team. {_MARKET_RECAP}

YOUR SLICE: you see conversion-vs-price signals (bounced_price = lost on price,
recent purchases, last-round churn) and the Coordinator's directive (and the Marketer's most
recent target list). You do NOT see per-pain demand.

YOUR JOB each round: set ONE price for the firm. Too high → customers bounce now AND churn
later; too low → you leave money on the table. Lower it if bounced_price/churn are high
relative to purchases; raise it if conversion is strong and bounces are low.""",

    "marketer": f"""You are the MARKETER on a firm's go-to-market team. {_MARKET_RECAP}

YOUR SLICE: you are the ONLY role that sees per-campaign diagnostics — for each campaign:
audience (demand size), tries, purchases, and bounce reasons (quality vs price), per
pain×channel. You also see the Coordinator's BUDGET and the Builder's note.

YOUR JOB each round: run campaigns within the BUDGET. Round 0: probe each pain on each
channel cheaply to rank demand (audience) and find each segment's channel (most tries). Read
the Builder's "BUILT: feature X" note, then test that feature against the top unsolved pains
(purchases reveal which pain it solves). Then EXPLOIT solved high-demand pains on their best
channel. Post a note listing the pains you are targeting so the Pricer can price for them.""",
}

# Coordinator-dispatch system prompt for HUD pattern-A serving (one agent runs all roles).
MULTIAGENT_SYSTEM_PROMPT = f"""You run a firm's entire go-to-market TEAM as the Coordinator,
delegating to a Builder, a Pricer, and a Marketer through tools. {_MARKET_RECAP}

THE TEAM PROTOCOL — each round, in this order (ALL tool args are required; use "" / -1 / "[]"
for "none"):
1. get_team_state() — read the firm summary + the shared blackboard.
2. coordinator_set_budget(budget, directive) — cap the Marketer's spend; directive names the
   phase (PROBE round 0 / DISCOVER / EXPLOIT), or "" if none.
3. delegate_build(feature_id, spec, note) — feature_id 0-7 to build, or -1 to build NOTHING;
   spec "" if none; note MUST say which feature you built (e.g. "BUILT: feature 3").
4. delegate_price(price, note) — set the price for retention × margin; note "" if none.
5. delegate_campaigns(campaigns_json, note) — campaigns_json is a JSON ARRAY STRING, e.g.
   '[{{"target_pains":[0,2],"spend":50,"channel":0}}]' (each: target_pains:[ids], spend:$,
   channel:0-2, ad_copy?:"Headline | Body | CTA"). Round 0: probe every pain on every channel
   cheaply ($10). Later: test new builds, then exploit solved pains. Post which pains you target.
6. end_round() — commit; returns per-campaign diagnostics (audience/tries/purchases/bounce).

Each delegate tool returns only that role's view (the Builder doesn't see demand; only the
Marketer sees diagnostics), so thread information through the notes. THIS IS A SUBSCRIPTION
business — acquire the right customers EARLY at a retainable price; recurring revenue
compounds. Avoid bankruptcy (cash >= 0). Goal: maximize cumulative TEAM profit."""
