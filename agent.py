"""
FirmBench — Phase 2: single-agent LLM harness (Fireworks, OpenAI-compatible).

The agent drives `FirmEnv` (from sim.py) by emitting a JSON action each round. We use
structured-JSON prompting (not the native tools API) so it works with any cheap open
model on Fireworks. Same `.reset()/.act(env, obs)` interface as the scripted policies,
so it plugs into the existing runner + oracle baseline.

Usage:
    export FIREWORKS_API_KEY=...                       # required for real runs
    export FIREWORKS_MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct   # optional
    python3 agent.py                                   # runs the LLM agent over seeds

Without a key it prints instructions and falls back to the scripted baseline so the
wiring can be checked offline.
"""

import os
import re
import json
import logging

from sim import (Config, generate_world, FirmEnv, OraclePolicy, ScriptedExperimenter,
                 run_episode)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("firmbench.agent")

DEFAULT_MODEL = "accounts/fireworks/models/glm-5p1"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


# ----------------------------- prompting -----------------------------

def system_prompt(cfg: Config) -> str:
    cost_per_reached = round(1.0 / cfg.impressions_per_dollar, 2)
    return f"""You run a software company in a simulated market. Maximize CUMULATIVE PROFIT
over {cfg.horizon} rounds.

THE MARKET (structure is HIDDEN — you must discover it by experimenting):
- There are {cfg.n_pains} customer pain points (ids 0..{cfg.n_pains-1}) and
  {cfg.n_features} buildable product features (ids 0..{cfg.n_features-1}).
- Each pain is solved by exactly ONE feature, via a hidden mapping you do NOT know.
- Customers have 1-3 pains each; some pains are far more common than others (hidden).
- A customer buys only if your product has the feature that solves their pain AND your
  price fits their willingness to pay.

EACH ROUND you choose an action (JSON):
- build: one feature id to build this round (costs ${int(cfg.build_cost)}), or null.
- price: the product price.
- campaigns: a list of ad campaigns, each = {{"target": [pain ids], "spend": dollars}}.
  An ad reaches customers who have at least one targeted pain. Reaching a customer costs
  about ${cost_per_reached}. Targeting pains your product can't solve wastes money.

FEEDBACK each round (per campaign): "audience" (how many customers match the target —
this reveals demand size), impressions reached, tries, purchases, revenue. Use cheap
single-pain campaigns to probe audience sizes and to discover which built feature solves
which pain (purchases appear only when the right feature is built).

STRATEGY: experiment cheaply first (probe demand, discover the mapping), then build the
features for the most common pains and spend to exploit them at a good price. Avoid
bankruptcy (cash must stay >= 0). You start with ${int(cfg.starting_cash)}.

KEY INSIGHT: The "audience" number is your most important signal — it tells you how many
customers have each pain (demand size). Rank pains by audience. Build only 2-3 features
for the highest-audience pains. After building, probe to discover which pain each feature
solves (purchases > 0 means match). Then spend big on exploitation campaigns for the
discovered matches. Build few, exploit hard.

ALWAYS end your reply with ONE JSON object in a ```json code block, e.g.:
```json
{{"build": 2, "price": 45, "campaigns": [{{"target": [0], "spend": 50}}]}}
```"""


def format_obs(obs: dict, cfg: Config, history: list) -> str:
    lines = [f"Round {obs['round']}/{cfg.horizon} | cash ${obs['cash']:.0f} | "
             f"price ${obs['price']:.0f} | built features {obs['built_features']}"]
    if obs["per_campaign"]:
        lines.append("Last round campaign results:")
        for c in obs["per_campaign"]:
            lines.append(
                f"  target {c['target']}: audience {c['audience']}, "
                f"impressions {c['impressions']}, tries {c['tries']}, "
                f"purchases {c['purchases']}, revenue ${c['revenue']:.0f}, "
                f"spend ${c['spend']:.0f}")
    if history:
        lines.append("Notes from earlier rounds:")
        lines.extend(history[-6:])
    lines.append("Choose your action for this round.")
    return "\n".join(lines)


# ----------------------------- action parsing -----------------------------

def extract_json(text: str):
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = m[:]
    if not candidates:
        # fall back to the last balanced-looking {...}
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates = [text[start:end + 1]]
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def validate_action(raw, cfg: Config) -> dict:
    safe = {"build": None, "price": 50.0, "campaigns": []}
    if not isinstance(raw, dict):
        return safe
    b = raw.get("build")
    if isinstance(b, int) and 0 <= b < cfg.n_features:
        safe["build"] = b
    try:
        safe["price"] = max(1.0, min(500.0, float(raw.get("price", 50.0))))
    except Exception:
        pass
    camps = raw.get("campaigns") or []
    if isinstance(camps, list):
        for c in camps[:cfg.n_pains]:
            if not isinstance(c, dict):
                continue
            tgt = c.get("target", [])
            if isinstance(tgt, int):
                tgt = [tgt]
            tgt = {int(p) for p in tgt if isinstance(p, (int, float)) and 0 <= int(p) < cfg.n_pains}
            try:
                spend = max(0.0, float(c.get("spend", 0.0)))
            except Exception:
                spend = 0.0
            if tgt and spend > 0:
                safe["campaigns"].append({"target": tgt, "spend": spend})
    return safe


# ----------------------------- the agent -----------------------------

class FireworksAgent:
    def __init__(self, cfg: Config, model: str = None, temperature: float = 0.3,
                 max_tokens: int = 2048):
        from openai import OpenAI  # lazy import so sim.py stays dependency-free
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")
        self.client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)
        self.model = model or os.environ.get("FIREWORKS_MODEL", DEFAULT_MODEL)
        self.cfg = cfg
        self.temperature = temperature
        # Reasoning models (glm, deepseek) spend tokens thinking before the JSON
        # action; 2048 leaves room so the ```json block isn't truncated to a no-op.
        self.max_tokens = max_tokens

    def reset(self):
        self.history = []
        self.last_record = None   # {"messages": [...]} for the most recent round (SFT export)

    def act(self, env, obs):
        msgs = [
            {"role": "system", "content": system_prompt(self.cfg)},
            {"role": "user", "content": format_obs(obs, self.cfg, self.history)},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs,
                temperature=self.temperature, max_tokens=self.max_tokens)
            text = resp.choices[0].message.content or ""
        except Exception as e:
            log.warning(f"LLM call failed ({e}); using safe no-op action")
            self.last_record = None
            return {"build": None, "price": obs["price"], "campaigns": []}
        # Record the exact (prompt, completion) so winning trajectories can be
        # exported verbatim as supervised fine-tuning data (STaR / expert iteration).
        self.last_record = {"messages": msgs + [{"role": "assistant", "content": text}]}
        action = validate_action(extract_json(text), self.cfg)
        self.history.append(
            f"  r{obs['round']}: built {action['build']}, price {action['price']:.0f}, "
            f"campaigns {[(sorted(c['target']), round(c['spend'])) for c in action['campaigns']]}")
        return action


# ----------------------------- runner -----------------------------

def main():
    cfg = Config()
    seeds = [1, 2, 3]
    have_key = bool(os.environ.get("FIREWORKS_API_KEY"))
    if not have_key:
        log.warning("FIREWORKS_API_KEY not set — running the SCRIPTED baseline to verify "
                    "wiring. Set the key (and optionally FIREWORKS_MODEL) for the LLM agent.")

    print(f"{'seed':>4} | {'agent':>10} | {'oracle':>10} | {'disc.eff':>8}")
    print("-" * 42)
    for s in seeds:
        world = generate_world(s, cfg)
        if have_key:
            agent = FireworksAgent(cfg)
        else:
            agent = ScriptedExperimenter(world, s)
        prof = run_episode(world, agent)
        oracle = run_episode(world, OraclePolicy(world))
        eff = prof / oracle if oracle > 0 else 0.0
        print(f"{s:>4} | {prof:>10.0f} | {oracle:>10.0f} | {eff:>7.0%}")


if __name__ == "__main__":
    main()
