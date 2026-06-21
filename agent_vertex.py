"""
FirmBench — Vertex AI (Google Cloud) single-agent LLM harness.

The Google/GCP counterpart to `agent.py` (Fireworks). It drives the same `FirmEnv`
(from sim.py) with the same `.reset()/.act(env, obs)` interface, so it plugs straight
into the existing runner, the verifier (run.py), and the RFT loop (rft_gcp.py). The
only difference is the inference backend: Gemini (and Gemma) served on **Vertex AI**.

Why a separate file (intentional duplication of the prompt/parse helpers): this is a
*parallel* pipeline meant to evolve independently of the Fireworks pipeline. Keeping it
self-contained — importing only from sim.py — means edits to agent.py/rft.py in another
session can't break it.

Auth (Application Default Credentials — the standard GCP way):
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT=your-project
    export GOOGLE_CLOUD_LOCATION=us-central1        # optional, defaults to us-central1
    export VERTEX_MODEL=gemini-2.0-flash-001        # optional; any tunable Vertex model

Usage:
    python3 agent_vertex.py            # runs the Vertex agent over a few seeds

Without GCP configured it prints setup instructions and falls back to the scripted
baseline so the wiring can still be exercised offline.
"""

import os
import re
import json
import logging

from sim import (Config, generate_world, FirmEnv, OraclePolicy, ScriptedExperimenter,
                 run_episode)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("firmbench.agent_vertex")

# gemini-2.0-flash is cheap, fast, and supports Vertex AI *supervised tuning* — so it
# closes the whole loop (cheap rollouts + managed SFT + serving of the tuned model)
# the same way glm-5p1 does on Fireworks. Gemma open models can also be served from a
# Vertex Model Garden endpoint (see docs/gcp-pipeline.md) and plugged in via VERTEX_MODEL.
DEFAULT_MODEL = "gemini-2.0-flash-001"
DEFAULT_LOCATION = "us-central1"


# ----------------------------- prompting -----------------------------
# (Kept byte-identical in spirit to agent.py so trajectories/datasets are comparable
#  across the Fireworks and Vertex pipelines; duplicated here for isolation.)

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

def _make_client(project: str = None, location: str = None):
    """Build a google-genai client bound to Vertex AI (vertexai=True).

    Uses Application Default Credentials. Raises a clear, actionable error if the SDK
    or project isn't configured, so failures are diagnosable rather than cryptic.
    """
    try:
        from google import genai  # google-genai SDK
    except Exception as e:
        raise RuntimeError(
            "google-genai not installed. `pip install google-genai` "
            "(see requirements-gcp.txt)."
        ) from e

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT not set. Run `gcloud auth application-default login` "
            "and `export GOOGLE_CLOUD_PROJECT=...` (see docs/gcp-pipeline.md)."
        )
    return genai.Client(vertexai=True, project=project, location=location)


class VertexAgent:
    """Single agent driving FirmEnv via Gemini/Gemma on Vertex AI.

    Same interface as agent.FireworksAgent: `.reset()` then `.act(env, obs)` returning a
    validated action dict. Also records the exact (prompt, completion) on `last_record`
    so winning trajectories can be exported as supervised tuning data (STaR / expert
    iteration) by rft_gcp.py.
    """

    def __init__(self, cfg: Config, model: str = None, temperature: float = 0.3,
                 max_tokens: int = 2048, project: str = None, location: str = None):
        self.client = _make_client(project, location)
        self.model = model or os.environ.get("VERTEX_MODEL", DEFAULT_MODEL)
        self.cfg = cfg
        self.temperature = temperature
        # Reasoning-style models can spend tokens before the JSON action; leave room so
        # the ```json block isn't truncated into a no-op.
        self.max_tokens = max_tokens

    def reset(self):
        self.history = []
        self.last_record = None

    def act(self, env, obs):
        from google.genai import types

        sys_text = system_prompt(self.cfg)
        user_text = format_obs(obs, self.cfg, self.history)
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=sys_text,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            text = resp.text or ""
        except Exception as e:
            log.warning(f"Vertex call failed ({e}); using safe no-op action")
            self.last_record = None
            return {"build": None, "price": obs["price"], "campaigns": []}

        # Record in OpenAI chat format (system/user/assistant) — the canonical
        # intermediate the dataset exporter converts to Vertex tuning format.
        self.last_record = {"messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": text},
        ]}
        action = validate_action(extract_json(text), self.cfg)
        self.history.append(
            f"  r{obs['round']}: built {action['build']}, price {action['price']:.0f}, "
            f"campaigns {[(sorted(c['target']), round(c['spend'])) for c in action['campaigns']]}")
        return action


def make_vertex_agent(model=None, temperature=0.3, project=None, location=None):
    """Factory matching rft.make_llm_agent: returns callable(world, seed=0) -> agent.

    The seed is accepted for interface compatibility (Vertex sampling isn't seedable);
    diversity across rollouts comes from temperature > 0.
    """
    def _factory(world, seed=0):
        return VertexAgent(world.cfg, model=model, temperature=temperature,
                           project=project, location=location)
    return _factory


# ----------------------------- runner -----------------------------

def _gcp_ready() -> bool:
    return bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))


def main():
    cfg = Config()
    seeds = [1, 2, 3]
    ready = _gcp_ready()
    if not ready:
        log.warning("GOOGLE_CLOUD_PROJECT not set — running the SCRIPTED baseline to verify "
                    "wiring. Configure GCP (see docs/gcp-pipeline.md) for the Vertex agent.")

    print(f"{'seed':>4} | {'agent':>10} | {'oracle':>10} | {'disc.eff':>8}")
    print("-" * 42)
    for s in seeds:
        world = generate_world(s, cfg)
        if ready:
            agent = VertexAgent(cfg)
        else:
            agent = ScriptedExperimenter(world, s)
        prof = run_episode(world, agent)
        oracle = run_episode(world, OraclePolicy(world))
        eff = prof / oracle if oracle > 0 else 0.0
        print(f"{s:>4} | {prof:>10.0f} | {oracle:>10.0f} | {eff:>7.0%}")


if __name__ == "__main__":
    main()
