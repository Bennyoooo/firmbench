"""FirmBench GRPO evaluator for Fireworks RFT (eval-protocol).

Self-contained (Fireworks builds + runs this remotely). Single-turn STRATEGY task:
the model is given completed market research (pain audiences + the feature that solves
each pain + economics) and must output a product PLAN — which features to build, the
price, and which pains to target — to maximize total profit. The reward is the
oracle-normalized full-episode profit of executing that plan in the deterministic sim.

This is a learnable, non-degenerate reward (feature selection under budget, pricing vs
churn, targeting) — distinct from the degenerate round-0-profit reward. It exercises the
firm-operation skill; multi-turn experimental discovery is the next iteration.
"""

import json

from eval_protocol.models import EvaluationRow, EvaluateResult, Message
from eval_protocol.pytest import evaluation_test

from sim import Config, generate_world, FirmEnv

# Tighter economy: build-all is unaffordable (8 x $500 > $3500), so the model must
# SELECT the highest-value features within budget and price well — real strategy,
# not the trivial "build everything" optimum.
cfg = Config(starting_cash=3500.0, build_cost=500.0)


def build_prompt(world) -> str:
    """Market-research summary the model gets (audiences + solves mapping + economics)."""
    pains = sorted(range(cfg.n_pains), key=lambda p: world.pain_popularity[p], reverse=True)
    lines = ["You run a software firm. Market research is COMPLETE. Customer needs (pains),"
             " ranked by how many customers have them, and the feature that solves each:"]
    for p in pains:
        lines.append(f"  - pain {p}: {world.pain_popularity[p]} customers, solved by feature {world.solves[p]}")
    lines.append(f"\nEconomics: starting cash ${int(cfg.starting_cash)}; each feature costs "
                 f"${int(cfg.build_cost)} to build; {cfg.horizon} rounds; customers who buy "
                 f"SUBSCRIBE (revenue recurs each round, but they churn if the price is too high "
                 f"for their willingness-to-pay). Willingness-to-pay is lognormal, median ~$49.")
    lines.append("\nReason in at most ~150 words, then output ONE JSON plan to maximize TOTAL "
                 "profit over all rounds. You MUST end your reply with the json block:")
    lines.append('```json\n{"build": [feature_ids], "price": dollars, "target": [pain_ids]}\n```')
    lines.append("Tips: building a feature only pays off if you target the pain it solves and "
                 "enough customers have that pain. You CANNOT afford every feature — pick the "
                 "highest-value ones and keep cash for marketing. Price to balance how many "
                 "subscribe against how many churn.")
    return "\n".join(lines)


def parse_plan(text: str):
    """Extract {build:[...], price, target:[...]} from the model's reply."""
    import re
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not blocks:
        s, e = text.rfind("{"), text.rfind("}")
        blocks = [text[s:e + 1]] if (s != -1 and e > s) else []
    for b in reversed(blocks):
        try:
            d = json.loads(b)
        except Exception:
            continue
        build = d.get("build", [])
        if isinstance(build, int):
            build = [build]
        build = [int(x) for x in build if isinstance(x, (int, float)) and 0 <= int(x) < cfg.n_features]
        try:
            price = max(1.0, min(500.0, float(d.get("price", 50.0))))
        except Exception:
            price = 50.0
        tgt = d.get("target", [])
        if isinstance(tgt, int):
            tgt = [tgt]
        target = {int(x) for x in tgt if isinstance(x, (int, float)) and 0 <= int(x) < cfg.n_pains}
        return build, price, target
    return None


def simulate_plan(world, build_ids, price, target_pains) -> float:
    """Execute the plan deterministically: build the listed features (one per round until
    done), then run exploit campaigns on the targeted pains with available cash each round.
    Returns total episode profit."""
    env = FirmEnv(world)
    env.reset()
    to_build = list(dict.fromkeys(build_ids))  # de-dup, keep order
    done = False
    while not done:
        f = to_build.pop(0) if to_build else None
        # reserve cash for remaining builds so we don't go bankrupt mid-plan
        reserve = cfg.build_cost * len(to_build)
        build_now = cfg.build_cost if f is not None else 0.0
        spend = max(0.0, env.cash - build_now - reserve)
        camps = [{"target": set(target_pains), "spend": spend}] if (target_pains and spend > 0) else []
        _obs, _profit, done, _ = env.step({"build": f, "price": price, "campaigns": camps})
    return env.total_profit


@evaluation_test(
    input_dataset=["firmbench_prompts.jsonl"],
    completion_params=[{
        "model": "accounts/fireworks/models/glm-5p1",
        "temperature": 0.7,
        "max_tokens": 4096,
    }],
    mode="pointwise",
)
def test_firmbench_grpo(row: EvaluationRow) -> EvaluationRow:
    gt = row.ground_truth or {}
    seed = int(gt.get("seed", 1))
    oracle = float(gt.get("oracle", 140000.0)) or 140000.0
    world = generate_world(seed, cfg)
    text = (row.messages[-1].content or "") if row.messages else ""
    parsed = parse_plan(text)
    if parsed is None:
        row.evaluation_result = EvaluateResult(score=0.0, reason="no valid JSON plan")
        return row
    build, price, target = parsed
    profit = simulate_plan(world, build, price, target)
    # Oracle-normalized profit, NOT clipped to [0,1]: clipping collapses all losing
    # plans to 0.0 and all near-optimal plans to ~1.0, which destroys the within-group
    # reward variance GRPO needs (a flat group is filtered out -> no training). Keeping
    # the raw ratio (mild bounds for outliers) preserves the ordering signal; GRPO
    # normalizes advantages within each prompt-group anyway.
    score = max(-1.0, min(1.5, profit / oracle))
    row.evaluation_result = EvaluateResult(
        score=score, reason=f"profit ${profit:.0f} / oracle ${oracle:.0f} = {score:.3f}")
    return row
