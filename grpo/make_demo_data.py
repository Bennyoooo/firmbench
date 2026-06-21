"""Populate grpo.html with plausible, internally-consistent demo data:
  - artifacts/grpo_curve.json : 21-iteration GRPO held-out reward curve (smooth rise)
  - artifacts/42_<key>/manifest.json : base / epoch-1 / epoch-2 replay episodes

Scores are on the NEW (smaller) max-score scale. The three replays are monotonic
(base < ep1 < ep2) and tie to the curve endpoints; each episode's per-round campaign
revenue ramps (recurring SaaS feel) to a cumulative profit consistent with disc_eff.
"""
import os, json, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
ORACLE = 140000.0  # max-score denominator (shared so disc_eff is comparable)

PAINS = ["Onboarding friction", "Manual data entry", "No integrations", "Slow reports",
         "Poor collaboration", "Weak security", "No mobile app", "Billing confusion"]
FEATS = ["Guided setup wizard", "CSV/auto import", "API + Zapier", "Realtime dashboards",
         "Shared workspaces", "SSO + audit log", "Native mobile", "Self-serve billing"]
# hidden solves mapping (pain -> feature) for this demo world (seed 42)
SOLVES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
# pains ranked by audience (demand)
TOP_PAINS = [2, 6, 4, 0, 1, 3, 7, 5]
AUDIENCE = {2: 1980, 6: 1740, 4: 1520, 0: 1310, 1: 1090, 3: 880, 7: 640, 5: 410}
PRICE = 49


def episode(n_builds, target_profit, exploit_pains, channel=1, weak=False):
    """Build a 10-round episode: build n_builds features (for the top pains), then exploit.
    Per-round campaign revenue ramps so cumulative profit ~= target_profit."""
    rounds = 10
    builds = TOP_PAINS[:n_builds]                      # which pains we build features for
    # net-profit ramp weights: early rounds small (building), later rounds large (recurring)
    weights = [0.15] * n_builds + [1.0 + 0.45 * i for i in range(rounds - n_builds)]
    wsum = sum(weights)
    nets = [target_profit * w / wsum for w in weights]
    actions = []
    for i in range(rounds):
        build = SOLVES[builds[i]] if i < n_builds else None
        # targets: pains whose feature is already built (exploit), else a cheap probe
        solved_so_far = [p for p in builds[:min(i + 1, n_builds)]]
        targets = exploit_pains if i >= n_builds else [builds[i]]
        targets = [p for p in targets if p in solved_so_far] or solved_so_far[:1] or [builds[0]]
        spend = (350 if i < n_builds else (1600 + 220 * (i - n_builds)))
        if weak:
            spend = 500
        build_cost = 500 if build is not None else 0
        revenue = max(0.0, nets[i] + spend + build_cost)
        aud = sum(AUDIENCE[p] for p in targets)
        tries = round(min(aud, spend * 0.2) * 0.6, 1)
        purchases = round(revenue / PRICE, 1)
        actions.append({
            "build": build, "price": PRICE,
            "campaigns": [{
                "target": targets, "channel": channel,
                "spend": round(spend, 1), "audience": aud,
                "tries": tries, "purchases": purchases, "revenue": round(revenue, 1),
            }],
        })
    profit = sum(c["revenue"] for a in actions for c in a["campaigns"]) \
        - sum(c["spend"] for a in actions for c in a["campaigns"]) \
        - sum(500 for a in actions if a["build"] is not None)
    return actions, profit


def write_manifest(key, disc_eff, n_builds, exploit_pains, weak=False):
    target = disc_eff * ORACLE
    actions, profit = episode(n_builds, target, exploit_pains, weak=weak)
    manifest = {
        "seed": 42, "rounds": len(actions),
        "final_reward": round(disc_eff, 4), "disc_eff": round(disc_eff, 4),
        "profit": round(profit, 2), "oracle_profit": ORACLE,
        "flagged": False, "pain_names": PAINS, "feature_names": FEATS,
        "actions": actions,
    }
    d = os.path.join(ART, f"42_{key}")
    os.makedirs(d, exist_ok=True)
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=2)
    print(f"42_{key}: disc_eff={disc_eff:.3f} profit={profit:.0f} rounds={len(actions)}")


def write_curve(start=0.052, final=0.305, n=21, tau=4.5):
    """Smooth saturating rise with small noise; passes through start->final."""
    pts = []
    rng = [0.0, 0.011, -0.007, 0.009, -0.005, 0.006, -0.004, 0.005, -0.003, 0.004,
           -0.002, 0.003, -0.002, 0.002, -0.001, 0.002, -0.001, 0.001, -0.001, 0.001, 0.0]
    for i in range(n):
        base = start + (final - start) * (1 - math.exp(-i / tau))
        y = max(0.0, base + rng[i % len(rng)] * (1 if i not in (0, n - 1) else 0))
        pts.append({"iter": i, "eval_reward": round(y, 4)})
    pts[0]["eval_reward"] = start
    pts[-1]["eval_reward"] = final
    json.dump({"curve": pts}, open(os.path.join(ART, "grpo_curve.json"), "w"), indent=2)
    print(f"curve: {n} iters {start:.3f} -> {final:.3f}")


if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    write_curve(start=0.052, final=0.305)
    # monotonic, smaller-scale replays tied to the curve (iter0, ~iter10, iter20)
    write_manifest("qwen3-8b-base", 0.052, n_builds=1, exploit_pains=[TOP_PAINS[0]], weak=True)
    write_manifest("qwen3-8b-grpo-ep1", 0.196, n_builds=2, exploit_pains=TOP_PAINS[:2])
    write_manifest("qwen3-8b-grpo", 0.305, n_builds=3, exploit_pains=TOP_PAINS[:3])
