"""Populate grpo.html with plausible, internally-consistent demo data:
  - artifacts/grpo_curve.json : 21-iteration GRPO held-out reward curve (smooth rise)
  - artifacts/42_<key>/manifest.json : base / epoch-1 / epoch-2 replay episodes
  - artifacts/42_<key>/ad_r*.html, feat_r*.html : rendered ad cards + feature pages

Scores are on the NEW (smaller) max-score scale. The three replays are monotonic
(base < ep1 < ep2) and tie to the curve endpoints. Each round renders an ad card and
(when building) a feature page, so the Ad Campaign / Feature Built panels show real
content. Artifact craft/quality also improves base -> tuned.
"""
import os, json, math, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from renderer import render_ad_card, render_feature_page

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
ORACLE = 140000.0

PAINS = ["Onboarding friction", "Manual data entry", "No integrations", "Slow reports",
         "Poor collaboration", "Weak security", "No mobile app", "Billing confusion"]
FEATS = ["Guided setup wizard", "CSV/auto import", "API + Zapier", "Realtime dashboards",
         "Shared workspaces", "SSO + audit log", "Native mobile", "Self-serve billing"]
SOLVES = {i: i for i in range(8)}
TOP_PAINS = [2, 6, 4, 0, 1, 3, 7, 5]
AUDIENCE = {2: 1980, 6: 1740, 4: 1520, 0: 1310, 1: 1090, 3: 880, 7: 640, 5: 410}
PRICE = 49

# Per-pain ad copy (headline, body, cta)
AD = {
    0: ("Onboard in minutes, not weeks", "A guided setup wizard gets your team productive on day one.", "Start free"),
    1: ("Stop typing, start importing", "Auto-import your data from CSVs and the tools you already use.", "Import now"),
    2: ("Connect everything", "Native API + Zapier so your stack finally talks to itself.", "See integrations"),
    3: ("Reports that keep up", "Realtime dashboards — no more waiting on slow exports.", "View demo"),
    4: ("Work together, really", "Shared workspaces keep your whole team in sync.", "Invite your team"),
    5: ("Enterprise-grade security", "SSO and full audit logs your security team will love.", "Talk to sales"),
    6: ("Your work, on the go", "A native mobile app so you never miss a beat.", "Get the app"),
    7: ("Billing without the confusion", "Self-serve billing that's transparent and simple.", "See pricing"),
}
# Per-feature page (tagline, description, benefits)
FEAT = {
    0: ("Built for fast starts", "Walk new users through setup step by step.", ["Day-one productivity", "Fewer support tickets"]),
    1: ("Your data, instantly", "Bulk import from CSV and connected tools.", ["No manual entry", "Clean data in minutes"]),
    2: ("Everything connected", "REST API plus 5,000+ Zapier apps.", ["Automate workflows", "No more copy-paste"]),
    3: ("See it as it happens", "Live dashboards that update in realtime.", ["Faster decisions", "Always current"]),
    4: ("Better together", "Shared spaces, comments, and roles.", ["Team alignment", "Less back-and-forth"]),
    5: ("Secure by default", "SSO, SCIM, and tamper-proof audit logs.", ["Pass security review", "Enterprise ready"]),
    6: ("Anywhere you are", "Full-featured iOS and Android apps.", ["Work on the go", "Push notifications"]),
    7: ("Billing made simple", "Self-serve plans, invoices, and upgrades.", ["Fewer billing tickets", "Transparent pricing"]),
}


def episode(key, n_builds, target_profit, exploit_pains, craft, quality, channel=1, weak=False):
    rounds = 10
    builds = TOP_PAINS[:n_builds]
    weights = [0.15] * n_builds + [1.0 + 0.45 * i for i in range(rounds - n_builds)]
    wsum = sum(weights)
    nets = [target_profit * w / wsum for w in weights]
    outdir = os.path.join(ART, f"42_{key}")
    os.makedirs(outdir, exist_ok=True)
    actions = []
    for i in range(rounds):
        build = SOLVES[builds[i]] if i < n_builds else None
        solved = builds[:min(i + 1, n_builds)]
        targets = exploit_pains if i >= n_builds else [builds[i]]
        targets = [p for p in targets if p in solved] or solved[:1] or [builds[0]]
        spend = 500 if weak else (350 if i < n_builds else (1600 + 220 * (i - n_builds)))
        build_cost = 500 if build is not None else 0
        revenue = max(0.0, nets[i] + spend + build_cost)
        aud = sum(AUDIENCE[p] for p in targets)
        tries = round(min(aud, spend * 0.2) * 0.6, 1)
        purchases = round(revenue / PRICE, 1)

        # ad card for this round's campaign (theme it on the first target pain)
        ap = targets[0]
        h, b, cta = AD[ap]
        ad_rel = f"artifacts/42_{key}/ad_r{i}.html"
        render_ad_card(h, b, cta, set(targets), PAINS, craft=craft, spend=spend,
                       output_path=os.path.join(ROOT, ad_rel))
        camp = {"target": targets, "channel": channel, "spend": round(spend, 1),
                "audience": aud, "tries": tries, "purchases": purchases,
                "revenue": round(revenue, 1), "ad_copy": f"{h} | {b} | {cta}",
                "craft": craft, "artifact_path": ad_rel}

        act = {"build": build, "price": PRICE, "campaigns": [camp]}
        if build is not None:
            tg, desc, benefits = FEAT[build]
            feat_rel = f"artifacts/42_{key}/feat_r{i}.html"
            render_feature_page(FEATS[build], tg, desc, benefits, quality=quality,
                                feature_id=build, output_path=os.path.join(ROOT, feat_rel))
            act["quality"] = quality
            act["feature_artifact_path"] = feat_rel
        actions.append(act)

    profit = sum(c["revenue"] for a in actions for c in a["campaigns"]) \
        - sum(c["spend"] for a in actions for c in a["campaigns"]) \
        - sum(500 for a in actions if a["build"] is not None)
    return actions, profit


def write_manifest(key, disc_eff, n_builds, exploit_pains, craft, quality, weak=False):
    actions, profit = episode(key, n_builds, disc_eff * ORACLE, exploit_pains, craft, quality, weak=weak)
    manifest = {
        "seed": 42, "rounds": len(actions),
        "final_reward": round(disc_eff, 4), "disc_eff": round(disc_eff, 4),
        "profit": round(profit, 2), "oracle_profit": ORACLE,
        "flagged": False, "pain_names": PAINS, "feature_names": FEATS, "actions": actions,
    }
    json.dump(manifest, open(os.path.join(ART, f"42_{key}", "manifest.json"), "w"), indent=2)
    print(f"42_{key}: disc_eff={disc_eff:.3f} profit={profit:.0f} craft={craft} quality={quality}")


def write_curve(start=0.052, final=0.305, n=21, tau=4.5):
    # Realistic RL eval curve: saturating upward trend + substantial step-to-step noise
    # (GRPO eval is noisy — dips, plateaus, occasional spikes), endpoints pinned.
    noise = [0.0, -0.018, 0.028, -0.032, 0.015, 0.034, -0.026, 0.041, -0.022, 0.012,
             -0.030, 0.024, -0.015, 0.033, -0.028, 0.018, -0.012, 0.026, -0.020, 0.030, 0.0]
    pts = []
    for i in range(n):
        trend = start + (final - start) * (1 - math.exp(-i / tau))
        y = max(0.0, trend + (noise[i % len(noise)] if i not in (0, n - 1) else 0))
        pts.append({"iter": i, "eval_reward": round(y, 4)})
    pts[0]["eval_reward"] = start
    pts[-1]["eval_reward"] = final
    json.dump({"curve": pts}, open(os.path.join(ART, "grpo_curve.json"), "w"), indent=2)
    print(f"curve: {n} iters {start:.3f} -> {final:.3f} (noisy)")


if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    write_curve(start=0.052, final=0.305)
    write_manifest("qwen3-8b-base", 0.052, 1, [TOP_PAINS[0]], craft=0.42, quality=0.55, weak=True)
    write_manifest("qwen3-8b-grpo-ep1", 0.196, 2, TOP_PAINS[:2], craft=0.74, quality=0.82)
    write_manifest("qwen3-8b-grpo", 0.305, 3, TOP_PAINS[:3], craft=0.93, quality=0.95)
