"""Export world data as JSON for the visualization dashboard.

Usage:
    python3 export_world.py [seed] [--phase-a]

Writes artifacts/{seed}/world.json with population stats, distributions,
segment breakdowns, and the hidden mapping — everything the world.html
dashboard needs.
"""

import json
import os
import sys
import random
from collections import Counter
from pathlib import Path

from sim import Config, generate_world


def export_world(seed: int, cfg: Config = None) -> dict:
    """Generate a world and extract all visualization data."""
    cfg = cfg or Config.phase_a()
    w = generate_world(seed, cfg)

    users = w.users
    n = len(users)

    # WTP distribution (sorted for histogram)
    wtps = sorted(u.wtp for u in users)
    wtp_buckets = _histogram(wtps, n_bins=30)

    # Pain popularity
    pain_pop = [{"id": i, "name": w.pain_names[i], "count": w.pain_popularity[i],
                 "pct": round(w.pain_popularity[i] / n * 100, 1)}
                for i in range(cfg.n_pains)]
    pain_pop.sort(key=lambda x: x["count"], reverse=True)

    # Pain count distribution (how many pains per user)
    pain_counts = Counter(len(u.pains) for u in users)

    # Hidden mapping (solves)
    mapping = [{"pain_id": p, "pain_name": w.pain_names[p],
                "feature_id": f, "feature_name": w.feature_names[f],
                "pain_popularity": w.pain_popularity[p]}
               for p, f in w.solves.items()]

    # Segments (Phase A)
    segments = []
    if w.segments:
        seg_counts = Counter(u.segment_id for u in users)
        for i, s in enumerate(w.segments):
            seg_users = [u for u in users if u.segment_id == i]
            seg_wtps = [u.wtp for u in seg_users]
            # Pain affinity for this segment
            seg_pain_counts = Counter()
            for u in seg_users:
                for p in u.pains:
                    seg_pain_counts[p] += 1
            top_pains = seg_pain_counts.most_common(3)

            segments.append({
                "id": i,
                "weight": round(s.weight, 3),
                "count": seg_counts.get(i, 0),
                "pct": round(seg_counts.get(i, 0) / n * 100, 1),
                "channel_pref": s.channel_pref,
                "elasticity_mu": round(s.elasticity_mu, 2),
                "quality_bar": round(s.quality_bar, 2),
                "churn_base": round(s.churn_base, 3),
                "wtp_median": round(sorted(seg_wtps)[len(seg_wtps) // 2], 0) if seg_wtps else 0,
                "wtp_p25": round(seg_wtps[len(seg_wtps) // 4], 0) if seg_wtps else 0,
                "wtp_p75": round(seg_wtps[3 * len(seg_wtps) // 4], 0) if seg_wtps else 0,
                "top_pains": [{"id": p, "name": w.pain_names[p], "count": c}
                              for p, c in top_pains],
            })

    # Channel distribution
    ch_counts = Counter(u.channel_pref for u in users)
    channels = [{"id": i, "count": ch_counts.get(i, 0),
                 "pct": round(ch_counts.get(i, 0) / n * 100, 1)}
                for i in range(cfg.n_channels)]

    # Sample users for scatter plot (200 random, deterministic)
    rng = random.Random(seed + 999)
    sample_indices = rng.sample(range(n), min(200, n))
    user_sample = [{"pains": sorted(users[i].pains),
                    "n_pains": len(users[i].pains),
                    "wtp": round(users[i].wtp, 1),
                    "segment": users[i].segment_id,
                    "channel": users[i].channel_pref,
                    "elasticity": round(users[i].elasticity, 2) if users[i].elasticity else None}
                   for i in sample_indices]

    # Elasticity distribution
    elasticities = [u.elasticity for u in users if u.elasticity is not None]
    elast_buckets = _histogram(elasticities, n_bins=20) if elasticities else []

    return {
        "seed": seed,
        "n_users": n,
        "n_pains": cfg.n_pains,
        "n_features": cfg.n_features,
        "n_segments": cfg.n_segments if cfg.use_segments else 0,
        "n_channels": cfg.n_channels if cfg.use_channels else 0,
        "horizon": cfg.horizon,
        "starting_cash": cfg.starting_cash,
        "build_cost": cfg.build_cost,
        "pain_names": w.pain_names,
        "feature_names": w.feature_names,
        "pain_popularity": pain_pop,
        "pain_count_distribution": {str(k): v for k, v in sorted(pain_counts.items())},
        "wtp_distribution": wtp_buckets,
        "wtp_percentiles": {
            "p10": round(wtps[n // 10], 0),
            "p25": round(wtps[n // 4], 0),
            "p50": round(wtps[n // 2], 0),
            "p75": round(wtps[3 * n // 4], 0),
            "p90": round(wtps[9 * n // 10], 0),
        },
        "hidden_mapping": mapping,
        "segments": segments,
        "channels": channels,
        "user_sample": user_sample,
        "elasticity_distribution": elast_buckets,
        "config": {
            "use_segments": cfg.use_segments,
            "use_channels": cfg.use_channels,
            "use_elasticity": cfg.use_elasticity,
            "use_quality_bar": cfg.use_quality_bar,
            "use_retention": cfg.use_retention,
        },
    }


def _histogram(values, n_bins=20):
    """Create histogram buckets from a list of values."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(values)}]
    step = (hi - lo) / n_bins
    buckets = []
    for i in range(n_bins):
        b_lo = lo + i * step
        b_hi = lo + (i + 1) * step
        count = sum(1 for v in values if b_lo <= v < b_hi) if i < n_bins - 1 else \
                sum(1 for v in values if b_lo <= v <= b_hi)
        buckets.append({"lo": round(b_lo, 1), "hi": round(b_hi, 1), "count": count})
    return buckets


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    use_phase_a = "--phase-a" in sys.argv or True  # default to Phase A

    cfg = Config.phase_a() if use_phase_a else Config()
    data = export_world(seed, cfg)

    out_dir = f"artifacts/{seed}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "world.json")
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Exported world data to {out_path}")
    print(f"  {data['n_users']} users, {data['n_segments']} segments, "
          f"{data['n_channels']} channels")


if __name__ == "__main__":
    main()
