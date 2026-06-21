"""Plot the multi-agent (team) leaderboard from ma_eval_results/seed*.json.

Each seed JSON is {model_name: team_disc_eff}. This averages each model across all available
seeds, prints an ASCII bar chart, and writes a self-contained HTML bar chart
(ma_eval_results/leaderboard.html). No matplotlib needed (it isn't installed here).

    python3 plot_leaderboard.py
"""
import glob
import json
import os

RESULTS_DIR = "ma_eval_results"


def load():
    per_model = {}      # model -> {seed: score}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "seed*.json"))):
        seed = os.path.basename(path)[4:-5]
        for model, score in json.load(open(path)).items():
            per_model.setdefault(model, {})[seed] = score
    return per_model


def main():
    per_model = load()
    if not per_model:
        print("no results in", RESULTS_DIR); return
    rows = []
    for model, by_seed in per_model.items():
        vals = list(by_seed.values())
        rows.append((model, sum(vals) / len(vals), len(vals), by_seed))
    rows.sort(key=lambda r: r[1], reverse=True)
    seeds = sorted({s for _, _, _, bs in rows for s in bs})

    print("=" * 64)
    print(f"FirmBench multi-agent (team) leaderboard — disc.eff = profit / theoretical_max")
    print(f"seeds: {', '.join(seeds)}  (avg of {len(seeds)} seed(s))")
    print("=" * 64)
    width = max((r[1] for r in rows), default=1.0) or 1.0
    for model, avg, n, by_seed in rows:
        bar = "█" * int(round(40 * avg / width))
        print(f"  {model:<20} {avg:6.3f}  {bar}")
    print("-" * 64)
    print("per-seed:")
    for model, avg, n, by_seed in rows:
        cells = "  ".join(f"s{s}={by_seed[s]:.3f}" for s in seeds if s in by_seed)
        print(f"  {model:<20} {cells}")

    # self-contained HTML bar chart
    bars = "\n".join(
        f'<div class="row"><div class="lbl">{model}</div>'
        f'<div class="bar" style="width:{max(1,round(100*avg/width))}%"></div>'
        f'<div class="val">{avg:.3f}</div></div>'
        for model, avg, n, by_seed in rows)
    html = f"""<!doctype html><meta charset=utf-8><title>FirmBench multi-agent leaderboard</title>
<style>body{{background:#0a0b0f;color:#c8cad0;font-family:-apple-system,sans-serif;padding:32px}}
h1{{color:#fff;font-size:20px}}.sub{{color:#6b7084;font-size:13px;margin-bottom:24px}}
.row{{display:flex;align-items:center;gap:12px;margin:7px 0}}
.lbl{{flex:0 0 180px;text-align:right;font-size:13px}}
.bar{{height:24px;background:linear-gradient(90deg,#38ef7d,#667eea);border-radius:5px;min-width:2px}}
.val{{font-variant-numeric:tabular-nums;font-weight:600;font-size:13px}}</style>
<h1>🏢🤝 FirmBench — multi-agent (team) leaderboard</h1>
<div class=sub>team disc.eff = profit / theoretical_max (Coordinator-dispatch, HUD gateway) ·
seeds: {', '.join(seeds)}</div>
{bars}"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    open(os.path.join(RESULTS_DIR, "leaderboard.html"), "w").write(html)
    print("-" * 64)
    print(f"wrote {RESULTS_DIR}/leaderboard.html  (open in a browser)")


if __name__ == "__main__":
    main()
