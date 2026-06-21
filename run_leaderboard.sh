#!/usr/bin/env bash
# run_leaderboard.sh — Run multiple models on multiple seeds via HUD eval
# Usage: bash run_leaderboard.sh
#
# Each model×seed saves to artifacts/{seed}_{model}/
# At the end, prints a leaderboard averaged across seeds.

set -e
export PATH="/Users/bennyjiang/Library/Python/3.12/bin:$PATH"

SEEDS=(42 123 7 99 200)

# model_spec format: "agent_type:model_id:display_name"
MODELS=(
  "claude:claude-sonnet-4-6:claude"
  "claude:claude-opus-4-8:claude-opus-4-8"
  "openai:gpt-5:gpt-5"
  "openai:gpt-5.5:gpt-5-5"
  "openai:gpt-5-mini:gpt-5-mini"
  "gemini:gemini-3.5-flash:gemini"
)

echo "============================================"
echo "FirmBench Leaderboard Run"
echo "${#MODELS[@]} models × ${#SEEDS[@]} seeds = $(( ${#MODELS[@]} * ${#SEEDS[@]} )) runs"
echo "============================================"
echo ""

for model_spec in "${MODELS[@]}"; do
  IFS=: read -r agent model safe_name <<< "$model_spec"
  echo "=== $safe_name ($model via $agent) ==="

  for seed in "${SEEDS[@]}"; do
    outdir="artifacts/${seed}_${safe_name}"

    # Skip if already run
    if [ -f "$outdir/manifest.json" ]; then
      reward=$(python3 -c "import json; print(json.load(open('$outdir/manifest.json'))['final_reward'])" 2>/dev/null)
      echo "  seed $seed: CACHED (reward=$reward)"
      continue
    fi

    echo -n "  seed $seed: running..."
    rm -rf "artifacts/$seed"
    python3 export_world.py "$seed" 2>/dev/null

    result=$(hud eval tasks.py "$agent" -m "$model" --task-ids "market_discovery_seed${seed}" -y --max-steps 120 2>&1)
    reward=$(echo "$result" | grep -oE '[0-9]+\.[0-9]+' | tail -1)

    if [ -d "artifacts/$seed" ]; then
      cp -r "artifacts/$seed" "$outdir"
      echo " reward=$reward ✓"
    else
      echo " FAILED ✗"
    fi
  done
  echo ""
done

echo "============================================"
echo "=== LEADERBOARD (averaged across seeds) ==="
echo "============================================"

python3 -c "
import json, os
from collections import defaultdict

model_results = defaultdict(list)

for d in sorted(os.listdir('artifacts')):
    mpath = f'artifacts/{d}/manifest.json'
    if not os.path.isfile(mpath): continue
    # Parse: {seed}_{model} format
    parts = d.split('_', 1)
    if len(parts) != 2: continue
    seed_str, model = parts
    if not seed_str.isdigit(): continue

    m = json.load(open(mpath))
    profit = m.get('profit', m.get('reported_profit', 0))
    reward = m.get('final_reward', 0)
    model_results[model].append({'seed': int(seed_str), 'profit': profit, 'reward': reward})

print(f'{\"Model\":<25s} {\"Seeds\":>5s} {\"Avg Reward\":>10s} {\"Avg Profit\":>12s} {\"Best\":>8s} {\"Worst\":>8s}')
print('-' * 70)
ranked = []
for model, runs in model_results.items():
    n = len(runs)
    avg_r = sum(r['reward'] for r in runs) / n
    avg_p = sum(r['profit'] for r in runs) / n
    best = max(r['reward'] for r in runs)
    worst = min(r['reward'] for r in runs)
    ranked.append((avg_r, model, n, avg_p, best, worst))

for avg_r, model, n, avg_p, best, worst in sorted(ranked, reverse=True):
    print(f'{model:<25s} {n:>5d} {avg_r:>10.3f} {avg_p:>12,.0f} {best:>8.3f} {worst:>8.3f}')
"
