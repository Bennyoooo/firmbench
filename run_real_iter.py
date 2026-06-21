"""Driver for ONE real RFT iteration on glm-5p1: base eval + rollouts + dataset export.
Step 1 of the real run (the expensive inference half). firectl SFT is run separately."""
import os, json, time
from sim import Config, generate_world, OraclePolicy, ScriptedExperimenter
from run import Verifier
from rft import make_llm_agent, rollout_and_filter, evaluate, write_jsonl

MODEL = "accounts/fireworks/models/glm-5p1"
cfg = Config(); v = Verifier()
train_seeds = list(range(1, 9))      # 8 training worlds
eval_seeds = list(range(100, 104))   # 4 disjoint held-out worlds
N_ROLLOUTS = 3
os.makedirs("rft_out", exist_ok=True)

t0 = time.time()
print(f"[base eval] {MODEL} on {len(eval_seeds)} held-out worlds...", flush=True)
base_eval = evaluate(eval_seeds, make_llm_agent(MODEL, temperature=0.0), cfg, v)
print(f"[base eval] mean_reward={base_eval['mean_reward']} flagged={base_eval['flagged']}/{base_eval['n']} "
      f"({time.time()-t0:.0f}s)", flush=True)

print(f"[rollout] {N_ROLLOUTS} rollouts x {len(train_seeds)} worlds @ temp 0.7...", flush=True)
t1 = time.time()
dataset, stats = rollout_and_filter(train_seeds, make_llm_agent(MODEL, temperature=0.7),
                                    cfg, v, N_ROLLOUTS)
kept = sum(1 for s in stats if s["rounds_added"] > 0)
ds_path = write_jsonl(dataset, "rft_out/sft_iter1.jsonl")
print(f"[rollout] kept {kept}/{len(train_seeds)} worlds, {len(dataset)} turns -> {ds_path} "
      f"({time.time()-t1:.0f}s)", flush=True)
for s in stats:
    print(f"   seed {s['seed']}: best_reward={s['best_reward']} mean={s['mean_reward']} "
          f"pos={s['kept_positive']} turns+={s['rounds_added']}", flush=True)

json.dump({"base_eval": base_eval, "stats": stats, "model": MODEL,
           "train_seeds": train_seeds, "eval_seeds": eval_seeds},
          open("rft_out/iter1_meta.json", "w"), indent=2)
print(f"[done] total {time.time()-t0:.0f}s", flush=True)
