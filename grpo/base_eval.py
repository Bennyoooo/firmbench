"""Measure base-model reward on the FirmBench strategy task (serverless inference).
K samples/prompt at temp 0.7 to gauge mean + variance (GRPO needs intra-prompt variance).
Usage: python3 base_eval.py [model] [K]"""
import os, sys, json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, pstdev
from openai import OpenAI
from sim import Config, generate_world
import test_firmbench_grpo as T

MODEL = sys.argv[1] if len(sys.argv) > 1 else "accounts/fireworks/models/glm-5p1"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cfg = T.cfg  # tight-economy cfg from evaluator
client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1")
DATA = sys.argv[3] if len(sys.argv) > 3 else "firmbench_prompts.jsonl"
rows = [json.loads(l) for l in open(DATA)]


def one(args):
    row, k = args
    msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    try:
        r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.7, max_tokens=4096)
        text = r.choices[0].message.content or ""
    except Exception as e:
        return row["ground_truth"]["seed"], 0.0
    gt = row["ground_truth"]
    world = generate_world(int(gt["seed"]), cfg)
    parsed = T.parse_plan(text)
    if not parsed:
        return gt["seed"], 0.0
    b, p, t = parsed
    profit = T.simulate_plan(world, b, p, t)
    return gt["seed"], max(0.0, min(1.0, profit / (gt["oracle"] or 1)))


if __name__ == "__main__":  # guard so eval-protocol test discovery can import safely
    tasks = [(row, k) for row in rows for k in range(K)]
    with ThreadPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(one, tasks))

    by_seed = {}
    for s, sc in out:
        by_seed.setdefault(s, []).append(sc)
    allscores = [sc for _, sc in out]
    print(f"model={MODEL}  K={K}  n={len(out)}")
    print(f"MEAN reward={mean(allscores):.3f}  std={pstdev(allscores):.3f}  "
          f"min={min(allscores):.3f}  max={max(allscores):.3f}")
    # per-prompt variance (does GRPO have signal within a prompt-group?)
    intra = mean(pstdev(v) for v in by_seed.values() if len(v) > 1)
    print(f"mean intra-prompt std={intra:.3f}  (GRPO needs >0)")
