"""
FirmBench — Real RFT: rejection-sampling fine-tuning (a.k.a. expert iteration / STaR).

This is the step that turns FirmBench from an *eval* into an *RL environment*: we
take a small open model on Fireworks, run episodes, KEEP ONLY the trajectories the
verifier rewards, fine-tune the model on those, and re-evaluate. Across iterations
the eval curve bends upward — the model learns the meta-skill of market discovery.

The loop (one iteration):
  1. ROLLOUT   — for each training world, sample N rollouts with the current model
                 (temperature > 0 for diversity). Record (prompt, completion) per round.
  2. GRADE     — score each rollout with the held-out Verifier (run.py). This curates
                 the training set: only non-cheating, profitable episodes survive.
  3. REJECT    — keep the best non-flagged, positive-reward rollout per world
                 (best-of-N rejection sampling).
  4. DATASET   — flatten the winning rollouts into a chat-format JSONL SFT dataset.
  5. FINE-TUNE — supervised fine-tune the base model on Fireworks (firectl) -> new model.
  6. EVALUATE  — score the new model on disjoint held-out worlds. Append to the curve.

Two ways to run:
  python3 rft.py --selftest      # offline, no network: a MockModel whose skill rises
                                  # each iteration drives the WHOLE machinery and prints
                                  # a bending curve. Validates rollout->filter->dataset->eval.

  python3 rft.py --run           # the real thing. Needs FIREWORKS_API_KEY + firectl.
                                  #   export FIREWORKS_API_KEY=...
                                  #   firectl signin   (or set FIREWORKS_API_KEY for firectl)
                                  # Generates real rollouts, fine-tunes on Fireworks,
                                  # deploys, and re-evaluates each iteration.

Flags: --iterations, --train-seeds, --eval-seeds, --rollouts, --model, --out.
"""

import os
import json
import time
import random
import shutil
import argparse
import subprocess
from statistics import mean
from concurrent.futures import ThreadPoolExecutor

from sim import (Config, generate_world, FirmEnv, OraclePolicy, NaivePolicy,
                 ScriptedExperimenter, best_price_for)
from run import Verifier
from agent import system_prompt, format_obs

# glm-5p1 is the only current serverless model that closes the whole cheap loop:
# serverless inference (cheap rollouts) + supervised LoRA tuning + *serverless LoRA
# serving* (so the fine-tuned adapter is served without a paid dedicated GPU
# deployment). deepseek-v4-flash / kimi / nemotron train LoRA but can't serve it
# serverlessly; gpt-oss-20b is RL-tunable only (no supervised SFT). It reasons before
# emitting the action, so the agent uses a larger max_tokens to avoid truncated JSON.
DEFAULT_MODEL = os.environ.get(
    "FIREWORKS_MODEL", "accounts/fireworks/models/glm-5p1")
MAX_PARALLEL = int(os.environ.get("RFT_PARALLEL", "8"))


# ----------------------------- episode recorder -----------------------------

def action_to_completion(action: dict) -> str:
    """Serialize an action dict into the same ```json block the model is asked to emit.
    Used so MockModel trajectories are byte-compatible with real LLM trajectories."""
    payload = {
        "build": action.get("build"),
        "price": round(float(action.get("price", 50.0)), 2),
        "campaigns": [
            {"target": sorted(c.get("target", set())), "spend": round(float(c.get("spend", 0.0)), 2)}
            for c in (action.get("campaigns") or [])
        ],
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def play_episode(world, agent, cfg):
    """Run one full episode. Returns (records, action_log, total_profit).

    records   : list of {"messages": [...]} per round (for SFT export)
    action_log: per-round dicts the Verifier replays on held-out users
    """
    env = FirmEnv(world)
    agent.reset()
    obs = env.reset()
    done = False
    records, action_log = [], []
    while not done:
        action = agent.act(env, obs)
        rec = getattr(agent, "last_record", None)
        if rec is not None:
            records.append(rec)
        action_log.append({
            "build": action.get("build"),
            "price": action.get("price", env.price),
            "campaigns": [
                {"target": sorted(c.get("target", set())), "spend": float(c.get("spend", 0.0))}
                for c in (action.get("campaigns") or [])
            ],
        })
        obs, _reward, done, _ = env.step(action)
    return records, action_log, env.total_profit


def grade_episode(world, action_log, profit, verifier):
    return verifier.grade(world, {
        "action_log": action_log,
        "reported_profit": profit,
        "total_profit": profit,
    })


# ----------------------------- agents -----------------------------

class MockModel:
    """A stand-in for the LLM whose competence is controlled by `skill` in [0,1].

    Each round it plays the disciplined ScriptedExperimenter action with prob `skill`,
    otherwise a NaivePolicy (random) action. Raising `skill` simulates the effect of
    fine-tuning on winning trajectories. It records (prompt, completion) exactly like
    the real agent, so the dataset/filtering/eval code is exercised end-to-end offline.
    """
    def __init__(self, world, skill=0.0, seed=0, temperature=0.0):
        self.w = world
        self.cfg = world.cfg
        self.skill = skill
        self.seed = seed
        self.temperature = temperature

    def reset(self):
        self.good = ScriptedExperimenter(self.w, self.seed)
        self.bad = NaivePolicy(self.w, self.seed)
        self.good.reset(); self.bad.reset()
        self.rng = random.Random(self.seed * 7919 + int(self.skill * 1000))
        self.history = []
        self.last_record = None

    def act(self, env, obs):
        use_good = self.rng.random() < self.skill
        # at eval (temperature 0) be deterministic: skill is the floor
        action = (self.good if use_good else self.bad).act(env, obs)
        self.last_record = {"messages": [
            {"role": "system", "content": system_prompt(self.cfg)},
            {"role": "user", "content": format_obs(obs, self.cfg, self.history)},
            {"role": "assistant", "content": action_to_completion(action)},
        ]}
        return action


def make_llm_agent(model, temperature):
    """Factory for the real Fireworks agent bound to a given model + temperature."""
    from agent import FireworksAgent

    def _factory(world, seed=0):
        a = FireworksAgent(world.cfg, model=model, temperature=temperature)
        return a
    return _factory


# ----------------------------- rollout + rejection sampling -----------------------------

def _one_rollout(args):
    s, k, make_agent, cfg, verifier = args
    world = generate_world(s, cfg)
    agent = make_agent(world, seed=s * 1000 + k)
    records, action_log, profit = play_episode(world, agent, cfg)
    g = grade_episode(world, action_log, profit, verifier)
    return s, {"reward": g["reward"], "flagged": g["flagged"],
               "profit": profit, "records": records}


def rollout_and_filter(train_seeds, make_agent, cfg, verifier, n_rollouts):
    """For each world: sample n_rollouts (in parallel), grade, keep the best
    non-flagged trajectory. Rollouts are IO-bound LLM calls, so we fan them out.

    Returns (dataset_records, stats) where dataset_records is a flat list of
    per-round {"messages": [...]} from the winning trajectories.
    """
    tasks = [(s, k, make_agent, cfg, verifier)
             for s in train_seeds for k in range(n_rollouts)]
    by_seed = {s: [] for s in train_seeds}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        for s, res in ex.map(_one_rollout, tasks):
            by_seed[s].append(res)

    dataset, stats = [], []
    for s in train_seeds:
        graded = by_seed[s]
        # rejection sampling: never train on a flagged (cheating) episode. Prefer
        # profitable episodes; if a world produced none (cold start with a weak base
        # model), keep its single best non-flagged rollout so the model still gets a
        # gradient toward its own least-bad behavior.
        clean = [r for r in graded if not r["flagged"]]
        positive = [r for r in clean if r["reward"] > 0]
        pool = positive if positive else clean
        pool.sort(key=lambda r: r["reward"], reverse=True)
        best = pool[0] if pool else None
        if best:
            dataset.extend(best["records"])
        stats.append({
            "seed": s,
            "best_reward": round(best["reward"], 1) if best else None,
            "mean_reward": round(mean(r["reward"] for r in graded), 1) if graded else None,
            "kept_positive": len(positive),
            "rounds_added": len(best["records"]) if best else 0,
        })
    return dataset, stats


def _one_eval(args):
    s, make_agent, cfg, verifier = args
    world = generate_world(s, cfg)
    agent = make_agent(world, seed=s)
    _records, action_log, profit = play_episode(world, agent, cfg)
    g = grade_episode(world, action_log, profit, verifier)
    return g["reward"], g["flagged"]


def evaluate(eval_seeds, make_agent, cfg, verifier):
    """Greedy eval (the factory should produce temperature~0 agents). Mean held-out reward."""
    tasks = [(s, make_agent, cfg, verifier) for s in eval_seeds]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        out = list(ex.map(_one_eval, tasks))
    rewards = [r for r, _ in out]
    flags = sum(int(f) for _, f in out)
    return {"mean_reward": round(mean(rewards), 1), "flagged": flags, "n": len(eval_seeds)}


# ----------------------------- dataset export -----------------------------

def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ----------------------------- Fireworks fine-tuning (firectl) -----------------------------

def _firectl_available():
    return shutil.which("firectl") is not None


def _firectl_base():
    """firectl invocation prefix with auth from the environment (no `signin` needed)."""
    cmd = ["firectl"]
    key = os.environ.get("FIREWORKS_API_KEY")
    acct = os.environ.get("FIREWORKS_ACCOUNT")
    if key:
        cmd += ["--api-key", key]
    if acct:
        cmd += ["--account-id", acct]
    return cmd


def finetune_firectl(dataset_path, base_model, tag, epochs=3, lora_rank=8):
    """Supervised LoRA fine-tune `base_model` on `dataset_path` via firectl; return the
    fine-tuned model id. glm-5p1 supports serverless LoRA serving, so the returned id is
    usable for inference directly (no paid dedicated deployment).

    Requires firectl on PATH and FIREWORKS_API_KEY (+ FIREWORKS_ACCOUNT) in the env.
    """
    if not _firectl_available():
        raise RuntimeError(
            "firectl not found. Install it (https://docs.fireworks.ai/tools-sdks/firectl), "
            "then set FIREWORKS_API_KEY (+ FIREWORKS_ACCOUNT). Skipping real fine-tune.")

    fc = _firectl_base()
    dataset_id = f"firmbench-{tag}"
    out_model = f"firmbench-rft-{tag}"
    acct = os.environ.get("FIREWORKS_ACCOUNT", "")

    def run(cmd):
        print("    $ " + " ".join(c if c != os.environ.get("FIREWORKS_API_KEY") else "***"
                                  for c in cmd))
        subprocess.run(cmd, check=True)

    # 1) upload dataset (chat-format JSONL)
    run(fc + ["create", "dataset", dataset_id, dataset_path])
    # 2) launch the supervised LoRA fine-tuning job
    run(fc + ["create", "supervised-fine-tuning-job",
              "--base-model", base_model,
              "--dataset", dataset_id,
              "--output-model", out_model,
              "--epochs", str(epochs),
              "--lora-rank", str(lora_rank)])
    # 3) poll the output model until it is READY
    print("    [waiting for fine-tuning job — watch with "
          "`firectl list supervised-fine-tuning-job`]")
    while True:
        time.sleep(30)
        res = subprocess.run(fc + ["get", "model", out_model],
                             capture_output=True, text=True)
        if "READY" in res.stdout.upper():
            break
    model_id = f"accounts/{acct}/models/{out_model}" if acct else out_model
    return model_id


# ----------------------------- the RFT loop -----------------------------

def rft_loop(iterations, train_seeds, eval_seeds, base_model, n_rollouts,
             cfg, out_dir, selftest):
    verifier = Verifier()
    os.makedirs(out_dir, exist_ok=True)
    curve = []

    # --- references for context (oracle = ceiling, scripted = strong heuristic) ---
    oracle = evaluate(eval_seeds, lambda w, seed=0: OraclePolicy(w), cfg, verifier)
    scripted = evaluate(eval_seeds, lambda w, seed=0: ScriptedExperimenter(w, seed), cfg, verifier)
    print(f"\nReference  oracle={oracle['mean_reward']}  scripted={scripted['mean_reward']}\n")

    # --- iteration 0: the BASE model, untrained ---
    if selftest:
        skill = 0.15  # weak base
        eval_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.0)
        roll_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.7)
    else:
        current_model = base_model
        eval_factory = make_llm_agent(current_model, temperature=0.0)
        roll_factory = make_llm_agent(current_model, temperature=0.7)

    base_eval = evaluate(eval_seeds, eval_factory, cfg, verifier)
    curve.append({"iter": 0, "model": "base", "eval": base_eval})
    print(f"[iter 0] BASE model  eval mean_reward={base_eval['mean_reward']}  "
          f"flagged={base_eval['flagged']}/{base_eval['n']}")

    for it in range(1, iterations + 1):
        print(f"\n[iter {it}] rollout + reject (N={n_rollouts}) over "
              f"{len(train_seeds)} worlds...")
        dataset, stats = rollout_and_filter(train_seeds, roll_factory, cfg, verifier, n_rollouts)
        kept_worlds = sum(1 for s in stats if s["rounds_added"] > 0)
        ds_path = write_jsonl(dataset, os.path.join(out_dir, f"sft_iter{it}.jsonl"))
        print(f"          kept {kept_worlds}/{len(train_seeds)} worlds, "
              f"{len(dataset)} training turns -> {ds_path}")

        if selftest:
            # simulate fine-tuning: training on winners raises the model's skill floor.
            skill = min(0.95, skill + 0.30)
            eval_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.0)
            roll_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.7)
            model_name = f"mock-skill-{skill:.2f}"
        else:
            if not dataset:
                print("          no winning trajectories — skipping fine-tune this iter.")
                continue
            current_model = finetune_firectl(ds_path, base_model, tag=f"it{it}")
            eval_factory = make_llm_agent(current_model, temperature=0.0)
            roll_factory = make_llm_agent(current_model, temperature=0.7)
            model_name = current_model

        ev = evaluate(eval_seeds, eval_factory, cfg, verifier)
        curve.append({"iter": it, "model": model_name, "eval": ev,
                      "kept_worlds": kept_worlds, "train_turns": len(dataset)})
        print(f"[iter {it}] {model_name}  eval mean_reward={ev['mean_reward']}  "
              f"flagged={ev['flagged']}/{ev['n']}")

    # --- the money shot: does the curve bend? ---
    print("\n" + "=" * 60)
    print("RFT CURVE (mean held-out reward by iteration)")
    print("=" * 60)
    base_r = curve[0]["eval"]["mean_reward"]
    span = max(1.0, oracle["mean_reward"] - min(0, base_r))
    for c in curve:
        r = c["eval"]["mean_reward"]
        bar = "#" * int(40 * max(0, r) / span)
        print(f"  iter {c['iter']}  {r:>9.1f}  {bar}")
    print(f"  {'oracle':>6}  {oracle['mean_reward']:>9.1f}  (ceiling)")
    final_r = curve[-1]["eval"]["mean_reward"]
    delta = final_r - base_r
    print("-" * 60)
    print(f"BASE -> RFT: {base_r:.1f} -> {final_r:.1f}   "
          f"({'+' if delta >= 0 else ''}{delta:.1f}, "
          f"{(delta / abs(base_r) * 100) if base_r else float('inf'):+.0f}%)")
    print("=" * 60)

    with open(os.path.join(out_dir, "curve.json"), "w") as f:
        json.dump({"curve": curve, "oracle": oracle, "scripted": scripted}, f, indent=2)
    return curve


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="FirmBench rejection-sampling fine-tuning")
    ap.add_argument("--selftest", action="store_true",
                    help="offline dry run with a mock model (no network)")
    ap.add_argument("--run", action="store_true",
                    help="real run: Fireworks rollouts + firectl fine-tune")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=4, help="rollouts per world (best-of-N)")
    ap.add_argument("--train-seeds", type=int, default=16)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="rft_out")
    args = ap.parse_args()

    if not args.selftest and not args.run:
        print("Choose a mode: --selftest (offline) or --run (real Fireworks). "
              "See `python3 rft.py -h`.")
        return

    cfg = Config()
    train_seeds = list(range(1, 1 + args.train_seeds))
    eval_seeds = list(range(100, 100 + args.eval_seeds))  # disjoint held-out

    print("=" * 60)
    print("FirmBench — Real RFT (rejection-sampling fine-tuning)")
    print(f"mode={'selftest' if args.selftest else 'run'}  base_model={args.model}")
    print(f"iterations={args.iterations}  rollouts/world={args.rollouts}  "
          f"train_worlds={len(train_seeds)}  eval_worlds={len(eval_seeds)}")
    print("=" * 60)

    if args.run and not os.environ.get("FIREWORKS_API_KEY"):
        print("\nERROR: --run needs FIREWORKS_API_KEY in the environment.")
        print("  export FIREWORKS_API_KEY=...   and   firectl signin")
        return

    rft_loop(args.iterations, train_seeds, eval_seeds, args.model, args.rollouts,
             cfg, args.out, selftest=args.selftest)


if __name__ == "__main__":
    main()
