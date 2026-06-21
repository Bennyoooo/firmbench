"""
FirmBench — Vertex AI RFT: rejection-sampling fine-tuning on Google Cloud.

The GCP counterpart to `rft.py` (Fireworks/firectl). Same recipe — expert iteration /
STaR — but the fine-tune step runs **Vertex AI supervised tuning** of a Gemini model
(dataset staged on GCS) instead of `firectl`. Everything else (rollout, held-out
verifier grading, rejection sampling, eval curve) is identical in shape.

The loop (one iteration):
  1. ROLLOUT   — for each training world, sample N rollouts with the current model
                 (temperature > 0). Record (prompt, completion) per round.
  2. GRADE     — score each rollout with the held-out Verifier (run.py). Curates the set:
                 only non-cheating, profitable episodes survive.
  3. REJECT    — keep the best non-flagged, positive-reward rollout per world.
  4. DATASET   — flatten winners into chat JSONL, convert to Vertex tuning format,
                 upload to GCS.
  5. TUNE      — launch a Vertex AI supervised tuning job -> tuned model endpoint.
  6. EVALUATE  — score the tuned model on disjoint held-out worlds. Append to the curve.

Two ways to run:
  python3 rft_gcp.py --selftest    # offline, no GCP: a MockModel whose skill rises each
                                    # iteration drives the WHOLE machinery and prints a
                                    # bending curve. Validates rollout->filter->dataset->eval.

  python3 rft_gcp.py --run         # the real thing. Needs:
                                    #   gcloud auth application-default login
                                    #   export GOOGLE_CLOUD_PROJECT=...
                                    #   export GCS_BUCKET=your-bucket   (for dataset staging)
                                    # Generates real rollouts, tunes on Vertex, re-evaluates.

This file is deliberately self-contained (imports only sim/run/agent_vertex) so it can
evolve independently of the Fireworks pipeline.

Flags: --iterations, --train-seeds, --eval-seeds, --rollouts, --model, --out.
"""

import os
import json
import time
import random
import argparse
from statistics import mean
from concurrent.futures import ThreadPoolExecutor

from sim import (Config, generate_world, FirmEnv, OraclePolicy, NaivePolicy,
                 ScriptedExperimenter)
from run import Verifier
from agent_vertex import (system_prompt, format_obs, make_vertex_agent, DEFAULT_MODEL,
                          DEFAULT_LOCATION)

MAX_PARALLEL = int(os.environ.get("RFT_PARALLEL", "8"))


# ----------------------------- episode recorder -----------------------------

def action_to_completion(action: dict) -> str:
    """Serialize an action dict into the same ```json block the model is asked to emit,
    so MockModel trajectories are byte-compatible with real Vertex trajectories."""
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
    """Run one full episode. Returns (records, action_log, total_profit)."""
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

    Plays the disciplined ScriptedExperimenter action with prob `skill`, else a random
    NaivePolicy action. Raising `skill` simulates fine-tuning on winning trajectories.
    Records (prompt, completion) exactly like VertexAgent, exercising the whole
    dataset/filter/eval path offline.
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
        action = (self.good if use_good else self.bad).act(env, obs)
        self.last_record = {"messages": [
            {"role": "system", "content": system_prompt(self.cfg)},
            {"role": "user", "content": format_obs(obs, self.cfg, self.history)},
            {"role": "assistant", "content": action_to_completion(action)},
        ]}
        return action


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
    """For each world: sample n_rollouts (in parallel), grade, keep the best non-flagged
    trajectory. Returns (dataset_records, stats)."""
    tasks = [(s, k, make_agent, cfg, verifier)
             for s in train_seeds for k in range(n_rollouts)]
    by_seed = {s: [] for s in train_seeds}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        for s, res in ex.map(_one_rollout, tasks):
            by_seed[s].append(res)

    dataset, stats = [], []
    for s in train_seeds:
        graded = by_seed[s]
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
    """Greedy eval (factory should produce temperature~0 agents). Mean held-out reward."""
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


def to_vertex_tuning_format(records):
    """Convert chat records ({"messages":[system,user,assistant]}) into the JSONL schema
    Vertex AI supervised tuning expects:

        {"systemInstruction": {"role":"system","parts":[{"text": ...}]},
         "contents": [{"role":"user","parts":[{"text": ...}]},
                      {"role":"model","parts":[{"text": ...}]}]}

    Returns a list of dicts (one per training turn).
    """
    out = []
    role_map = {"user": "user", "assistant": "model"}
    for rec in records:
        msgs = rec.get("messages", [])
        sys_text = next((m["content"] for m in msgs if m["role"] == "system"), None)
        contents = [
            {"role": role_map[m["role"]], "parts": [{"text": m["content"]}]}
            for m in msgs if m["role"] in role_map
        ]
        ex = {"contents": contents}
        if sys_text is not None:
            ex["systemInstruction"] = {"role": "system", "parts": [{"text": sys_text}]}
        out.append(ex)
    return out


def write_vertex_jsonl(records, path):
    return write_jsonl(to_vertex_tuning_format(records), path)


# ----------------------------- Vertex AI fine-tuning -----------------------------

def _upload_to_gcs(local_path, bucket, blob_name, project=None):
    """Upload a file to GCS and return its gs:// URI. Requires google-cloud-storage."""
    try:
        from google.cloud import storage
    except Exception as e:
        raise RuntimeError(
            "google-cloud-storage not installed. `pip install google-cloud-storage` "
            "(see requirements-gcp.txt)."
        ) from e
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    client = storage.Client(project=project)
    client.bucket(bucket).blob(blob_name).upload_from_filename(local_path)
    return f"gs://{bucket}/{blob_name}"


def finetune_vertex(dataset_records, base_model, tag, epochs=3):
    """Supervised-tune `base_model` on Vertex AI; return the tuned model resource name
    (usable directly as the `model` arg for inference via google-genai).

    Steps: convert -> stage JSONL on GCS -> launch tuning job -> poll until done.
    Requires ADC (`gcloud auth application-default login`), GOOGLE_CLOUD_PROJECT, and
    GCS_BUCKET for dataset staging.
    """
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError("google-genai not installed. `pip install google-genai`.") from e

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
    bucket = os.environ.get("GCS_BUCKET")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set (see docs/gcp-pipeline.md).")
    if not bucket:
        raise RuntimeError("GCS_BUCKET not set — Vertex tuning needs a GCS dataset URI.")

    # 1) convert + stage dataset on GCS
    local = f"vertex_sft_{tag}.jsonl"
    write_vertex_jsonl(dataset_records, local)
    gcs_uri = _upload_to_gcs(local, bucket, f"firmbench/{tag}/train.jsonl", project)
    print(f"    staged dataset -> {gcs_uri}")

    client = genai.Client(vertexai=True, project=project, location=location)

    # 2) launch the supervised tuning job
    job = client.tunings.tune(
        base_model=base_model,
        training_dataset=types.TuningDataset(gcs_uri=gcs_uri),
        config=types.CreateTuningJobConfig(
            epoch_count=epochs,
            tuned_model_display_name=f"firmbench-rft-{tag}",
        ),
    )
    print(f"    launched Vertex tuning job: {getattr(job, 'name', job)}")

    # 3) poll until the job finishes
    terminal_ok = {"JOB_STATE_SUCCEEDED"}
    terminal_bad = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        time.sleep(30)
        job = client.tunings.get(name=job.name)
        state = str(getattr(job, "state", ""))
        print(f"    [tuning] state={state}")
        if any(s in state for s in terminal_ok):
            break
        if any(s in state for s in terminal_bad):
            raise RuntimeError(f"Vertex tuning job ended in {state}")

    tuned = job.tuned_model
    # The endpoint is what you pass back as `model` for generate_content.
    model_id = getattr(tuned, "endpoint", None) or getattr(tuned, "model", None) or str(tuned)
    print(f"    tuned model: {model_id}")
    return model_id


# ----------------------------- the RFT loop -----------------------------

def rft_loop(iterations, train_seeds, eval_seeds, base_model, n_rollouts,
             cfg, out_dir, selftest):
    verifier = Verifier()
    os.makedirs(out_dir, exist_ok=True)
    curve = []

    # references: oracle = ceiling, scripted = strong heuristic
    oracle = evaluate(eval_seeds, lambda w, seed=0: OraclePolicy(w), cfg, verifier)
    scripted = evaluate(eval_seeds, lambda w, seed=0: ScriptedExperimenter(w, seed), cfg, verifier)
    print(f"\nReference  oracle={oracle['mean_reward']}  scripted={scripted['mean_reward']}\n")

    # iteration 0: the BASE model, untrained
    if selftest:
        skill = 0.15  # weak base
        eval_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.0)
        roll_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.7)
    else:
        current_model = base_model
        eval_factory = make_vertex_agent(model=current_model, temperature=0.0)
        roll_factory = make_vertex_agent(model=current_model, temperature=0.7)

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
            # simulate tuning: training on winners raises the model's skill floor.
            skill = min(0.95, skill + 0.30)
            eval_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.0)
            roll_factory = lambda w, seed=0, sk=skill: MockModel(w, skill=sk, seed=seed, temperature=0.7)
            model_name = f"mock-skill-{skill:.2f}"
        else:
            if not dataset:
                print("          no winning trajectories — skipping tune this iter.")
                continue
            current_model = finetune_vertex(dataset, base_model, tag=f"it{it}")
            eval_factory = make_vertex_agent(model=current_model, temperature=0.0)
            roll_factory = make_vertex_agent(model=current_model, temperature=0.7)
            model_name = current_model

        ev = evaluate(eval_seeds, eval_factory, cfg, verifier)
        curve.append({"iter": it, "model": model_name, "eval": ev,
                      "kept_worlds": kept_worlds, "train_turns": len(dataset)})
        print(f"[iter {it}] {model_name}  eval mean_reward={ev['mean_reward']}  "
              f"flagged={ev['flagged']}/{ev['n']}")

    # the money shot: does the curve bend?
    print("\n" + "=" * 60)
    print("RFT CURVE (mean held-out reward by iteration) — Vertex AI")
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
    ap = argparse.ArgumentParser(description="FirmBench Vertex AI rejection-sampling fine-tuning")
    ap.add_argument("--selftest", action="store_true",
                    help="offline dry run with a mock model (no GCP)")
    ap.add_argument("--run", action="store_true",
                    help="real run: Vertex rollouts + Vertex AI supervised tuning")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=4, help="rollouts per world (best-of-N)")
    ap.add_argument("--train-seeds", type=int, default=16)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="rft_gcp_out")
    args = ap.parse_args()

    if not args.selftest and not args.run:
        print("Choose a mode: --selftest (offline) or --run (real Vertex). "
              "See `python3 rft_gcp.py -h`.")
        return

    cfg = Config()
    train_seeds = list(range(1, 1 + args.train_seeds))
    eval_seeds = list(range(100, 100 + args.eval_seeds))  # disjoint held-out

    print("=" * 60)
    print("FirmBench — Vertex AI RFT (rejection-sampling fine-tuning)")
    print(f"mode={'selftest' if args.selftest else 'run'}  base_model={args.model}")
    print(f"iterations={args.iterations}  rollouts/world={args.rollouts}  "
          f"train_worlds={len(train_seeds)}  eval_worlds={len(eval_seeds)}")
    print("=" * 60)

    if args.run and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("\nERROR: --run needs GCP configured.")
        print("  gcloud auth application-default login")
        print("  export GOOGLE_CLOUD_PROJECT=...   export GCS_BUCKET=...")
        return

    rft_loop(args.iterations, train_seeds, eval_seeds, args.model, args.rollouts,
             cfg, args.out, selftest=args.selftest)


if __name__ == "__main__":
    main()
