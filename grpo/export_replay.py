"""Render GRPO eval runs into the replay UI.

Writes artifacts/<seed>_<key>/manifest.json in the format replay.html expects (per-round
build/price/campaigns+results, plus final_reward/disc_eff/profit/pain_names). Two sources:
  - a deployed MODEL (e.g. the GRPO checkpoint): get its strategy plan, expand into a
    per-round episode via FirmEnv (build features, then exploit-campaign the targets).
  - a local POLICY (oracle/scripted/naive): run multi-turn through FirmEnv directly.

Usage:
  python3 export_replay.py 42 model  qwen3-8b-grpo  "<model_id#deployment>"
  python3 export_replay.py 42 policy oracle
  python3 export_replay.py 42 policy scripted
"""
import os, sys, json
from sim import generate_world, FirmEnv, OraclePolicy, ScriptedExperimenter, NaivePolicy, run_episode
import test_firmbench_grpo as T

cfg = T.cfg  # same tight economy the model was trained/eval'd on
ARTROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def round_from_obs(action, per_campaign):
    """Build a replay 'action' (round) dict from the env action + campaign results."""
    camps = []
    for c in per_campaign:
        camps.append({
            "target": list(c.get("target", [])),
            "spend": round(float(c.get("spend", 0.0)), 1),
            "revenue": round(float(c.get("revenue", 0.0)), 1),
            "purchases": round(float(c.get("purchases", 0.0)), 2),
            "audience": int(c.get("audience", 0)),
            "channel": int(c.get("channel", 0)),
        })
    return {"build": action.get("build"), "price": round(float(action.get("price", 50.0)), 1),
            "campaigns": camps}


def episode_from_actions(world, action_iter):
    """Run an episode given a callable producing each round's action; capture rounds."""
    env = FirmEnv(world); env.reset(); actions = []; done = False
    while not done:
        act = action_iter(env)
        obs, _profit, done, _ = env.step(act)
        actions.append(round_from_obs(act, obs.get("per_campaign", [])))
    return actions, env.total_profit


def plan_episode(world, build_ids, price, target):
    """Expand a strategy plan into a per-round episode (build, then exploit)."""
    to_build = list(dict.fromkeys(int(b) for b in build_ids))
    def step(env):
        f = to_build.pop(0) if to_build else None
        reserve = cfg.build_cost * len(to_build)
        bn = cfg.build_cost if f is not None else 0.0
        spend = max(0.0, env.cash - bn - reserve)
        camps = [{"target": set(target), "spend": spend}] if (target and spend > 0) else []
        return {"build": f, "price": price, "campaigns": camps}
    return episode_from_actions(world, step)


def policy_episode(world, policy):
    """Run a local multi-turn policy, capturing per-round actions+results."""
    policy.reset()
    def step(env):
        obs = env._state_obs(per_campaign=getattr(env, "_last_pc", []))
        act = policy.act(env, obs)
        return act
    # simpler: replicate run loop capturing
    env = FirmEnv(world); env.reset(); policy.reset()
    obs = env._state_obs(per_campaign=[]); actions = []; done = False
    while not done:
        act = policy.act(env, obs)
        obs, _p, done, _ = env.step(act)
        actions.append(round_from_obs(act, obs.get("per_campaign", [])))
    return actions, env.total_profit


def write_manifest(seed, key, actions, profit):
    world = generate_world(seed, cfg)
    oracle = run_episode(world, OraclePolicy(world))
    disc = profit / oracle if oracle > 0 else 0.0
    manifest = {
        "seed": seed, "rounds": len(actions),
        "final_reward": round(max(0.0, min(1.0, disc)), 4),
        "profit": round(profit, 2), "oracle_profit": round(oracle, 2),
        "disc_eff": round(disc, 3), "flagged": False,
        "pain_names": getattr(world, "pain_names", None) or [f"Pain {i}" for i in range(cfg.n_pains)],
        "feature_names": getattr(world, "feature_names", None) or [f"Feature {i}" for i in range(cfg.n_features)],
        "actions": actions,
    }
    d = os.path.join(ARTROOT, f"{seed}_{key}")
    os.makedirs(d, exist_ok=True)
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=2)
    print(f"wrote {d}/manifest.json  rounds={len(actions)} profit={profit:.0f} disc_eff={disc:.2f}")


def main():
    seed = int(sys.argv[1]); mode = sys.argv[2]; key = sys.argv[3]
    world = generate_world(seed, cfg)
    if mode == "policy":
        pol = {"oracle": OraclePolicy(world), "scripted": ScriptedExperimenter(world, seed),
               "naive": NaivePolicy(world, seed)}[key]
        actions, profit = policy_episode(world, pol)
    elif mode == "model":
        from openai import OpenAI
        model_id = sys.argv[4]
        client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1")
        msgs = [{"role": "system", "content": "You are a sharp startup operator. Reason briefly, then output the JSON plan."},
                {"role": "user", "content": T.build_prompt(world)}]
        r = client.chat.completions.create(model=model_id, messages=msgs, temperature=0.0, max_tokens=3072)
        parsed = T.parse_plan(r.choices[0].message.content or "")
        if not parsed:
            print("model produced no valid plan; writing empty episode"); parsed = ([], 50.0, set())
        b, p, t = parsed
        actions, profit = plan_episode(world, b, p, t)
    else:
        raise SystemExit("mode must be 'policy' or 'model'")
    write_manifest(seed, key, actions, profit)


if __name__ == "__main__":
    main()
