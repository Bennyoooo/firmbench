"""Render multi-agent (team) runs into the replay UI.

Writes artifacts_multiagent/<seed>_<key>/manifest.json in the format replay_multiagent.html
expects: per-round {build, price, budget, messages (blackboard), campaigns (results),
round_profit} + top-level {seed, disc_eff=profit/theoretical_max, theoretical_max,
oracle_profit, team_profit, coordination_tax, pain_names, feature_names, rounds}.

Runs the team POLICIES (oracle-team / scripted-team / naive-team) through MultiAgentFirmEnv
directly (deterministic, offline, free) — the coordination story (blackboard + role actions)
the frontier-model leaderboard is measured against.

    python3 export_replay_multiagent.py            # seed 42, all three team policies
    python3 export_replay_multiagent.py 123        # another seed
"""
import json
import os
import sys

from sim import Config, generate_world, OraclePolicy, run_episode, theoretical_max
from multiagent import MultiAgentFirmEnv, ROLES, OracleTeam, ScriptedTeam, NaiveTeam

CFG = Config.phase_a()
ARTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_multiagent")

TEAMS = {
    "oracle-team": lambda w, s: OracleTeam(w, s),
    "scripted-team": lambda w, s: ScriptedTeam(w, s),
    "naive-team": lambda w, s: NaiveTeam(w, s),
}


def run_team_capturing(world, team):
    """Drive a team episode, capturing the per-round replay log (incl. the blackboard)."""
    menv = MultiAgentFirmEnv(world)
    team.reset()
    menv.reset()
    rounds = []
    while not menv.done:
        for role in ROLES:
            action, message = team.act(role, menv.role_obs(role), menv)
            menv.submit(role, action, message=message)
        messages = [dict(m) for m in menv.bb.msgs]      # blackboard before commit clears it
        budget = menv._budget
        _obs, profit, _done, committed = menv.commit()
        rounds.append({
            "build": committed["build"],
            "price": round(float(committed["price"]), 1),
            "budget": (round(budget, 2) if budget is not None else None),
            "messages": messages,
            "campaigns": [{"target": list(c.get("target", [])), "channel": int(c.get("channel", 0)),
                           "spend": round(float(c.get("spend", 0.0)), 1),
                           "audience": int(c.get("audience", 0)),
                           "tries": round(float(c.get("tries", 0.0)), 2),
                           "purchases": round(float(c.get("purchases", 0.0)), 2),
                           "revenue": round(float(c.get("revenue", 0.0)), 1)}
                          for c in menv._last_per_campaign],
            "round_profit": round(profit, 2),
        })
    return rounds, menv.total_profit


def write_manifest(seed, key, rounds, profit, world):
    tmax = theoretical_max(world) or 1.0
    oracle = run_episode(world, OraclePolicy(world))
    disc = profit / tmax
    manifest = {
        "seed": seed, "rounds": len(rounds),
        "final_reward": round(max(0.0, min(1.0, disc)), 4),
        "team_profit": round(profit, 2), "profit": round(profit, 2),
        "theoretical_max": round(tmax, 2), "oracle_profit": round(oracle, 2),
        "disc_eff": round(disc, 3),
        "pct_of_oracle": round(profit / oracle if oracle > 0 else 0.0, 3),
        "coordination_tax": round(oracle - profit, 2),
        "pain_names": world.pain_names, "feature_names": world.feature_names,
        "actions": rounds,
    }
    d = os.path.join(ARTROOT, f"{seed}_{key}")
    os.makedirs(d, exist_ok=True)
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=2, default=list)
    print(f"  wrote {os.path.relpath(d)}/manifest.json  rounds={len(rounds)} "
          f"team_profit={profit:.0f} disc_eff={disc:.3f}")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    world = generate_world(seed, CFG)
    print(f"Exporting team-policy replays for seed {seed} (theoretical_max grade):")
    for key, factory in TEAMS.items():
        rounds, profit = run_team_capturing(world, factory(world, seed))
        write_manifest(seed, key, rounds, profit, world)


if __name__ == "__main__":
    main()
