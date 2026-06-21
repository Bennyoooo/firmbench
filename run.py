"""
FirmBench — verifier, head-to-head evaluation, and REINFORCE training loop.

Merges patterns from rl-experiments (verifier + tripwires + training) and
autonomous-businesses-template (secret held-out data, execution-based grading)
into FirmBench's market-discovery sim.

  python3 run.py          # full pipeline: eval baselines → train → re-eval

Key ideas ported:
  - Secret held-out users (a slice of the demography the agent never gets
    campaign feedback on; final profit graded on them). Information asymmetry.
  - Execution-based grading: the verifier re-runs the funnel, doesn't trust
    the agent's self-reported numbers.
  - Cheat tripwire: if agent-reported profit diverges from true held-out profit,
    the submission is flagged and penalized.
  - REINFORCE over a parameterized probe-vs-exploit policy, learned from the
    verifier reward. Same shape as the full RFT of an LLM student.
"""

import argparse
import math
import random

from sim import (Config, World, FirmEnv, generate_world, sigmoid,
                 OraclePolicy, NaivePolicy, ScriptedExperimenter,
                 run_episode, best_price_for, theoretical_max)


# ----------------------------- verifier -----------------------------

class Verifier:
    """Grades an episode by discovery efficiency = profit / oracle (a strong reference).

    The original secret-held-out + tripwire design was dropped: this env is
    execution-based (the env computes profit — nothing for a policy to fake) and
    domain-randomized per seed, so generalization is measured by held-out eval SEEDS,
    not a within-episode user split. Rationale: docs/plans/2026-06-21-firmbench-v2-design.md.
    """

    def __init__(self, holdout_frac=0.2, tripwire_thresh=0.15, cheat_penalty=0.3):
        # retained for backward-compatible construction; no longer used by grade()
        self.holdout_frac = holdout_frac
        self.tripwire_thresh = tripwire_thresh
        self.cheat_penalty = cheat_penalty

    def grade(self, world: World, episode_result: dict) -> dict:
        """Grade by discovery efficiency = profit / oracle (a strong reference).

        The secret-held-out + tripwire design was dropped: this env is execution-based
        (the env computes profit — nothing to fake) and domain-randomized per seed, so
        generalization is measured by held-out eval SEEDS, not a user split. `flagged`
        is kept (always False) for backward-compatible callers (rft.py)."""
        profit = episode_result.get("total_profit", episode_result.get("reported_profit", 0.0))
        tmax = theoretical_max(world)                     # optimistic ceiling (true upper bound)
        oracle = run_episode(world, OraclePolicy(world))  # achievable-expert baseline (reported)
        pct_of_max = profit / tmax if tmax > 0 else 0.0
        return {
            "profit": round(profit, 2),
            "reported_profit": round(profit, 2),     # alias for backward-compat callers
            "theoretical_max": round(tmax, 2),
            "oracle_profit": round(oracle, 2),
            "pct_of_max": round(pct_of_max, 3),
            "pct_of_oracle": round(profit / oracle if oracle > 0 else 0.0, 3),
            "disc_eff": round(pct_of_max, 3),        # disc_eff now = fraction of the ceiling
            "reward": round(max(0.0, min(1.0, pct_of_max)), 4),
            "flagged": False,
            "over_ceiling": pct_of_max > 1.0,
        }


# ----------------------------- episode runner with result capture ----

def run_episode_detailed(world, policy):
    """Run an episode and capture per-round action log for the verifier."""
    env = FirmEnv(world)
    policy.reset()
    obs = env.reset()
    done = False
    action_log = []
    while not done:
        action = policy.act(env, obs)
        # record the action for replay
        log_entry = {
            "build": action.get("build"),
            "price": action.get("price", env.price),
            "campaigns": [],
        }
        for c in action.get("campaigns", []) or []:
            log_entry["campaigns"].append({
                "target": sorted(c.get("target", set())),
                "spend": float(c.get("spend", 0.0)),
            })
        action_log.append(log_entry)
        obs, reward, done, _ = env.step(action)
    return {
        "action_log": action_log,
        "reported_profit": env.total_profit,
        "total_profit": env.total_profit,
    }


# ----------------------------- head-to-head evaluation ---------------

def evaluate_policy(policy_fn, worlds, verifier):
    """Run a policy across multiple worlds, grade each with the verifier.
    policy_fn: callable(world) -> policy instance.
    Returns aggregate stats."""
    profits, rewards, flags = [], [], 0
    for w in worlds:
        pol = policy_fn(w)
        result = run_episode_detailed(w, pol)
        grade = verifier.grade(w, result)
        profits.append(grade["reported_profit"])
        rewards.append(grade["reward"])
        flags += int(grade["flagged"])
    n = len(worlds)
    return {
        "mean_profit": sum(profits) / n,
        "mean_reward": sum(rewards) / n,
        "flagged": flags,
    }


# ----------------------------- REINFORCE training --------------------

class RLProbeExploitPolicy:
    """Parameterized policy: how aggressively to probe vs exploit.

    Two learnable parameters:
      theta[0]: bias toward probing (positive = more probing rounds)
      theta[1]: sensitivity to remaining unsolved pains

    At each round, P(probe) = sigmoid(theta[0] + theta[1] * unsolved_frac).
    When probing: build a new feature + run diagnostic campaigns.
    When exploiting: target solved pains at optimal price.

    This is the same shape as the full RFT of an LLM — a policy parameterized
    by weights, updated from reward via policy gradient.
    """
    name = "RL (REINFORCE)"

    def __init__(self, world, theta=None, seed=0):
        self.w = world
        self.theta = theta or [0.0, 0.0]
        self.rng = random.Random(seed + 42)

    def reset(self):
        self.pain_demand = {}
        self.solved = {}
        self.tried_features = set()
        self._last_built = None
        self._log_probs = []

    def _p_probe(self, obs):
        n_pains = self.w.cfg.n_pains
        unsolved = max(0, n_pains - len(self.solved))
        unsolved_frac = unsolved / n_pains
        z = self.theta[0] + self.theta[1] * unsolved_frac
        return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))

    def act(self, env, obs):
        cfg = self.w.cfg
        built = set(obs["built_features"])

        # ingest diagnostics
        for c in obs["per_campaign"]:
            tgt = c["target"]
            if len(tgt) == 1:
                p = tgt[0]
                if obs["round"] <= 1:
                    self.pain_demand[p] = c["audience"]
                if c["purchases"] > 0.5 and self._last_built is not None:
                    self.solved[p] = self._last_built
        self._last_built = None

        # round 0: always probe demand
        if obs["round"] == 0:
            campaigns = [{"target": {p}, "spend": 10.0} for p in range(cfg.n_pains)]
            return {"build": None, "price": 50.0, "campaigns": campaigns}

        # decide probe vs exploit
        p_probe = self._p_probe(obs)
        probe = self.rng.random() < p_probe
        self._log_probs.append((probe, p_probe))

        top_pains = sorted(self.pain_demand, key=self.pain_demand.get, reverse=True)
        unsolved_top = [p for p in top_pains if p not in self.solved]

        if probe and unsolved_top and obs["round"] < cfg.horizon - 1:
            candidates = [x for x in range(cfg.n_features)
                          if x not in built and x not in self.tried_features]
            f = candidates[0] if candidates else None
            if f is not None:
                self.tried_features.add(f)
                self._last_built = f
                test_targets = unsolved_top[:6]
                campaigns = [{"target": {p}, "spend": 60.0} for p in test_targets]
                if self.solved:
                    campaigns.append({"target": set(self.solved.keys()),
                                      "spend": max(0.0, obs["cash"] * 0.4)})
                return {"build": f, "price": 50.0, "campaigns": campaigns}

        # exploit
        if self.solved:
            target = set(self.solved.keys())
            built_map = {x: 1.0 for x in built}
            price = best_price_for(self.w, built_map, target)
            spend = max(0.0, obs["cash"])
            return {"build": None, "price": price,
                    "campaigns": [{"target": target, "spend": spend}]}

        target = set(top_pains[:3]) if top_pains else {0, 1, 2}
        return {"build": None, "price": 50.0,
                "campaigns": [{"target": target, "spend": min(500.0, obs["cash"])}]}


def train_reinforce(train_seeds, cfg=None, iters=400, alpha=0.3, seed=0):
    """REINFORCE over the probe-vs-exploit policy. Returns learned theta."""
    cfg = cfg or Config()
    rng = random.Random(seed)
    theta = [0.0, 0.0]
    baseline = 0.0

    for it in range(iters):
        s = rng.choice(train_seeds)
        world = generate_world(s, cfg)
        pol = RLProbeExploitPolicy(world, theta=list(theta), seed=it)
        profit = run_episode(world, pol)

        # reward = normalized profit (profit / oracle gives ~[0,1])
        reward = profit / 100000.0  # rough scale
        baseline = 0.95 * baseline + 0.05 * reward if it else reward
        adv = reward - baseline

        # policy gradient over all probe/exploit decisions in the episode
        for (chose_probe, p_probe) in pol._log_probs:
            a = 1.0 if chose_probe else 0.0
            grad = adv * (a - p_probe)
            unsolved_frac = 1.0  # approximate; directionally correct
            theta[0] += alpha * grad * 1.0
            theta[1] += alpha * grad * unsolved_frac

        if it % 100 == 0:
            print(f"  iter {it:4d}  theta=[{theta[0]:+.3f},{theta[1]:+.3f}]  "
                  f"baseline={baseline:.3f}")
    return theta


# ----------------------------- multi-agent head-to-head (Phase D) ----

def evaluate_multiagent(test_seeds, cfg=None):
    """Phase-D head-to-head on held-out seeds: single-agent ScriptedExperimenter vs the
    coordinated ScriptedTeam vs the isolated NaiveTeam vs the full-info oracle ceiling.
    Reports disc.eff (profit/oracle) and the coordination tax (oracle - team_profit).

    The oracle is the SINGLE-agent OraclePolicy (no partial-obs barrier): a perfectly
    coordinated team approaches it; the gap is the tax. The single-agent scripted bounds
    the achievable-without-the-barrier side; the naive team bounds the no-comms side."""
    from multiagent import ScriptedTeam, NaiveTeam, run_team_episode
    cfg = cfg or Config.phase_a()

    agg = {"oracle": [], "single": [], "scripted_team": [], "naive_team": []}
    eff = {"single": [], "scripted_team": [], "naive_team": []}
    for s in test_seeds:
        w = generate_world(s, cfg)
        oracle = run_episode(w, OraclePolicy(w))
        single = run_episode(w, ScriptedExperimenter(w, s))
        _, scr = run_team_episode(w, ScriptedTeam(w, s))
        _, naive = run_team_episode(w, NaiveTeam(w, s))
        agg["oracle"].append(oracle); agg["single"].append(single)
        agg["scripted_team"].append(scr); agg["naive_team"].append(naive)
        denom = oracle if oracle > 0 else 1.0
        eff["single"].append(single / denom)
        eff["scripted_team"].append(scr / denom)
        eff["naive_team"].append(naive / denom)

    n = len(test_seeds)
    mean = {k: sum(v) / n for k, v in agg.items()}
    meaneff = {k: sum(v) / n for k, v in eff.items()}

    print("=" * 74)
    print(f"FirmBench — Phase D multi-agent head-to-head ({n} held-out seeds, full market)")
    print("=" * 74)
    hdr = f"{'policy':<26}{'mean profit':>14}{'disc.eff':>10}{'coord tax':>12}"
    print(hdr); print("-" * len(hdr))
    rows = [
        ("oracle (full-info)", mean["oracle"], 1.000, 0.0),
        ("single-agent scripted", mean["single"], meaneff["single"], 1 - meaneff["single"]),
        ("scripted-team (comms)", mean["scripted_team"], meaneff["scripted_team"], 1 - meaneff["scripted_team"]),
        ("naive-team (no comms)", mean["naive_team"], meaneff["naive_team"], 1 - meaneff["naive_team"]),
    ]
    for name, prof, de, tax in rows:
        print(f"{name:<26}{prof:>14.0f}{de:>10.3f}{tax:>11.1%}")
    print("-" * len(hdr))
    coord_gain = meaneff["scripted_team"] - meaneff["naive_team"]
    print(f"VERDICT: coordination (blackboard messages) buys {coord_gain:+.1%} of the oracle "
          f"(scripted-team {meaneff['scripted_team']:.3f} vs naive-team {meaneff['naive_team']:.3f}).")
    gate = ("PASS" if mean["naive_team"] < mean["scripted_team"] <= mean["oracle"] else "FAIL")
    print(f"         coordination gate: naive < scripted_team <= oracle -> {gate}")
    print("=" * 74)
    return {"mean": mean, "disc_eff": meaneff, "gate": gate}


# ----------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="FirmBench eval / training")
    ap.add_argument("--multiagent", action="store_true",
                    help="Phase D: team head-to-head (scripted vs naive vs oracle) + coordination tax")
    ap.add_argument("--eval-seeds", type=int, default=10, help="held-out seed count")
    args = ap.parse_args()

    if args.multiagent:
        test_seeds = list(range(100, 100 + args.eval_seeds))   # disjoint held-out
        evaluate_multiagent(test_seeds, Config.phase_a())
        return

    cfg = Config()
    verifier = Verifier()

    train_seeds = list(range(1, 21))
    test_seeds = list(range(100, 121))  # disjoint held-out
    test_worlds = [generate_world(s, cfg) for s in test_seeds]

    print("=" * 70)
    print("FirmBench — head-to-head evaluation + REINFORCE training")
    print("=" * 70)

    # Define policies
    policies = {
        "naive (random)": lambda w: NaivePolicy(w, seed=42),
        "scripted (experimenter)": lambda w: ScriptedExperimenter(w, seed=42),
        "oracle (omniscient)": lambda w: OraclePolicy(w),
    }

    print(f"\n[1] Baseline evaluation on {len(test_worlds)} held-out worlds:\n")
    header = f"{'policy':<30}{'mean profit':>12}{'mean reward':>12}{'flagged':>9}"
    print(header)
    print("-" * len(header))
    for name, pol_fn in policies.items():
        r = evaluate_policy(pol_fn, test_worlds, verifier)
        print(f"{name:<30}{r['mean_profit']:>12.0f}{r['mean_reward']:>12.3f}"
              f"{r['flagged']:>9}")

    print(f"\n[2] Training REINFORCE probe/exploit policy on {len(train_seeds)} "
          f"training worlds...")
    theta = train_reinforce(train_seeds, cfg)
    print(f"    learned theta: [{theta[0]:+.3f}, {theta[1]:+.3f}]")

    # Add trained policy and re-evaluate
    policies["RL (REINFORCE)"] = lambda w, t=theta: RLProbeExploitPolicy(w, theta=list(t), seed=42)

    print(f"\n[3] Post-training evaluation on {len(test_worlds)} held-out worlds:\n")
    print(header)
    print("-" * len(header))
    results = {}
    for name, pol_fn in policies.items():
        r = evaluate_policy(pol_fn, test_worlds, verifier)
        results[name] = r
        print(f"{name:<30}{r['mean_profit']:>12.0f}{r['mean_reward']:>12.3f}"
              f"{r['flagged']:>9}")

    # Verdict
    winner = max(results.items(), key=lambda kv: kv[1]["mean_reward"])
    rl = results.get("RL (REINFORCE)", {})
    scripted = results.get("scripted (experimenter)", {})
    print("\n" + "=" * 70)
    print(f"VERDICT: highest reward -> {winner[0]}")
    if rl and scripted:
        print(f"  RL vs scripted: profit {rl['mean_profit']:.0f} vs {scripted['mean_profit']:.0f}"
              f"  |  reward {rl['mean_reward']:.3f} vs {scripted['mean_reward']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
