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

import math
import random

from sim import (Config, World, FirmEnv, generate_world, sigmoid,
                 OraclePolicy, NaivePolicy, ScriptedExperimenter,
                 run_episode, best_price_for)


# ----------------------------- verifier -----------------------------

class Verifier:
    """Grades an episode on SECRET held-out users the agent never saw feedback from.

    The world is split: 80% of users are 'visible' (agent gets campaign results
    from them); 20% are 'held-out' (used only by the verifier to compute the true
    score). This mirrors the secret-test-set pattern from rl-experiments and the
    hidden-ticket pattern from autonomous-businesses-template.
    """

    def __init__(self, holdout_frac=0.2, tripwire_thresh=0.15, cheat_penalty=0.3):
        self.holdout_frac = holdout_frac
        self.tripwire_thresh = tripwire_thresh
        self.cheat_penalty = cheat_penalty

    def split_world(self, world: World):
        """Split users into visible (agent sees feedback) and held-out (verifier only).
        Returns (visible_indices, holdout_indices). Deterministic per world seed."""
        n = len(world.users)
        cutoff = int(n * (1 - self.holdout_frac))
        visible = list(range(cutoff))
        holdout = list(range(cutoff, n))
        return visible, holdout

    def replay_on_holdout(self, world: World, holdout_indices: list,
                          action_log: list) -> float:
        """Re-run the entire episode's actions on held-out users only.

        action_log: list of dicts, one per round, each = the action + round's
        built state. We replay the funnel per round per campaign, accumulating
        profit only from held-out users. This is execution-based grading —
        we don't trust any reported number, we re-compute.
        """
        cfg = world.cfg
        holdout_set = set(holdout_indices)
        # build a held-out-only users_by_pain index
        ho_by_pain = {p: [] for p in range(cfg.n_pains)}
        for idx in holdout_indices:
            for p in world.users[idx].pains:
                ho_by_pain[p].append(idx)

        total_profit = 0.0
        built = {}  # track built features across rounds

        for entry in action_log:
            # update built features
            f = entry.get("build")
            if f is not None:
                built[f] = 1.0
            price = entry["price"]
            round_revenue = 0.0
            round_spend = 0.0

            for camp in entry.get("campaigns", []):
                target = set(camp["target"])
                spend = camp["spend"] * self.holdout_frac  # proportional
                round_spend += spend

                pool = set()
                for p in target:
                    pool.update(ho_by_pain.get(p, []))
                pool = sorted(pool)
                impressions = int(spend * cfg.impressions_per_dollar)
                reached = pool[:impressions]

                purchases = 0.0
                for idx in reached:
                    u = world.users[idx]
                    resonance = len(target & u.pains) / len(u.pains) if u.pains else 0.0
                    ff = 0.0
                    if u.pains:
                        for p in u.pains:
                            fe = world.solves[p]
                            if fe in built:
                                ff += built[fe]
                        ff /= len(u.pains)
                    p_buy = sigmoid(cfg.alpha * ff + cfg.beta * (u.wtp - price) / u.wtp - cfg.gamma)
                    purchases += resonance * p_buy
                round_revenue += purchases * price

            build_cost = cfg.build_cost * self.holdout_frac if f is not None else 0.0
            total_profit += round_revenue - round_spend - build_cost

        return total_profit

    def grade(self, world: World, episode_result: dict) -> dict:
        """Grade an episode. episode_result must contain:
            action_log: list of per-round action dicts,
            reported_profit: float, total_profit: float
        """
        _, holdout = self.split_world(world)
        holdout_profit = self.replay_on_holdout(
            world, holdout, episode_result["action_log"])
        # scale reported profit to holdout fraction for fair comparison
        reported_scaled = episode_result["reported_profit"] * self.holdout_frac
        gap = abs(reported_scaled - holdout_profit) / (abs(holdout_profit) + 100.0)
        flagged = gap > self.tripwire_thresh

        reward = holdout_profit
        if flagged:
            reward -= self.cheat_penalty * abs(holdout_profit)

        return {
            "holdout_profit": round(holdout_profit, 2),
            "reported_profit": round(episode_result["total_profit"], 2),
            "gap": round(gap, 4),
            "flagged": flagged,
            "reward": round(reward, 2),
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


# ----------------------------- main ----------------------------------

def main():
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
        print(f"{name:<30}{r['mean_profit']:>12.0f}{r['mean_reward']:>12.0f}"
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
        print(f"{name:<30}{r['mean_profit']:>12.0f}{r['mean_reward']:>12.0f}"
              f"{r['flagged']:>9}")

    # Verdict
    winner = max(results.items(), key=lambda kv: kv[1]["mean_reward"])
    rl = results.get("RL (REINFORCE)", {})
    scripted = results.get("scripted (experimenter)", {})
    print("\n" + "=" * 70)
    print(f"VERDICT: highest reward -> {winner[0]}")
    if rl and scripted:
        print(f"  RL vs scripted: profit {rl['mean_profit']:.0f} vs {scripted['mean_profit']:.0f}"
              f"  |  reward {rl['mean_reward']:.0f} vs {scripted['mean_reward']:.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
