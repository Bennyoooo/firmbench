"""
FirmBench — market-discovery simulation core (Phase 1, single-agent, deterministic).

Pure standard library (no numpy) so it runs anywhere. This module provides:
  - generate_world(seed): a randomized hidden market (pains, features, solves, demography)
  - FirmEnv: a gym-like reset()/step() environment; reward = round profit
  - OraclePolicy: strong informed reference (knows the world) -> regret baseline
  - NaivePolicy: no experimentation (random builds/targeting) -> floor
  - ScriptedExperimenter: probes demand, discovers solves mapping, then exploits

Run `python3 sim.py` to print a learnability check across seeds:
expect  naive  <<  scripted  <=  oracle.
"""

import math
import random
from dataclasses import dataclass, field


# ----------------------------- config -----------------------------

@dataclass
class Config:
    n_pains: int = 8
    n_features: int = 8
    n_users: int = 5000
    horizon: int = 10
    starting_cash: float = 6000.0
    build_cost: float = 300.0
    impressions_per_dollar: float = 0.1   # cost per reached user = 1 / ipd = $10
    # purchase model: p_buy = sigmoid(alpha*fulfilled_frac + beta*(wtp-price)/wtp - gamma)
    alpha: float = 4.0
    beta: float = 2.0
    gamma: float = 3.0
    wtp_mu: float = 3.9      # lognormal -> median ~ $49
    wtp_sigma: float = 0.5
    price_grid: tuple = tuple(range(20, 121, 10))


def sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------- world -----------------------------

@dataclass
class User:
    pains: frozenset
    wtp: float


@dataclass
class World:
    cfg: Config
    solves: dict                      # pain_id -> feature_id (hidden)
    users: list                       # list[User]
    users_by_pain: dict               # pain_id -> sorted list[user_idx]
    pain_popularity: list             # pain_id -> #users having it


def _weighted_sample_without_replacement(items, weights, k, rng):
    items = list(items)
    weights = list(weights)
    chosen = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        r = rng.uniform(0, total)
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                chosen.append(items[i])
                items.pop(i)
                weights.pop(i)
                break
    return chosen


def generate_world(seed: int, cfg: Config = None) -> World:
    cfg = cfg or Config()
    rng = random.Random(seed)

    pains = list(range(cfg.n_pains))
    features = list(range(cfg.n_features))

    # hidden bijection pain -> feature
    perm = features[:]
    rng.shuffle(perm)
    solves = {p: perm[p] for p in pains}

    # skewed pain popularity (zipf over a random rank assignment)
    ranks = pains[:]
    rng.shuffle(ranks)
    weight_by_pain = [0.0] * cfg.n_pains
    for rank_index, p in enumerate(ranks):
        weight_by_pain[p] = 1.0 / (rank_index + 1)

    users = []
    for _ in range(cfg.n_users):
        k = rng.choice([1, 2, 3])
        up = _weighted_sample_without_replacement(pains, weight_by_pain, k, rng)
        wtp = rng.lognormvariate(cfg.wtp_mu, cfg.wtp_sigma)
        users.append(User(frozenset(up), wtp))

    users_by_pain = {p: [] for p in pains}
    for idx, u in enumerate(users):
        for p in u.pains:
            users_by_pain[p].append(idx)
    pain_popularity = [len(users_by_pain[p]) for p in pains]

    return World(cfg, solves, users, users_by_pain, pain_popularity)


# ----------------------------- environment -----------------------------

class FirmEnv:
    """Gym-like single-agent env. Action is a dict:
        { "build": feature_id|None, "price": float,
          "campaigns": [ {"target": set[pain], "spend": float}, ... ] }
    Observation gives per-campaign diagnostic breakdowns (the discovery signal).
    Reward = round profit (revenue - ad spend - build cost). Deterministic.
    """

    def __init__(self, world: World):
        self.w = world
        self.cfg = world.cfg
        self.reset()

    def reset(self):
        self.cash = self.cfg.starting_cash
        self.built = {}          # feature_id -> implementation_quality
        self.price = 50.0
        self.round = 0
        self.done = False
        self.total_profit = 0.0
        return self._state_obs(per_campaign=[])

    def _state_obs(self, per_campaign):
        return {
            "round": self.round,
            "cash": round(self.cash, 2),
            "price": self.price,
            "built_features": sorted(self.built.keys()),
            "per_campaign": per_campaign,
            "done": self.done,
        }

    def _fulfilled_fraction(self, user: User) -> float:
        if not user.pains:
            return 0.0
        got = 0.0
        for p in user.pains:
            f = self.w.solves[p]
            if f in self.built:
                got += self.built[f]
        return got / len(user.pains)

    def _p_buy(self, user: User) -> float:
        ff = self._fulfilled_fraction(user)
        price_term = (user.wtp - self.price) / user.wtp
        return sigmoid(self.cfg.alpha * ff + self.cfg.beta * price_term - self.cfg.gamma)

    def _run_campaign(self, target: set, spend: float):
        cfg = self.cfg
        target = set(target)
        # matching pool: users sharing >=1 targeted pain (deterministic order by idx)
        pool = set()
        for p in target:
            pool.update(self.w.users_by_pain.get(p, []))
        pool = sorted(pool)
        if spend <= 0 or not target:
            return {"target": sorted(target), "audience": len(pool), "impressions": 0,
                    "tries": 0.0, "purchases": 0.0, "revenue": 0.0, "spend": spend}
        impressions = int(spend * cfg.impressions_per_dollar)
        reached = pool[:impressions]

        tries = 0.0
        purchases = 0.0
        for idx in reached:
            u = self.w.users[idx]
            resonance = len(target & u.pains) / len(u.pains)
            p_try = resonance            # craft = 1.0 in the structured phase
            p_buy = self._p_buy(u)
            tries += p_try
            purchases += p_try * p_buy
        revenue = purchases * self.price
        return {"target": sorted(target), "audience": len(pool), "impressions": len(reached),
                "tries": round(tries, 2), "purchases": round(purchases, 2),
                "revenue": round(revenue, 2), "spend": spend}

    def step(self, action: dict):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        cfg = self.cfg
        revenue = 0.0
        spend_total = 0.0
        build_cost = 0.0

        if action.get("price") is not None:
            self.price = float(action["price"])

        f = action.get("build")
        if f is not None and f not in self.built and self.cash >= cfg.build_cost:
            self.cash -= cfg.build_cost
            build_cost = cfg.build_cost
            self.built[f] = 1.0       # structured phase: full implementation quality

        per_campaign = []
        for c in action.get("campaigns", []) or []:
            spend = float(c.get("spend", 0.0))
            spend = max(0.0, min(spend, self.cash))   # can't overspend cash
            res = self._run_campaign(c.get("target", set()), spend)
            self.cash -= spend
            self.cash += res["revenue"]
            revenue += res["revenue"]
            spend_total += spend
            per_campaign.append(res)

        profit = revenue - spend_total - build_cost
        self.total_profit += profit
        self.round += 1
        if self.round >= cfg.horizon or self.cash < 0:
            self.done = True
        return self._state_obs(per_campaign), profit, self.done, {}


# ----------------------------- helpers -----------------------------

def best_price_for(world: World, built: dict, target_pains, sample=600):
    """Grid-search a price that maximizes expected purchase value over matching users."""
    cfg = world.cfg
    pool = set()
    for p in target_pains:
        pool.update(world.users_by_pain.get(p, []))
    pool = sorted(pool)[:sample]
    best_price, best_val = cfg.price_grid[0], -1.0
    for price in cfg.price_grid:
        val = 0.0
        for idx in pool:
            u = world.users[idx]
            ff = 0.0
            for p in u.pains:
                fe = world.solves[p]
                if fe in built:
                    ff += built[fe]
            ff = ff / len(u.pains) if u.pains else 0.0
            pb = sigmoid(cfg.alpha * ff + cfg.beta * (u.wtp - price) / u.wtp - cfg.gamma)
            val += pb * price
        if val > best_val:
            best_val, best_price = val, price
    return best_price


# ----------------------------- policies -----------------------------

class NaivePolicy:
    """No experimentation: builds random features, targets random pains, fixed spend."""

    def __init__(self, world, seed=0):
        self.w = world
        self.rng = random.Random(seed + 777)

    def reset(self):
        pass

    def act(self, env, obs):
        cfg = self.w.cfg
        f = None
        if obs["round"] < 4:
            candidates = [x for x in range(cfg.n_features) if x not in obs["built_features"]]
            if candidates:
                f = self.rng.choice(candidates)
        target = set(self.rng.sample(range(cfg.n_pains), 3))
        spend = min(700.0, max(0.0, obs["cash"] * 0.4))
        return {"build": f, "price": 50.0, "campaigns": [{"target": target, "spend": spend}]}


class OraclePolicy:
    """Strong informed reference: knows solves + popularity. Builds features for the most
    popular pains, prices optimally, and spends all cash on perfectly-targeted ads."""

    def __init__(self, world):
        self.w = world
        order = sorted(range(world.cfg.n_pains), key=lambda p: world.pain_popularity[p],
                       reverse=True)
        self.top_pains = order[:6]
        self.target_features = [world.solves[p] for p in self.top_pains]

    def reset(self):
        pass

    def act(self, env, obs):
        built = set(obs["built_features"])
        # build the next valuable unbuilt feature
        f = None
        for tf, tp in zip(self.target_features, self.top_pains):
            if tf not in built:
                f = tf
                break
        built_after = set(built)
        if f is not None:
            built_after.add(f)
        built_map = {x: 1.0 for x in built_after}
        solved_pains = [p for p, tf in zip(self.top_pains, self.target_features)
                        if tf in built_after]
        target = set(solved_pains) if solved_pains else set(self.top_pains)
        price = best_price_for(self.w, built_map, target)
        reserve = self.w.cfg.build_cost if f is not None else 0.0
        spend = max(0.0, obs["cash"] - reserve)
        return {"build": f, "price": price, "campaigns": [{"target": target, "spend": spend}]}


class ScriptedExperimenter:
    """Discovery via experiments, NO knowledge of the world:
       1) probe each pain cheaply to rank demand (try counts),
       2) build features one by one; small test campaigns reveal which pain each solves,
       3) exploit: spend remaining cash on solved popular pains at a searched price."""

    def __init__(self, world, seed=0):
        self.w = world
        self.rng = random.Random(seed + 13)

    def reset(self):
        self.pain_demand = {}        # pain -> observed tries (popularity proxy)
        self.solved = {}             # pain -> feature (discovered)
        self.tried_features = set()
        self.phase = "probe"

    def act(self, env, obs):
        cfg = self.w.cfg
        built = set(obs["built_features"])

        # ingest last round's diagnostics
        for c in obs["per_campaign"]:
            tgt = c["target"]
            if len(tgt) == 1:
                p = tgt[0]
                if obs["round"] <= 1:
                    self.pain_demand[p] = c["audience"]   # audience size = demand proxy
                if c["purchases"] > 0.5:
                    # the most recently built feature solves this pain
                    if self._last_built is not None:
                        self.solved[p] = self._last_built
        self._last_built = None

        # Phase 1 (round 0): probe demand with cheap single-pain campaigns
        if obs["round"] == 0:
            campaigns = [{"target": {p}, "spend": 10.0} for p in range(cfg.n_pains)]
            return {"build": None, "price": 50.0, "campaigns": campaigns}

        top_pains = sorted(self.pain_demand, key=self.pain_demand.get, reverse=True)
        unsolved_top = [p for p in top_pains if p not in self.solved]

        # Phase 2: build a new feature and test it against unsolved top pains
        rounds_left = cfg.horizon - obs["round"]
        still_discovering = unsolved_top and rounds_left > 2 and \
            len(self.tried_features) < cfg.n_features
        if still_discovering:
            candidates = [x for x in range(cfg.n_features)
                          if x not in built and x not in self.tried_features]
            f = candidates[0] if candidates else None
            if f is not None:
                self.tried_features.add(f)
                self._last_built = f
                # test the new feature against the top unsolved pains by audience, each
                # separately so purchases attribute cleanly -> every build is informative
                test_targets = unsolved_top[:6]
                campaigns = [{"target": {p}, "spend": 60.0} for p in test_targets]
                # also keep earning on already-solved pains
                if self.solved:
                    campaigns.append({"target": set(self.solved.keys()),
                                      "spend": max(0.0, obs["cash"] * 0.4)})
                return {"build": f, "price": 50.0, "campaigns": campaigns}

        # Phase 3: exploit solved popular pains
        if self.solved:
            target = set(self.solved.keys())
            built_map = {x: 1.0 for x in built}
            price = best_price_for(self.w, built_map, target)
            spend = max(0.0, obs["cash"])
            return {"build": None, "price": price, "campaigns": [{"target": target, "spend": spend}]}

        # fallback: nothing discovered, target best-demand pains broadly
        target = set(top_pains[:3]) if top_pains else {0, 1, 2}
        return {"build": None, "price": 50.0,
                "campaigns": [{"target": target, "spend": min(500.0, obs["cash"])}]}


# ----------------------------- run helpers -----------------------------

def run_episode(world, policy):
    env = FirmEnv(world)
    policy.reset()
    obs = env._state_obs(per_campaign=[])
    obs = env.reset()
    done = False
    while not done:
        action = policy.act(env, obs)
        obs, reward, done, _ = env.step(action)
    return env.total_profit


def main():
    cfg = Config()
    seeds = [1, 2, 3, 4, 5]
    print(f"{'seed':>4} | {'naive':>10} | {'scripted':>10} | {'oracle':>10} | "
          f"{'disc.eff':>8}")
    print("-" * 56)
    agg = {"naive": [], "scripted": [], "oracle": []}
    for s in seeds:
        world = generate_world(s, cfg)
        naive = run_episode(world, NaivePolicy(world, s))
        scripted = run_episode(world, ScriptedExperimenter(world, s))
        oracle = run_episode(world, OraclePolicy(world))
        agg["naive"].append(naive)
        agg["scripted"].append(scripted)
        agg["oracle"].append(oracle)
        disc = scripted / oracle if oracle > 0 else 0.0
        print(f"{s:>4} | {naive:>10.0f} | {scripted:>10.0f} | {oracle:>10.0f} | "
              f"{disc:>7.0%}")
    print("-" * 56)
    avg = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"{'mean':>4} | {avg['naive']:>10.0f} | {avg['scripted']:>10.0f} | "
          f"{avg['oracle']:>10.0f} | {avg['scripted']/avg['oracle']:>7.0%}")


if __name__ == "__main__":
    main()
