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
    impressions_per_dollar: float = 0.2   # $5 per reached user — informative probes at $50
    # purchase model: p_buy = sigmoid(alpha*fulfilled_frac + beta*(wtp-price)/wtp - gamma)
    alpha: float = 4.0
    beta: float = 2.0
    gamma: float = 3.0
    wtp_mu: float = 3.9      # lognormal -> median ~ $49
    wtp_sigma: float = 0.5
    price_grid: tuple = tuple(range(20, 121, 10))

    # ── Phase A ablation flags (default OFF → v1 reproducible + clean baseline) ──
    use_segments: bool = False
    use_channels: bool = False
    use_elasticity: bool = False
    use_quality_bar: bool = False
    use_retention: bool = False
    # population / channel structure
    n_segments: int = 5
    n_channels: int = 3
    channel_fit_off: float = 0.25      # reach multiplier on the wrong channel
    # elasticity: per-segment beta mean + per-user noise (hybrid population)
    elasticity_mu: float = 2.0
    elasticity_sigma: float = 0.5
    # quality_bar: per-segment min fulfilled-fraction to convert (soft gate)
    quality_bar_mu: float = 0.3
    quality_bar_sigma: float = 0.1
    quality_gate_k: float = 8.0        # softness of the bar (higher = harder gate)
    # retention / subscription
    subscription: bool = True
    churn_base: float = 0.10
    churn_price_coef: float = 0.30     # churn rises when price > wtp
    churn_quality_coef: float = 0.30   # churn rises when fulfilled < quality_bar

    @classmethod
    def phase_a(cls, scale_budget=True, **overrides):
        """All Phase A latents ON. scale_budget scales horizon/cash with channels
        (C2) so an ablation FAIL signals unobservability, not just running out of money."""
        base = dict(use_segments=True, use_channels=True, use_elasticity=True,
                    use_quality_bar=True, use_retention=True)
        if scale_budget:
            nch = overrides.get("n_channels", 3)
            base["horizon"] = 10 + 2 * nch
            base["starting_cash"] = 6000.0 * nch
        base.update(overrides)
        return cls(**base)


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
    segment_id: int = -1
    elasticity: float = None     # per-user price sensitivity (segment mean + noise)
    channel_pref: int = 0        # channel that reaches this user best
    quality_bar: float = 0.0     # min fulfilled-fraction to convert


@dataclass
class Segment:
    """Hidden persona: a correlated pain cluster + shared economics (Phase A)."""
    pain_affinity: list      # weights over pains -> correlated cluster
    wtp_mu: float
    wtp_sigma: float
    elasticity_mu: float     # per-segment price sensitivity (per-user noise added on top)
    channel_pref: int        # which channel reaches this segment best
    quality_bar: float       # min fulfilled-fraction to convert
    churn_base: float        # baseline churn rate (segment-varied)
    weight: float            # share of population (the new "popularity" to discover)


@dataclass
class World:
    cfg: Config
    solves: dict                      # pain_id -> feature_id (hidden)
    users: list                       # list[User]
    users_by_pain: dict               # pain_id -> sorted list[user_idx]
    pain_popularity: list             # pain_id -> #users having it
    pain_names: list = None           # pain_id -> human name (cosmetic)
    pain_keywords: dict = None        # pain_id -> list[str] for NL matching
    feature_names: list = None        # feature_id -> human name (cosmetic)
    feature_keywords: dict = None     # feature_id -> list[str] for NL matching
    segments: list = None             # list[Segment] (Phase A; None when use_segments off)


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


def _weighted_pick(rng, items, weights):
    """Pick a single item proportional to weights (deterministic given rng)."""
    items = list(items)
    total = sum(weights)
    r = rng.uniform(0, total)
    upto = 0.0
    for it, w in zip(items, weights):
        upto += w
        if upto >= r:
            return it
    return items[-1]


def _make_segments(rng, cfg):
    """K hidden personas: each a skewed (correlated) pain affinity + shared economics,
    a preferred channel, and a zipf population weight (which segments dominate)."""
    segs = []
    seg_ranks = list(range(cfg.n_segments))
    rng.shuffle(seg_ranks)
    for s in range(cfg.n_segments):
        ranks = list(range(cfg.n_pains))
        rng.shuffle(ranks)
        affinity = [0.0] * cfg.n_pains
        for r, p in enumerate(ranks):
            affinity[p] = 1.0 / (r + 1)
        segs.append(Segment(
            pain_affinity=affinity,
            wtp_mu=cfg.wtp_mu + rng.uniform(-0.3, 0.3),
            wtp_sigma=cfg.wtp_sigma,
            elasticity_mu=max(0.2, rng.gauss(cfg.elasticity_mu, 0.4)),
            channel_pref=rng.randrange(cfg.n_channels),
            quality_bar=min(0.9, max(0.0, rng.gauss(cfg.quality_bar_mu, cfg.quality_bar_sigma))),
            churn_base=min(0.4, max(0.02, rng.gauss(cfg.churn_base, 0.04))),
            weight=1.0 / (seg_ranks[s] + 1),
        ))
    return segs


def _resample_users_from_segments(rng, cfg, segments):
    """Hybrid population: persona backbone (pains/channel/quality_bar) + per-user
    noise on wtp/elasticity."""
    pains = list(range(cfg.n_pains))
    seg_weights = [s.weight for s in segments]
    users = []
    for _ in range(cfg.n_users):
        s_idx = _weighted_pick(rng, range(cfg.n_segments), seg_weights)
        seg = segments[s_idx]
        k = rng.choice([1, 2, 3])
        up = _weighted_sample_without_replacement(pains, list(seg.pain_affinity), k, rng)
        wtp = rng.lognormvariate(seg.wtp_mu, seg.wtp_sigma)
        elasticity = max(0.1, rng.gauss(seg.elasticity_mu, cfg.elasticity_sigma))
        users.append(User(frozenset(up), wtp, segment_id=s_idx, elasticity=elasticity,
                          channel_pref=seg.channel_pref, quality_bar=seg.quality_bar))
    return users


# ----------------------------- name / keyword pools -----------------
# Sampled per-seed to give the NL artifact layer meaningful labels.
# Each entry is (name, [keywords]). Pools are larger than n_pains/n_features
# so domain randomization produces different names per episode.

_PAIN_POOL = [
    ("slow onboarding", ["onboarding", "setup", "getting started", "first time"]),
    ("billing errors", ["billing", "invoice", "charge", "payment error"]),
    ("missing integrations", ["integration", "connect", "api", "third-party"]),
    ("poor mobile experience", ["mobile", "app", "responsive", "phone"]),
    ("data export limitations", ["export", "download", "csv", "data portability"]),
    ("confusing permissions", ["permissions", "access", "roles", "authorization"]),
    ("lack of reporting", ["reporting", "analytics", "dashboard", "metrics"]),
    ("slow page loads", ["slow", "performance", "loading", "speed"]),
    ("no offline mode", ["offline", "sync", "disconnected", "local"]),
    ("complex pricing", ["pricing", "plan", "subscription", "cost"]),
    ("poor search", ["search", "find", "filter", "lookup"]),
    ("no collaboration", ["collaboration", "team", "sharing", "multi-user"]),
    ("weak security", ["security", "encryption", "vulnerability", "breach"]),
    ("no notifications", ["notification", "alert", "reminder", "update"]),
    ("limited customization", ["customization", "theme", "branding", "configure"]),
    ("difficult migration", ["migration", "import", "transfer", "switching"]),
]

_FEATURE_POOL = [
    ("quick-start wizard", ["wizard", "onboarding", "setup", "walkthrough"]),
    ("payment dashboard", ["payment", "billing", "invoice", "transaction"]),
    ("API connector", ["api", "integration", "webhook", "connector"]),
    ("mobile app", ["mobile", "app", "ios", "android"]),
    ("data exporter", ["export", "csv", "download", "report"]),
    ("role manager", ["role", "permission", "access", "admin"]),
    ("analytics suite", ["analytics", "dashboard", "metrics", "charts"]),
    ("CDN accelerator", ["cdn", "cache", "speed", "performance"]),
    ("offline sync engine", ["offline", "sync", "local", "cache"]),
    ("plan configurator", ["plan", "pricing", "tier", "subscription"]),
    ("smart search", ["search", "elasticsearch", "filter", "autocomplete"]),
    ("team workspace", ["workspace", "collaboration", "team", "shared"]),
    ("security hardener", ["security", "encryption", "audit", "firewall"]),
    ("notification center", ["notification", "alert", "push", "email"]),
    ("theme engine", ["theme", "customization", "branding", "style"]),
    ("migration toolkit", ["migration", "import", "converter", "transfer"]),
]


def _sample_names(rng, n, pool):
    """Pick n unique (name, keywords) from pool; return (names_list, keywords_dict)."""
    selected = rng.sample(pool, min(n, len(pool)))
    # pad if pool is smaller than n (shouldn't happen with 16-entry pools and n=8)
    while len(selected) < n:
        selected.append((f"item-{len(selected)}", [f"keyword-{len(selected)}"]))
    names = [s[0] for s in selected]
    keywords = {i: list(s[1]) for i, s in enumerate(selected)}
    return names, keywords


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

    # Phase A (gated): resample the population from hidden personas. Runs ONLY when
    # use_segments is on, so the flags-off RNG stream + world stay byte-identical to v1.
    segments = None
    if cfg.use_segments:
        segments = _make_segments(rng, cfg)
        users = _resample_users_from_segments(rng, cfg, segments)
        users_by_pain = {p: [] for p in pains}
        for idx, u in enumerate(users):
            for p in u.pains:
                users_by_pain[p].append(idx)
        pain_popularity = [len(users_by_pain[p]) for p in pains]

    # cosmetic names + keywords for the NL artifact layer (Phase 3).
    # Generated AFTER demography so the existing RNG sequence is preserved
    # and all prior seeds produce identical sim results.
    pain_names, pain_keywords = _sample_names(rng, cfg.n_pains, _PAIN_POOL)
    feature_names, feature_keywords = _sample_names(rng, cfg.n_features, _FEATURE_POOL)

    return World(cfg, solves, users, users_by_pain, pain_popularity,
                 pain_names, pain_keywords, feature_names, feature_keywords,
                 segments=segments)


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
        self.base = {}           # segment_id -> expected active subscribers (LTV; latent)
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
        cfg = self.cfg
        ff = self._fulfilled_fraction(user)
        price_term = (user.wtp - self.price) / user.wtp
        beta_u = user.elasticity if (cfg.use_elasticity and user.elasticity is not None) else cfg.beta
        p = sigmoid(cfg.alpha * ff + beta_u * price_term - cfg.gamma)
        if cfg.use_quality_bar:
            p *= sigmoid(cfg.quality_gate_k * (ff - user.quality_bar))   # soft quality gate
        return p

    def _run_campaign(self, target: set, spend: float, channel: int = 0, craft: float = 1.0):
        cfg = self.cfg
        target = set(target)
        # matching pool: users sharing >=1 targeted pain (deterministic order by idx)
        pool = set()
        for p in target:
            pool.update(self.w.users_by_pain.get(p, []))
        pool = sorted(pool)
        if spend <= 0 or not target:
            return {"target": sorted(target), "audience": len(pool), "impressions": 0,
                    "tries": 0.0, "purchases": 0.0, "bounced_quality": 0.0,
                    "bounced_price": 0.0, "revenue": 0.0, "spend": spend}
        impressions = int(spend * cfg.impressions_per_dollar)
        reached = pool[:impressions]

        tries = 0.0
        purchases = 0.0
        bounced_quality = 0.0      # resonated + reached, but blocked by the quality bar
        bounced_price = 0.0        # passed quality, but lost on price
        by_segment = {}            # internal: new conversions per segment (for LTV base)
        for idx in reached:
            u = self.w.users[idx]
            resonance = len(target & u.pains) / len(u.pains)
            # channel: the right channel converts at full weight, the wrong channel is
            # downweighted (use_channels off -> 1.0, i.e. exactly v1).
            ch = (1.0 if u.channel_pref == channel else cfg.channel_fit_off) \
                if cfg.use_channels else 1.0
            p_try = craft * ch * resonance       # craft now applies in the LIVE funnel (bug fix)
            p_buy = self._p_buy(u)
            contrib = p_try * p_buy
            tries += p_try
            purchases += contrib
            if cfg.use_retention:
                by_segment[u.segment_id] = by_segment.get(u.segment_id, 0.0) + contrib
            # bounce-reason diagnostics: make quality vs price failures separately
            # observable (the C1 separability fix). Heuristic signals, not exact partition.
            if cfg.use_quality_bar or cfg.use_elasticity:
                ff = self._fulfilled_fraction(u)
                beta_u = u.elasticity if (cfg.use_elasticity and u.elasticity is not None) else cfg.beta
                gate = sigmoid(cfg.quality_gate_k * (ff - u.quality_bar)) if cfg.use_quality_bar else 1.0
                price_accept = sigmoid(beta_u * (u.wtp - self.price) / u.wtp)
                bounced_quality += p_try * (1.0 - gate)
                bounced_price += p_try * gate * (1.0 - price_accept)
        revenue = purchases * self.price
        return {"target": sorted(target), "audience": len(pool), "impressions": len(reached),
                "tries": round(tries, 2), "purchases": round(purchases, 2),
                "bounced_quality": round(bounced_quality, 2),
                "bounced_price": round(bounced_price, 2),
                "_by_segment": by_segment,                 # internal: stripped before the agent sees it
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

        # LTV: existing subscribers re-pay (recurring revenue) at the current price.
        retention = cfg.use_retention and cfg.subscription
        if retention and self.base:
            recurring = sum(self.base.values()) * self.price
            self.cash += recurring
            revenue += recurring

        per_campaign = []
        new_subs = {}            # segment_id -> conversions acquired this round
        for c in action.get("campaigns", []) or []:
            spend = float(c.get("spend", 0.0))
            spend = max(0.0, min(spend, self.cash))   # can't overspend cash
            res = self._run_campaign(c.get("target", set()), spend,
                                     channel=c.get("channel", 0), craft=c.get("craft", 1.0))
            self.cash -= spend
            self.cash += res["revenue"]
            revenue += res["revenue"]
            spend_total += spend
            for s, q in res.get("_by_segment", {}).items():
                new_subs[s] = new_subs.get(s, 0.0) + q
            # never leak internal (underscore) keys — e.g. per-segment attribution — to the agent
            per_campaign.append({k: v for k, v in res.items() if not k.startswith("_")})

        if retention:
            for s, q in new_subs.items():
                self.base[s] = self.base.get(s, 0.0) + q
            self._apply_churn()       # responsive + segment-varied

        profit = revenue - spend_total - build_cost
        self.total_profit += profit
        self.round += 1
        if self.round >= cfg.horizon or self.cash < 0:
            self.done = True
        return self._state_obs(per_campaign), profit, self.done, {}

    def _seg_avg_ff(self, seg) -> float:
        """Segment-level 'fraction of needs met' (affinity-weighted), drives churn."""
        tot = sum(seg.pain_affinity)
        if tot <= 0:
            return 0.0
        got = 0.0
        for p, w in enumerate(seg.pain_affinity):
            got += w * self.built.get(self.w.solves[p], 0.0)
        return got / tot

    def _apply_churn(self):
        """Per-segment churn responsive to price (vs segment wtp) and quality (vs bar)."""
        cfg = self.cfg
        for s in list(self.base.keys()):
            if self.w.segments and 0 <= s < len(self.w.segments):
                seg = self.w.segments[s]
                seg_wtp = math.exp(seg.wtp_mu)        # segment median willingness-to-pay
                price_pressure = cfg.churn_price_coef * max(0.0, (self.price - seg_wtp) / seg_wtp)
                quality_gap = cfg.churn_quality_coef * max(0.0, seg.quality_bar - self._seg_avg_ff(seg))
                churn_eff = min(0.95, seg.churn_base + price_pressure + quality_gap)
            else:
                churn_eff = cfg.churn_base            # flat churn when segments are off (ablation)
            self.base[s] *= (1.0 - churn_eff)


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
            beta_u = u.elasticity if (cfg.use_elasticity and u.elasticity is not None) else cfg.beta
            pb = sigmoid(cfg.alpha * ff + beta_u * (u.wtp - price) / u.wtp - cfg.gamma)
            val += pb * price
        if val > best_val:
            best_val, best_price = val, price
    return best_price


def _best_channel(world: World, target_pains):
    """Resonance-weighted modal channel_pref among a target's matching users (the
    channel that maximizes conversion under the funnel). 0 when channels are off."""
    if not world.segments:
        return 0
    tset = set(target_pains)
    pool = set()
    for p in target_pains:
        pool.update(world.users_by_pain.get(p, []))
    wt = {}
    for idx in pool:
        u = world.users[idx]
        if u.pains:
            wt[u.channel_pref] = wt.get(u.channel_pref, 0.0) + len(tset & u.pains) / len(u.pains)
    return max(wt, key=wt.get) if wt else 0


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
        channel = self.rng.randrange(cfg.n_channels) if cfg.use_channels else 0  # no extra draw in v1
        return {"build": f, "price": 50.0,
                "campaigns": [{"target": target, "spend": spend, "channel": channel}]}


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
        channel = _best_channel(self.w, target)
        return {"build": f, "price": price,
                "campaigns": [{"target": target, "spend": spend, "channel": channel}]}


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


def ablation_gate(seeds=(1, 2, 3, 4, 5)):
    """The reactive-fix bisection tool. Runs naive/scripted/oracle under v1, each single
    Phase A latent (on top of segments), and full. Gate PASS iff naive < scripted < oracle;
    WARN if scripted > oracle (reference too weak); FAIL otherwise (the latent broke
    discovery for the scripted experimenter -> fix its observation/strategy)."""
    BUD = dict(horizon=16, starting_cash=18000.0)   # uniform scaled budget for Phase A rows (C2)
    configs = [
        ("v1", Config()),
        ("+segments", Config(use_segments=True, **BUD)),
        ("+channels", Config(use_segments=True, use_channels=True, **BUD)),
        ("+elasticity", Config(use_segments=True, use_elasticity=True, **BUD)),
        ("+quality_bar", Config(use_segments=True, use_quality_bar=True, **BUD)),
        ("+retention", Config(use_segments=True, use_retention=True, **BUD)),
        ("full", Config.phase_a()),
    ]
    rows = []
    print(f"{'config':>13} | {'naive':>11} | {'scripted':>11} | {'oracle':>11} | {'gate':>5}")
    print("-" * 66)
    for name, cfg in configs:
        nl, sl, ol = [], [], []
        for s in seeds:
            world = generate_world(s, cfg)
            nl.append(run_episode(world, NaivePolicy(world, s)))
            sl.append(run_episode(world, ScriptedExperimenter(world, s)))
            ol.append(run_episode(world, OraclePolicy(world)))
        nv, sc, orc = sum(nl) / len(nl), sum(sl) / len(sl), sum(ol) / len(ol)
        gate = "WARN" if sc > orc else ("PASS" if nv < sc else "FAIL")
        rows.append({"config": name, "naive": nv, "scripted": sc, "oracle": orc, "gate": gate})
        print(f"{name:>13} | {nv:>11.0f} | {sc:>11.0f} | {orc:>11.0f} | {gate:>5}")
    return rows


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
