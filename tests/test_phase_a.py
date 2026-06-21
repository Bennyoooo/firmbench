# tests/test_phase_a.py — run from repo root: python3 tests/test_phase_a.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # CR1: repo root on path
from sim import (Config, generate_world, run_episode, FirmEnv, replay_profit,
                 NaivePolicy, ScriptedExperimenter, OraclePolicy)


def _popular_pain(w, cfg):
    return max(range(cfg.n_pains), key=lambda x: w.pain_popularity[x])


# ----------------------------- Step 1: Config -----------------------------

def test_flags_default_off_preserves_v1():
    c = Config()
    assert not (c.use_segments or c.use_channels or c.use_elasticity
                or c.use_quality_bar or c.use_retention), "Phase A flags must default OFF"


def test_phase_a_factory_turns_all_on():
    c = Config.phase_a()
    assert (c.use_segments and c.use_channels and c.use_elasticity
            and c.use_quality_bar and c.use_retention)


def test_phase_a_scales_budget():            # CR4: C2 budget/horizon dials must scale
    c = Config.phase_a()
    assert c.horizon > 10 and c.starting_cash > 6000.0


# ----------------------------- Step 2: segments -----------------------------

def test_segments_present_when_enabled():
    w = generate_world(1, Config.phase_a())
    assert w.segments is not None and len(w.segments) == 5
    assert all(0 <= u.segment_id < 5 for u in w.users[:50])
    # hybrid: per-user wtp varies within a segment
    seg0 = [u.wtp for u in w.users if u.segment_id == 0][:20]
    assert len(set(round(x, 3) for x in seg0)) > 1, "per-user noise expected"

def test_v1_world_unchanged_when_disabled():
    a = generate_world(3, Config())
    assert a.segments is None
    assert a.pain_popularity == generate_world(3, Config()).pain_popularity


# ----------------------------- Step 3: channels + craft -----------------------------

def test_channel_matters():
    cfg = Config.phase_a(n_users=400)
    w = generate_world(1, cfg); env = FirmEnv(w)
    p = _popular_pain(w, cfg)
    tries = [env._run_campaign({p}, 9000.0, channel=c)["tries"] for c in range(cfg.n_channels)]
    assert max(tries) > min(tries) + 1e-6        # channel choice changes conversion

def test_best_channel_is_resonance_weighted_modal():
    from collections import defaultdict
    cfg = Config.phase_a(n_users=400)
    w = generate_world(1, cfg); env = FirmEnv(w)
    p = _popular_pain(w, cfg)
    wt = defaultdict(float)
    for i in w.users_by_pain[p]:
        u = w.users[i]; wt[u.channel_pref] += len({p} & u.pains) / len(u.pains)
    modal = max(wt, key=wt.get)
    tries = {c: env._run_campaign({p}, 9000.0, channel=c)["tries"] for c in range(cfg.n_channels)}
    assert max(tries, key=tries.get) == modal    # right channel reaches the right segment

def test_craft_scales_tries_live():              # bug fix: craft applies in the LIVE funnel
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    p = _popular_pain(w, cfg)
    base = env._run_campaign({p}, 200.0, channel=0, craft=1.0)["tries"]
    half = env._run_campaign({p}, 200.0, channel=0, craft=0.5)["tries"]
    assert base > 0 and half < base

def test_channel_forwarded_through_step():        # CR2: channel/craft survive env.step()
    from collections import defaultdict
    cfg = Config.phase_a(n_users=400); w = generate_world(1, cfg)
    p = _popular_pain(w, cfg)
    wt = defaultdict(float)
    for i in w.users_by_pain[p]:
        u = w.users[i]; wt[u.channel_pref] += len({p} & u.pains) / len(u.pains)
    best = max(wt, key=wt.get); worst = min(range(cfg.n_channels), key=lambda c: wt.get(c, 0.0))
    e1 = FirmEnv(w); o1, _, _, _ = e1.step({"build": None, "price": 50.0,
        "campaigns": [{"target": {p}, "spend": 9000.0, "channel": best}]})
    e2 = FirmEnv(w); o2, _, _, _ = e2.step({"build": None, "price": 50.0,
        "campaigns": [{"target": {p}, "spend": 9000.0, "channel": worst}]})
    assert o1["per_campaign"][0]["tries"] > o2["per_campaign"][0]["tries"]


# --------------------- Steps 4-5: elasticity + quality_bar/bounce ---------------------

def test_elasticity_uses_per_user_beta():        # CR3: fails on v1 (shared beta), passes after
    import sim as S
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    u = max(w.users[:500], key=lambda x: abs(x.elasticity - cfg.beta))
    assert abs(u.elasticity - cfg.beta) > 1e-3, "need a user with non-default elasticity"
    for p in u.pains:
        env.built[w.solves[p]] = 1.0
    env.price = 90.0
    ff = env._fulfilled_fraction(u); pt = (u.wtp - env.price) / u.wtp
    gate = S.sigmoid(cfg.quality_gate_k * (ff - u.quality_bar))           # quality_bar also on under phase_a
    per_user = S.sigmoid(cfg.alpha * ff + u.elasticity * pt - cfg.gamma) * gate
    v1_value = S.sigmoid(cfg.alpha * ff + cfg.beta * pt - cfg.gamma)      # v1: shared beta, no gate
    got = env._p_buy(u)
    assert abs(got - per_user) < 1e-9            # _p_buy uses beta_u = user.elasticity
    assert abs(got - v1_value) > 1e-6            # ...and that differs from v1

def test_quality_bar_emits_distinct_bounce_signal():
    from collections import defaultdict
    cfg = Config.phase_a(n_users=400); w = generate_world(1, cfg); env = FirmEnv(w)
    p = _popular_pain(w, cfg)
    env.built[w.solves[p]] = 0.05                # built but LOW quality
    env.price = 25.0                             # cheap, so price isn't the blocker
    wt = defaultdict(float)
    for i in w.users_by_pain[p]:
        wt[w.users[i].channel_pref] += 1
    ch = max(wt, key=wt.get)
    r = env._run_campaign({p}, 9000.0, channel=ch)
    assert r["bounced_quality"] > r["bounced_price"]     # failure attributed to quality, not price
    assert r["purchases"] < 0.5 * r["tries"]             # low quality suppresses conversion


# ----------------------------- Step 6: LTV / retention -----------------------------

def test_base_persists_and_churns():
    from collections import defaultdict
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    p = _popular_pain(w, cfg)
    wt = defaultdict(float)
    for i in w.users_by_pain[p]:
        wt[w.users[i].channel_pref] += 1
    ch = max(wt, key=wt.get)
    # acquire subscribers: build the solving feature, fair price, big campaign on modal channel
    env.step({"build": w.solves[p], "price": 40.0,
              "campaigns": [{"target": {p}, "spend": 3000.0, "channel": ch}]})
    acquired = sum(env.base.values())
    assert acquired > 0, "should acquire subscribers into the base"
    # next round, no campaigns: recurring revenue > 0 and base decays via churn
    obs, profit, done, _ = env.step({"build": None, "price": 40.0, "campaigns": []})
    assert profit > 0, "recurring revenue from retained subscribers"
    assert sum(env.base.values()) < acquired, "churn applied to the base"

def test_segment_id_not_leaked_to_agent():
    cfg = Config.phase_a(); w = generate_world(1, cfg); env = FirmEnv(w)
    p = _popular_pain(w, cfg)
    obs, _, _, _ = env.step({"build": None, "price": 40.0,
                             "campaigns": [{"target": {p}, "spend": 500.0, "channel": 0}]})
    assert obs["per_campaign"], "campaign should be reported"
    assert all(not k.startswith("_") for k in obs["per_campaign"][0]), "no internal keys leaked"


# ----------------------------- Step 10: ablation gate -----------------------------

def test_ablation_gate_reports_each_latent():
    from sim import ablation_gate
    rows = ablation_gate(seeds=[1, 2])
    keys = {r["config"] for r in rows}
    assert {"v1", "+segments", "+channels", "+elasticity", "+quality_bar",
            "+retention", "full"} <= keys
    for r in rows:
        assert "naive" in r and "scripted" in r and "oracle" in r and "gate" in r


# ----------------------------- Steps 11-13: holdout grading -----------------------------

def _play_capturing_log(world, policy):
    env = FirmEnv(world); policy.reset(); obs = env.reset()
    log, done = [], False
    while not done:
        a = policy.act(env, obs)
        log.append({"build": a.get("build"), "price": a.get("price", env.price),
                    "campaigns": [{"target": sorted(c.get("target", set())),
                                   "spend": float(c.get("spend", 0.0)),
                                   "channel": c.get("channel", 0)} for c in a.get("campaigns", [])]})
        obs, _, done, _ = env.step(a)
    return env.total_profit, log

def test_replay_profit_matches_live_on_full_set():
    # replay_profit is exact: replaying ALL users at scale 1.0 reproduces the live profit.
    w = generate_world(7, Config.phase_a())
    reported, log = _play_capturing_log(w, ScriptedExperimenter(w, 7))
    full = replay_profit(w, list(range(len(w.users))), log, spend_scale=1.0)
    assert abs(full - reported) < 1.0

def test_replay_profit_deterministic():
    w = generate_world(2, Config.phase_a())
    _, log = _play_capturing_log(w, ScriptedExperimenter(w, 2))
    n = len(w.users); ho = list(range(int(n * 0.8), n))
    assert replay_profit(w, ho, log, 0.2) == replay_profit(w, ho, log, 0.2)


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"ok   {name}")
            except Exception:
                fails += 1; print(f"FAIL {name}"); traceback.print_exc()
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
