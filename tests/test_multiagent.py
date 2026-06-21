# tests/test_multiagent.py — run from repo root: python3 tests/test_multiagent.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from sim import (Config, generate_world, FirmEnv, run_episode, ablation_gate,
                 OraclePolicy, ScriptedExperimenter)
from multiagent import (Blackboard, MultiAgentFirmEnv, ROLES, ROLE_READS,
                        ScriptedTeam, NaiveTeam, run_team_episode,
                        coordination_tax, coordination_gate)

PA = Config.phase_a()

# ----------------------------- single-agent stays intact -----------------------------

def test_single_agent_env_unchanged_after_import():
    # importing multiagent must not perturb the v1 sim: a v1 ablation row still PASSes.
    rows = ablation_gate(seeds=[1, 2])
    v1 = next(r for r in rows if r["config"] == "v1")
    assert v1["gate"] == "PASS", v1
    # and a v1 world is byte-identical
    assert generate_world(3, Config()).pain_popularity == generate_world(3, Config()).pain_popularity
    assert generate_world(3, Config()).segments is None


# ----------------------------- round-trip equivalence with FirmEnv.step -----------------------------

def _assemble_via_team(world, action, budget=1e12):
    """Submit `action` through the four role slices (no budget clamp) and commit one round."""
    menv = MultiAgentFirmEnv(world)
    menv.reset()
    menv.submit("coordinator", {"budget": budget})
    menv.submit("builder", {"build": action.get("build")})
    menv.submit("pricer", {"price": action.get("price", menv.env.price)})
    menv.submit("marketer", {"campaigns": action.get("campaigns", [])})
    return menv.commit()

def test_commit_matches_firmenv_step_single_round():
    w = generate_world(5, PA)
    p = max(range(PA.n_pains), key=lambda x: w.pain_popularity[x])
    action = {"build": w.solves[p], "price": 45.0,
              "campaigns": [{"target": {p}, "spend": 800.0, "channel": 1, "craft": 0.8}]}
    # team path
    (_obs, team_profit, _done, committed) = _assemble_via_team(w, action)
    # raw single-agent path on an identical world
    env = FirmEnv(generate_world(5, PA)); env.reset()
    raw_obs, raw_profit, _d, _ = env.step({"build": action["build"], "price": action["price"],
        "campaigns": [{"target": {p}, "spend": 800.0, "channel": 1, "craft": 0.8}]})
    assert abs(team_profit - raw_profit) < 1e-6, (team_profit, raw_profit)
    # per-campaign diagnostics match exactly (tries/purchases/bounce/revenue)
    tc = _obs["marketer"]["per_campaign"][0]; rc = raw_obs["per_campaign"][0]
    for k in ("tries", "purchases", "bounced_quality", "bounced_price", "revenue", "audience"):
        assert abs(tc[k] - rc[k]) < 1e-6, (k, tc[k], rc[k])

def test_commit_matches_firmenv_step_multi_round():
    w1 = generate_world(7, PA); w2 = generate_world(7, PA)
    p = max(range(PA.n_pains), key=lambda x: w1.pain_popularity[x])
    menv = MultiAgentFirmEnv(w1); menv.reset()
    env = FirmEnv(w2); env.reset()
    plan = [
        {"build": w1.solves[p], "price": 50.0, "campaigns": [{"target": {p}, "spend": 500.0, "channel": 0, "craft": 1.0}]},
        {"build": None, "price": 40.0, "campaigns": [{"target": {p}, "spend": 1200.0, "channel": 0, "craft": 1.0}]},
        {"build": None, "price": 40.0, "campaigns": [{"target": {p}, "spend": 900.0, "channel": 0, "craft": 1.0}]},
    ]
    for a in plan:
        menv.submit("coordinator", {"budget": 1e12})
        menv.submit("builder", {"build": a["build"]})
        menv.submit("pricer", {"price": a["price"]})
        menv.submit("marketer", {"campaigns": a["campaigns"]})
        menv.commit()
        env.step(a)
    assert abs(menv.total_profit - env.total_profit) < 1e-6, (menv.total_profit, env.total_profit)
    assert abs(menv.env.cash - env.cash) < 1e-6

def test_budget_clamps_marketer_spend():
    w = generate_world(2, PA)
    menv = MultiAgentFirmEnv(w); menv.reset()
    menv.submit("coordinator", {"budget": 100.0})
    menv.submit("builder", {"build": None})
    menv.submit("pricer", {"price": 50.0})
    menv.submit("marketer", {"campaigns": [{"target": {0}, "spend": 5000.0, "channel": 0}]})
    action = menv.assemble()
    assert action["campaigns"][0]["spend"] == 100.0, "spend must be clamped to the budget"


# ----------------------------- coordination tax: messages buy coordination -----------------------------

def test_scripted_team_beats_naive_team():
    for s in (1, 2, 3):
        w = generate_world(s, PA)
        _, scr = run_team_episode(w, ScriptedTeam(w, s))
        _, naive = run_team_episode(w, NaiveTeam(w, s))
        assert scr > naive, f"seed {s}: scripted_team {scr:.0f} should beat naive_team {naive:.0f}"

def test_coordination_tax_smaller_when_coordinated():
    w = generate_world(1, PA)
    scr = coordination_tax(w, lambda w, s: ScriptedTeam(w, s))
    naive = coordination_tax(w, lambda w, s: NaiveTeam(w, s))
    assert scr["tax"] > 0, "even a coordinated team pays some tax vs the full-info oracle"
    assert scr["tax"] < naive["tax"], "coordination (messages) must shrink the tax"
    assert 0.0 <= scr["team_disc_eff"] <= 1.0

def test_coordination_gate_passes():
    rows = coordination_gate(seeds=[1, 2, 3])
    assert rows[0]["gate"] == "PASS", rows[0]
    assert rows[0]["naive"] < rows[0]["scripted"] <= rows[0]["oracle"]


# ----------------------------- partial observability: slicing + no leak -----------------------------

def test_only_marketer_sees_campaign_diagnostics():
    w = generate_world(1, PA)
    menv = MultiAgentFirmEnv(w); menv.reset()
    # run one real round so diagnostics exist
    menv.submit("coordinator", {"budget": 1e9}); menv.submit("builder", {"build": None})
    menv.submit("pricer", {"price": 50.0})
    menv.submit("marketer", {"campaigns": [{"target": {0, 1}, "spend": 500.0, "channel": 0}]})
    menv.commit()
    assert "per_campaign" in menv.role_obs("marketer")
    for r in ("coordinator", "builder", "pricer"):
        assert "per_campaign" not in menv.role_obs(r), f"{r} must NOT see per-campaign diagnostics"

def test_message_slicing_respects_role_reads():
    w = generate_world(1, PA)
    menv = MultiAgentFirmEnv(w); menv.reset()
    menv.submit("coordinator", {"budget": 1e9}, message="PHASE: discover")
    menv.submit("builder", {"build": 3}, message="BUILT: feature 3")
    # marketer reads coordinator+builder -> sees both
    mm = menv.role_obs("marketer")["messages"]
    assert any(m["role"] == "builder" for m in mm) and any(m["role"] == "coordinator" for m in mm)
    # builder reads only coordinator -> must NOT see its own/other builder-targeted slices
    bm = menv.role_obs("builder")["messages"]
    assert all(m["role"] in ROLE_READS["builder"] for m in bm)
    assert all(m["role"] != "builder" for m in bm)
    # pricer reads only coordinator (current round) -> must not see the builder message
    pm = menv.role_obs("pricer")["messages"]
    assert all(m["role"] == "coordinator" for m in pm)

def _walk_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k); yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk_strings(v)
    else:
        yield str(obj)

def test_no_hidden_market_info_leaks_to_any_role():
    forbidden = {"segment_id", "segment", "segments", "wtp", "solves", "pain_affinity",
                 "elasticity", "quality_bar", "channel_pref", "users", "pain_popularity"}
    w = generate_world(1, PA)
    menv = MultiAgentFirmEnv(w); menv.reset()
    # run a couple of rounds so messages + diagnostics populate the obs
    for _ in range(2):
        menv.submit("coordinator", {"budget": 1e9}, message="PHASE: discover")
        menv.submit("builder", {"build": 0}, message="BUILT: feature 0")
        menv.submit("pricer", {"price": 50.0})
        menv.submit("marketer", {"campaigns": [{"target": {0, 1}, "spend": 300.0, "channel": 0}]},
                    message="TARGETS: [0, 1]")
        menv.commit()
    for role in ROLES:
        obs = menv.role_obs(role)
        keys = set()
        if isinstance(obs, dict):
            stack = [obs]
            while stack:
                d = stack.pop()
                if isinstance(d, dict):
                    keys |= set(d.keys()); stack += list(d.values())
                elif isinstance(d, (list, tuple, set)):
                    stack += list(d)
        leaked = forbidden & keys
        assert not leaked, f"role {role} obs leaked hidden keys: {leaked}"


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
