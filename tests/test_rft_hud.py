# tests/test_rft_hud.py — run from repo root: python3 tests/test_rft_hud.py
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from rft_hud import (MockBackend, rl_loop, disc_eff, local_ref, _check_groups,
                     CFG)
from sim import generate_world, run_episode, OraclePolicy, NaivePolicy


# ----------------------------- reward parity with env.py -----------------------------

def test_disc_eff_matches_env_formula():
    # oracle scores ~1.0 (it IS the ceiling), naive scores low — both clipped to [0,1]
    w = generate_world(42, CFG)
    oracle_r = disc_eff(w, run_episode(w, OraclePolicy(w)))
    naive_r = disc_eff(w, run_episode(w, NaivePolicy(w, 42)))
    assert 0.0 <= naive_r <= oracle_r <= 1.0
    assert oracle_r > 0.95, f"oracle should be near the ceiling, got {oracle_r}"


def test_references_ranked():
    seeds = list(range(100, 104))
    oracle = local_ref(lambda w, s: OraclePolicy(w), seeds)
    naive = local_ref(lambda w, s: NaivePolicy(w, s), seeds)
    assert oracle > naive, "oracle must beat naive on disc_eff"


# ----------------------------- GRPO group divisibility guard -----------------------------

def test_check_groups_rejects_incomplete_groups():
    _check_groups(12, 6)        # 12 = 2 full groups of 6 — ok
    _check_groups(8, None)      # no grouping — ok
    try:
        _check_groups(10, 4)    # 10 not divisible by 4 — must raise
    except ValueError:
        return
    raise AssertionError("expected ValueError for an incomplete final GRPO group")


# ----------------------------- the loop bends the curve -----------------------------

def test_mock_rl_loop_bends_curve():
    curve = asyncio.run(rl_loop(
        MockBackend(skill0=0.15),
        train_seeds=[1, 2, 3, 4], eval_seeds=[100, 101, 102, 103],
        steps=4, group_size=4, lr=1e-5, loss_fn="importance_sampling",
        temperature=0.7, out_dir="rft_hud_out"))
    base, final = curve[0]["eval"], curve[-1]["eval"]
    assert final > base + 0.2, f"RL curve should bend up: {base:.3f} -> {final:.3f}"
    # every step records its grouped rollout count (4 seeds x group_size 4)
    assert all(p["n_rollouts"] == 16 for p in curve[1:])


def test_curve_json_written():
    import json
    asyncio.run(rl_loop(
        MockBackend(), train_seeds=[1, 2], eval_seeds=[100, 101],
        steps=2, group_size=2, lr=1e-5, loss_fn="importance_sampling",
        temperature=0.7, out_dir="rft_hud_out"))
    with open("rft_hud_out/curve.json") as f:
        data = json.load(f)
    assert "curve" in data and "references" in data and "config" in data
    assert data["config"]["loss_fn"] == "importance_sampling"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
