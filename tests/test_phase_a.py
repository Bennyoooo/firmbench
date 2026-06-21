# tests/test_phase_a.py — run from repo root: python3 tests/test_phase_a.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # CR1: repo root on path
from sim import (Config, generate_world, run_episode, FirmEnv,
                 NaivePolicy, ScriptedExperimenter, OraclePolicy)


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
