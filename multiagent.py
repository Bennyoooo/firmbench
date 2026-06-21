"""
FirmBench — Phase D: the multi-agent layer (additive wrapper over the single-agent sim).

Splits the single firm-agent into 4 cooperating ROLE-AGENTS that share one firm but see
only their slice and must COMMUNICATE (via a shared blackboard) to coordinate:

    Coordinator — sets the per-round marketing BUDGET + a directive; commits the round.
    Builder     — chooses what to BUILD; tells the Marketer what it built.
    Pricer      — sets the PRICE.
    Marketer    — runs CAMPAIGNS (the only role that sees per-campaign diagnostics).

Partial observability is the point: only the Marketer sees demand/solve signals, and it
only knows which feature exists if the Builder *tells* it. If a role can't observe what it
needs (directly or via a message) it can't coordinate — that waste is the COORDINATION TAX.

Design discipline (mirrors Phase A):
  * The single-agent env (sim.py) is untouched — this is an opt-in wrapper.
  * `ScriptedTeam` and `NaiveTeam` share the SAME role policies; they differ ONLY in whether
    the roles READ the blackboard. So `coordination_tax(naive) - coordination_tax(scripted)`
    isolates exactly the value of the messages (a clean matched quad).
  * Deterministic: no RNG outside the seeded policies; the funnel stays expectation-based.

Run `python3 multiagent.py` to print the coordination gate:
    expect  naive_team  <  scripted_team  <=  oracle   (scripted tax  <  naive tax).
"""

import re
from dataclasses import dataclass

from sim import (Config, World, FirmEnv, generate_world, OraclePolicy,
                 ScriptedExperimenter, run_episode, best_price_for)


# Deterministic turn order (the round protocol). Coordinator first (sets budget/directive);
# Marketer last (so it can read the Builder's "what I built" message this same round).
ROLES = ("coordinator", "builder", "pricer", "marketer")

# Partial-obs slicing: which roles' CURRENT-round messages each role may read. The Pricer
# additionally gets the Marketer's PRIOR-round targets from the log (the Marketer acts after
# the Pricer, so its current-round message isn't available yet — price reacts to last round).
ROLE_READS = {
    "coordinator": (),                  # goes first; decides from the firm summary
    "builder":     ("coordinator",),    # reads the phase directive
    "pricer":      ("coordinator",),    # + prior-round Marketer targets (from the log)
    "marketer":    ("coordinator", "builder"),  # reads "what's built" — the load-bearing link
}


# ----------------------------- blackboard -----------------------------

class Blackboard:
    """A per-round list of structured messages ``{role, text}`` (free-form text, capped).

    `msgs` holds the CURRENT round (cleared each commit); `log` is the full episode history
    (round-stamped) for the Coordinator's context window, the replay viewer, and SFT export.
    """
    MAX_LEN = 280

    def __init__(self):
        self.msgs = []     # current round's messages
        self.log = []      # full episode history (never cleared)

    def post(self, role, text, rnd=None):
        m = {"role": role, "text": str(text)[:self.MAX_LEN]}
        self.msgs.append(m)
        self.log.append({**m, "round": rnd})
        return m

    def read(self, roles):
        roles = set(roles)
        return [dict(m) for m in self.msgs if m["role"] in roles]

    def last_from(self, role):
        """Most recent message from `role` across the whole episode (prior rounds incl.)."""
        for m in reversed(self.log):
            if m["role"] == role:
                return m
        return None

    def clear(self):
        self.msgs = []


# ----------------------------- the multi-agent env -----------------------------

class MultiAgentFirmEnv:
    """Wraps a single `FirmEnv`. Each round the four roles `submit()` their partial action
    (and optionally post a message); `commit()` assembles them into ONE `FirmEnv` action
    dict and steps the underlying env. Role observations are sliced per ROLE_READS + the
    obs table in the build spec.

    Round-trip equivalence (tested): if the submitted slices assemble to action A and the
    Coordinator's budget doesn't bind, `commit()` reproduces `FirmEnv.step(A)` exactly.
    """

    def __init__(self, world: World):
        self.w = world
        self.cfg = world.cfg
        self.env = FirmEnv(world)
        self.reset()

    # -- lifecycle --
    def reset(self):
        self.env.reset()
        self.bb = Blackboard()
        self._stash = {}                 # role -> partial action dict (this round)
        self._budget = None              # Coordinator-set Marketer budget (None => full cash)
        self._last_per_campaign = []     # last round's diagnostics (Marketer's eyes)
        self._last_profit = 0.0
        self._last_churn = 0.0
        self.total_profit = 0.0
        return {r: self.role_obs(r) for r in ROLES}

    @property
    def round(self):
        return self.env.round

    @property
    def done(self):
        return self.env.done

    # -- observation slicing --
    def _firm_summary(self):
        return {
            "round": self.env.round,
            "horizon": self.cfg.horizon,
            "cash": round(self.env.cash, 2),
            "price": self.env.price,
            "built_features": sorted(self.env.built.keys()),
            "last_round_profit": round(self._last_profit, 2),
            "last_round_churn": round(self._last_churn, 4),
            "done": self.env.done,
        }

    def role_obs(self, role):
        """Return the sliced observation for `role`. Firm STATE (round/cash/price/built) is
        shared (it's the firm's own books); the partial part is WHO sees which campaign
        diagnostics and WHICH messages. Hidden market structure (segments/solves/wtp) is in
        NO role's obs — that's the no-leak invariant."""
        s = self._firm_summary()
        base = {"round": s["round"], "horizon": s["horizon"], "cash": s["cash"],
                "price": s["price"], "built_features": s["built_features"],
                "n_pains": self.cfg.n_pains, "n_channels": self.cfg.n_channels}
        budget = self._budget if self._budget is not None else s["cash"]

        if role == "coordinator":
            # firm summary + a window into the message history (it goes first, so the
            # current round's board is empty; it sees prior rounds' coordination instead).
            return {**base, "last_round_profit": s["last_round_profit"],
                    "last_round_churn": s["last_round_churn"],
                    "messages": [dict(m) for m in self.bb.log[-12:]]}

        if role == "builder":
            bq = round(sum(c.get("bounced_quality", 0.0) for c in self._last_per_campaign), 2)
            return {**base, "bounced_quality": bq,
                    "messages": self.bb.read(ROLE_READS["builder"])}

        if role == "pricer":
            bp = round(sum(c.get("bounced_price", 0.0) for c in self._last_per_campaign), 2)
            purch = round(sum(c.get("purchases", 0.0) for c in self._last_per_campaign), 2)
            return {**base, "bounced_price": bp, "recent_purchases": purch,
                    "last_round_churn": s["last_round_churn"],
                    "prior_marketer_msg": self.bb.last_from("marketer"),
                    "messages": self.bb.read(ROLE_READS["pricer"])}

        if role == "marketer":
            return {**base, "budget": budget, "per_campaign": self._last_per_campaign,
                    "messages": self.bb.read(ROLE_READS["marketer"])}

        raise ValueError(f"unknown role {role!r}")

    # -- the round protocol --
    def submit(self, role, action, message=None):
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        self._stash[role] = dict(action or {})
        if role == "coordinator":
            b = (action or {}).get("budget")
            if b is not None:
                self._budget = max(0.0, float(b))
        if message:
            self.bb.post(role, message, rnd=self.env.round)

    def assemble(self):
        """Combine the four stashed slices into one FirmEnv action dict (Marketer spend
        clamped to the Coordinator's budget). Pure — does not step the env."""
        bu = self._stash.get("builder", {})
        pr = self._stash.get("pricer", {})
        ma = self._stash.get("marketer", {})
        budget = self._budget if self._budget is not None else self.env.cash

        camps, spent = [], 0.0
        for c in (ma.get("campaigns") or []):
            spend = max(0.0, float(c.get("spend", 0.0)))
            spend = min(spend, max(0.0, budget - spent))   # budget contention
            spent += spend
            camps.append({"target": set(c.get("target", set())), "spend": spend,
                          "channel": int(c.get("channel", 0)),
                          "craft": float(c.get("craft", 1.0))})
        action = {"build": bu.get("build"),
                  "price": pr.get("price", self.env.price),
                  "campaigns": camps}
        if bu.get("quality") is not None:
            action["quality"] = bu["quality"]
        return action

    def commit(self):
        """Assemble + step the underlying env; distribute fresh sliced obs; clear the round."""
        action = self.assemble()
        churned_before = sum(self.env.churned.values()) if self.env.churned else 0.0

        obs, profit, done, _ = self.env.step({
            "build": action["build"], "price": action["price"],
            "campaigns": [{"target": set(c["target"]), "spend": c["spend"],
                           "channel": c["channel"], "craft": c["craft"]}
                          for c in action["campaigns"]],
        })
        # honor NL build quality (next-round, exactly like env.py / replay_profit)
        if action["build"] is not None and action.get("quality", 1.0) != 1.0:
            self.env.built[action["build"]] = action["quality"]

        churned_after = sum(self.env.churned.values()) if self.env.churned else 0.0
        self.total_profit = self.env.total_profit
        self._last_per_campaign = obs["per_campaign"]
        self._last_profit = profit
        self._last_churn = churned_after - churned_before

        self._stash = {}
        self._budget = None
        self.bb.clear()
        return {r: self.role_obs(r) for r in ROLES}, profit, done, action


# ----------------------------- episode runner -----------------------------

def run_team_episode(world, team, on_turn=None):
    """Drive a team through one episode. `team` exposes `reset()` and
    `act(role, obs, menv) -> (action, message)`. `on_turn(role, obs, action, menv)` (optional)
    fires per role-turn so a training harness can record (prompt, completion) role-turns.

    Returns (action_log, total_profit). action_log is replay-compatible (sim.replay_profit).
    """
    menv = MultiAgentFirmEnv(world)
    team.reset()
    menv.reset()
    action_log = []
    while not menv.done:
        for role in ROLES:
            obs = menv.role_obs(role)            # fresh: picks up earlier roles' messages
            action, message = team.act(role, obs, menv)
            if on_turn is not None:
                on_turn(role, obs, action, menv)
            menv.submit(role, action, message=message)
        _obs, _profit, _done, committed = menv.commit()
        action_log.append({
            "build": committed["build"],
            "price": committed["price"],
            "quality": committed.get("quality", 1.0),
            "campaigns": [{"target": sorted(c["target"]), "spend": c["spend"],
                           "channel": c["channel"], "craft": c["craft"]}
                          for c in committed["campaigns"]],
        })
    return action_log, menv.total_profit


# ----------------------------- scripted role policies -----------------------------
# These reproduce ScriptedExperimenter's disciplined discovery, SPLIT across the four roles
# and communicating via the blackboard. A `read_messages` flag toggles whether a role
# consumes the board: ScriptedTeam=True (coordinated), NaiveTeam=False (isolated). The gap
# between them is exactly the coordination tax.

_BUILD_TARGET = 6        # solve the top-6 pains (matches OraclePolicy / ScriptedExperimenter)
_PROBE_SPEND = 10.0
_TEST_SPEND = 60.0


def _parse_targets(text):
    """Pull a pain-id list out of a 'TARGETS: [0, 2, 5]' style message."""
    if not text:
        return []
    nums = re.findall(r"\d+", text)
    return [int(n) for n in nums]


class _RolePolicies:
    """The four scripted role brains with PRIVATE per-role memory. Cross-role information
    flows only through `obs["messages"]` (when `read_messages`), never shared Python state —
    so toggling `read_messages` faithfully removes the coordination channel."""

    def __init__(self, world, seed=0, read_messages=True):
        self.w = world
        self.cfg = world.cfg
        self.seed = seed
        self.read = read_messages

    def reset(self):
        # marketer memory (the only role that sees campaign diagnostics)
        self.pain_demand = {}        # pain -> audience (demand proxy)
        self.best_channel = {}       # pain -> channel with most observed tries
        self._best_ch_tries = {}
        self.solved = {}             # pain -> feature (discovered from purchases)
        self._prev_built = None      # feature the Builder told us about last round
        # builder memory
        self.tried = set()

    # -- coordinator: budget + phase directive (decides from firm summary; goes first) --
    def coordinator(self, obs):
        r, H = obs["round"], obs["horizon"]
        built = len(obs["built_features"])
        if r == 0:
            phase = "probe"
            budget = obs["n_pains"] * obs["n_channels"] * _PROBE_SPEND * 1.5
        elif built < _BUILD_TARGET and (H - r) > 2:
            phase = "discover"
            budget = obs["cash"]
        else:
            phase = "exploit"
            budget = obs["cash"]
        msg = f"PHASE: {phase} | BUDGET: {budget:.0f}"
        return {"budget": budget}, msg

    def _phase_from(self, obs, default):
        """Read the Coordinator's phase from the board (or infer from firm state if isolated)."""
        if self.read:
            for m in obs.get("messages", []):
                if m["role"] == "coordinator" and "PHASE:" in m["text"]:
                    return m["text"].split("PHASE:")[1].split("|")[0].strip()
        # isolated fallback: infer the phase from observable firm state (round + build count)
        r, H, built = obs["round"], obs["horizon"], len(obs["built_features"])
        if r == 0:
            return "probe"
        if built < _BUILD_TARGET and (H - r) > 2:
            return "discover"
        return "exploit"

    # -- builder: build the next untried feature during discovery; announce it --
    def builder(self, obs):
        phase = self._phase_from(obs, "discover")
        if phase != "discover":
            return {"build": None}, None
        built = set(obs["built_features"])
        candidates = [f for f in range(self.cfg.n_features)
                      if f not in built and f not in self.tried]
        if not candidates:
            return {"build": None}, None
        f = candidates[0]
        self.tried.add(f)
        return {"build": f}, f"BUILT: feature {f}"

    # -- pricer: best price for the (prior-round) target pains; else a safe default --
    def pricer(self, obs):
        built_map = {f: 1.0 for f in obs["built_features"]}
        targets = []
        if self.read and obs.get("prior_marketer_msg"):
            targets = _parse_targets(obs["prior_marketer_msg"]["text"])
        if built_map and targets:
            price = best_price_for(self.w, built_map, set(targets))
        else:
            price = 50.0
        return {"price": price}, None

    # -- marketer: the eyes. probe -> test new builds -> exploit solved pains --
    def marketer(self, obs):
        cfg = self.cfg
        # 1) ingest last round's diagnostics (attribute purchases to the prev-built feature)
        for c in obs.get("per_campaign", []):
            tgt = c.get("target", [])
            if len(tgt) == 1:
                p = tgt[0]
                if obs["round"] <= 1:
                    self.pain_demand[p] = c.get("audience", 0)
                    ch = c.get("channel", 0)
                    if c.get("tries", 0.0) > self._best_ch_tries.get(p, -1.0):
                        self._best_ch_tries[p] = c.get("tries", 0.0)
                        self.best_channel[p] = ch
                if c.get("purchases", 0.0) > 0.5 and self._prev_built is not None:
                    self.solved[p] = self._prev_built

        # 2) read what the Builder built THIS round (the load-bearing message)
        built_now = None
        if self.read:
            for m in obs.get("messages", []):
                if m["role"] == "builder" and "BUILT:" in m["text"]:
                    nums = _parse_targets(m["text"])
                    if nums:
                        built_now = nums[-1]

        def ch(p):
            return self.best_channel.get(p, 0)

        phase = self._phase_from(obs, "discover")
        top_pains = sorted(self.pain_demand, key=self.pain_demand.get, reverse=True)
        unsolved_top = [p for p in top_pains if p not in self.solved]

        # 3) act by phase
        if phase == "probe" or (obs["round"] == 0):
            camps = [{"target": {p}, "spend": _PROBE_SPEND, "channel": chn}
                     for p in range(cfg.n_pains) for chn in range(cfg.n_channels)]
            self._prev_built = None
            return {"campaigns": camps}, f"TARGETS: {list(range(cfg.n_pains))}"

        if phase == "discover" and built_now is not None:
            # test the freshly-built feature against the top unsolved pains (so purchases
            # attribute cleanly next round) and keep earning on already-solved pains
            test = unsolved_top[:6] or list(range(cfg.n_pains))[:6]
            camps = [{"target": {p}, "spend": _TEST_SPEND, "channel": ch(p)} for p in test]
            if self.solved:
                ts = max(self.solved, key=lambda x: self.pain_demand.get(x, 0))
                camps.append({"target": set(self.solved.keys()),
                              "spend": max(0.0, obs["budget"] * 0.4), "channel": ch(ts)})
            self._prev_built = built_now
            return {"campaigns": camps}, f"TARGETS: {sorted(set(test) | set(self.solved))}"

        # exploit (or discover with nothing new built): pour budget into solved pains
        self._prev_built = built_now
        if self.solved:
            target = set(self.solved.keys())
            ts = max(self.solved, key=lambda x: self.pain_demand.get(x, 0))
            return ({"campaigns": [{"target": target, "spend": max(0.0, obs["budget"]),
                                    "channel": ch(ts)}]},
                    f"TARGETS: {sorted(target)}")
        # nothing discovered yet — fall back to the best-demand pains broadly
        target = set(top_pains[:3]) if top_pains else {0, 1, 2}
        c0 = top_pains[0] if top_pains else 0
        return ({"campaigns": [{"target": target,
                                "spend": min(500.0, obs["budget"]), "channel": ch(c0)}]},
                f"TARGETS: {sorted(target)}")


class _Team:
    """Dispatches role-turns to a `_RolePolicies` brain. `READS_MESSAGES` distinguishes the
    coordinated team from the isolated one."""
    READS_MESSAGES = True
    name = "team"

    def __init__(self, world, seed=0):
        self.brain = _RolePolicies(world, seed, read_messages=self.READS_MESSAGES)

    def reset(self):
        self.brain.reset()

    def act(self, role, obs, menv):
        return getattr(self.brain, role)(obs)


class ScriptedTeam(_Team):
    """Coordinated team: roles READ the blackboard (Builder tells Marketer what's built,
    Marketer tells Pricer what to price for). Proves the protocol is coordinatable."""
    READS_MESSAGES = True
    name = "scripted-team"


class NaiveTeam(_Team):
    """Isolated team: identical role policies but roles IGNORE the blackboard. The Marketer
    never learns what's built (can't attribute solves -> can't exploit); the Pricer never
    learns the targets (flat price). The gap vs ScriptedTeam is the coordination tax."""
    READS_MESSAGES = False
    name = "naive-team"


# ----------------------------- coordination tax + gate -----------------------------

def coordination_tax(world, team_factory=None, seed=0):
    """`tax = oracle_profit - team_profit` (and `team_disc_eff = team/oracle`). The oracle is
    the single-agent FULL-INFO OraclePolicy (no partial-obs barrier) — the ceiling a perfectly
    coordinated team approaches. Returns a dict."""
    oracle = run_episode(world, OraclePolicy(world))
    team = (team_factory or (lambda w, s: ScriptedTeam(w, s)))(world, seed)
    _log, team_profit = run_team_episode(world, team)
    disc = max(0.0, min(1.0, team_profit / oracle)) if oracle > 0 else 0.0
    return {"oracle": oracle, "team_profit": team_profit,
            "tax": oracle - team_profit, "team_disc_eff": disc}


def coordination_gate(seeds=(1, 2, 3, 4, 5), cfg=None):
    """The Phase-D analog of `ablation_gate`. Reports, per the held-out seeds, the disc.eff
    and coordination tax of the naive team, the scripted team, and (for reference) the
    single-agent ScriptedExperimenter, against the oracle ceiling.

    Gate PASS iff  naive_team < scripted_team <= oracle  AND  scripted tax < naive tax
    (the messages must buy coordination). WARN if a team beats the oracle (reference too
    weak); FAIL if the scripted team can't beat the naive one (a role lacks the obs/message
    it needs — fix the slice, don't hand-wave)."""
    cfg = cfg or Config.phase_a()
    rows = []
    hdr = (f"{'config':>14} | {'naive_team':>10} | {'scripted_team':>13} | "
           f"{'1agent_scr':>10} | {'oracle':>10} | {'gate':>5}")
    print(hdr)
    print("-" * len(hdr))

    agg = {"naive": [], "scripted": [], "single": [], "oracle": []}
    for s in seeds:
        world = generate_world(s, cfg)
        oracle = run_episode(world, OraclePolicy(world))
        _, naive_p = run_team_episode(world, NaiveTeam(world, s))
        _, scr_p = run_team_episode(world, ScriptedTeam(world, s))
        single = run_episode(world, ScriptedExperimenter(world, s))
        agg["naive"].append(naive_p); agg["scripted"].append(scr_p)
        agg["single"].append(single); agg["oracle"].append(oracle)

    nv = sum(agg["naive"]) / len(seeds)
    sc = sum(agg["scripted"]) / len(seeds)
    sg = sum(agg["single"]) / len(seeds)
    orc = sum(agg["oracle"]) / len(seeds)
    if sc > orc:
        gate = "WARN"
    elif nv < sc:
        gate = "PASS"
    else:
        gate = "FAIL"
    print(f"{'full':>14} | {nv:>10.0f} | {sc:>13.0f} | {sg:>10.0f} | {orc:>10.0f} | {gate:>5}")
    print("-" * len(hdr))
    print(f"  disc.eff   naive_team={nv/orc:.3f}  scripted_team={sc/orc:.3f}  "
          f"single_scripted={sg/orc:.3f}  (oracle=1.000)")
    print(f"  coord tax  naive={orc - nv:.0f} ({1 - nv/orc:.1%})  "
          f"scripted={orc - sc:.0f} ({1 - sc/orc:.1%})  "
          f"-> messages buy {((orc - nv) - (orc - sc)) / orc:.1%} of the oracle")
    rows.append({"naive": nv, "scripted": sc, "single": sg, "oracle": orc, "gate": gate})
    return rows


if __name__ == "__main__":
    print("FirmBench — Phase D coordination gate (full Phase A market)\n")
    coordination_gate()
