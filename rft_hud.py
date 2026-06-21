"""
FirmBench — HUD-native on-policy RL (GRPO / policy gradient via ``hud.train``).

Where ``rft.py`` does rejection-sampling SFT on an *external* provider (Fireworks),
this trains the policy ON-POLICY through HUD's own training service (Tinker-backed).
The whole loop lives inside HUD:

  1. FORK     — `hud models fork <base>` makes a team-owned *trainable* model whose
                weights live behind a gateway slug (one-time, done outside this script).
  2. ROLLOUT  — for each training world, sample `group_size` episodes with the current
                model through the HUD gateway (`hud.eval.rollout`). Each returned
                `hud.Run` carries token-level samples (ids + sampling logprobs) AND the
                environment's reward — the env's `disc_eff` = profit / oracle in [0,1].
  3. UPDATE   — one grouped policy-gradient step on the batch: `forward_backward`
                (advantages normalized within each world's group — GRPO) + `optim_step`
                (checkpoint + promote). The gateway now serves the new weights.
  4. EVALUATE — re-roll the held-out seeds greedily; append mean reward to the curve.
                Because the gateway promoted the checkpoint, step k+1's rollouts are
                automatically on-policy. No dataset files, no SFT — true RL.

Contrast with rft.py:
  rft.py     : rollout -> reject -> SFT(cross_entropy) on winners -> redeploy   (off-policy, imitation)
  rft_hud.py : rollout -> grouped policy gradient -> checkpoint promote          (on-policy, RL)

Two modes (mirrors rft.py):
  python3 rft_hud.py --selftest
      Offline, no HUD/network/keys. A mock gateway+trainer whose policy `skill`
      rises per step drives the REAL loop orchestration (per-seed grouping, GRPO
      group divisibility, curve recording), graded with the env's actual disc_eff
      reward via the local sim. Proves the harness bends the curve.

  python3 rft_hud.py --run --model accounts/<team>/models/<forked-trainable>
      The real thing. First:  hud login   then   hud models fork <base-model>.
      Rolls out through the HUD gateway and trains via hud.train.TrainingClient.

Flags: --steps, --train-seeds, --eval-seeds, --group-size, --lr, --loss,
       --temperature, --model, --out.
"""

import argparse
import asyncio
import json
import os
from statistics import mean

from sim import (Config, generate_world, run_episode,
                 OraclePolicy, ScriptedExperimenter, NaivePolicy)

# Match the DEPLOYED env: env.py grades against Config.phase_a() worlds, so the
# local references and the offline selftest must generate the same market.
CFG = Config.phase_a()

DEFAULT_MODEL = os.environ.get("HUD_MODEL", "")
# HUD's built-in on-policy policy-gradient loss (rollout-logprob importance ratio).
# Others (provider-dependent): "ppo", "cispo", "dro". Discover the live set with
# `await TrainingClient(model).available_losses()`.
DEFAULT_LOSS = "importance_sampling"


# ----------------------------- reward (matches env.py) -----------------------------

def disc_eff(world, profit):
    """The env's reward: discovery efficiency = profit / oracle_profit, clipped to
    [0,1]. Identical to env.py:_grade_episode so offline rewards match real rollouts."""
    oracle = run_episode(world, OraclePolicy(world))
    return max(0.0, min(1.0, profit / oracle)) if oracle > 0 else 0.0


_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


async def _with_retry(coro_fn, *, what, tries=6, base_delay=10.0):
    """Retry a training HTTP call on transient server errors (5xx / 429). Beta infra
    throws intermittent 503s; without this a multi-hour run dies on a momentary blip.
    A transient status is a proxy-level rejection (request not processed), so a retry
    does not double-count gradients. Re-raises non-transient errors immediately."""
    from hud.utils.exceptions import HudRequestError
    for attempt in range(1, tries + 1):
        try:
            return await coro_fn()
        except HudRequestError as e:
            transient = getattr(e, "status_code", None) in _TRANSIENT_STATUS
            if not transient or attempt == tries:
                raise
            delay = base_delay * attempt
            print(f"    [retry {attempt}/{tries - 1}] {what}: {e.status_code} transient; "
                  f"waiting {delay:.0f}s")
            await asyncio.sleep(delay)


def _check_groups(n, group_size):
    """Fail before a training step if the batch can't form full GRPO groups: an
    incomplete final group gets a skewed advantage baseline. Cheap, no round-trip.
    (The HUD service re-checks server-side; this catches it locally first.)"""
    if group_size is not None and n % group_size != 0:
        raise ValueError(
            f"{n} rollouts do not divide evenly into groups of {group_size}; "
            "GRPO normalizes advantages within each group, so every group must be full")


def local_ref(policy_factory, seeds):
    """Deterministic local reward for a reference policy (oracle/scripted/naive),
    graded with the exact env reward. Gives the curve its context lines."""
    return mean(disc_eff(w := generate_world(s, CFG), run_episode(w, policy_factory(w, s)))
                for s in seeds)


# ============================== backends ==============================
# A backend abstracts "roll out the env with the current policy" + "take one RL
# step that improves it". The real one talks to HUD; the mock one runs the sim
# locally so the loop logic can be exercised with zero credentials.


class MockBackend:
    """Offline stand-in for the HUD gateway + training service.

    Holds a single scalar `skill` in [0,1] standing in for the policy's weights.
    `rollout_group` samples `rft.MockModel(skill)` episodes (which play the
    disciplined expert with prob `skill`, else random) and grades them with the
    env's real disc_eff reward. `train_step` validates GRPO grouping and raises
    `skill` toward 1.0 in proportion to the batch reward — a faithful stand-in for
    an on-policy policy-gradient update (good batches improve the policy faster).
    """

    def __init__(self, skill0=0.15):
        self.skill = skill0

    def _episode_reward(self, seed, k, temperature):
        from rft import MockModel  # offline-safe (README: rft.py --selftest needs no keys)
        world = generate_world(seed, CFG)
        agent = MockModel(world, skill=self.skill, seed=seed * 1000 + k, temperature=temperature)
        return disc_eff(world, run_episode(world, agent))

    async def rollout_group(self, seed, n, temperature):
        # contiguous block of n rollouts on the SAME world == one GRPO group
        return [_MockRun(self._episode_reward(seed, k, temperature)) for k in range(n)]

    async def eval(self, seeds, temperature=0.0):
        return mean(self._episode_reward(s, 0, temperature) for s in seeds)

    async def train_step(self, runs, lr, loss_fn, group_size):
        _check_groups(len(runs), group_size)
        mr = mean(r.reward for r in runs)
        # policy-gradient stand-in: improvement scales with the reward signal and
        # the remaining headroom, so the curve bends and saturates near the ceiling.
        gain = (0.18 + 0.55 * mr) * (1.0 - self.skill)
        self.skill = min(0.97, self.skill + gain)
        return {"mean_reward": mr, "skill": round(self.skill, 3)}


class _MockRun:
    """Minimal Run stand-in for the offline path (only `.reward` is read)."""
    __slots__ = ("reward",)

    def __init__(self, reward):
        self.reward = reward


class HudBackend:
    """Real backend: rolls out through the HUD gateway and trains via hud.train.

    `model` is a *trainable* gateway slug (create one with `hud models fork <base>`).
    Training advances the weights behind that slug in place; each optim_step promotes
    a new checkpoint so the next rollout is on-policy.
    """

    def __init__(self, model, env_path="env.py", max_steps=80, max_tokens=4096):
        from hud.eval import LocalRuntime          # local: serve env.py in a child proc
        from hud.agents import create_agent
        from hud.train import TrainingClient

        self.model = model
        self.max_steps = max_steps                 # ~16 rounds x a few tool calls each
        self.max_tokens = max_tokens               # per-turn token budget (reasoning + tool call)
        self.runtime = LocalRuntime(env_path)
        self._create_agent = create_agent
        self.client = TrainingClient(model)

    def _agent(self, temperature):
        # openai_compatible (Tinker) config ignores a top-level `temperature`/`max_tokens`
        # — they live in completion_kwargs — and defaults max_steps=10 (too few for a
        # full episode). Three settings are load-bearing for on-policy RL:
        #   extra_body.return_token_ids=True  -> the gateway returns token ids + per-token
        #       logprobs so each AgentStep carries a trainable Sample (without it the run
        #       has NO token samples and forward_backward gets nothing to learn from).
        #   max_tokens large enough that a reasoning model finishes <think> AND emits the
        #       tool call in one turn (else the truncated turn has no tool call and the
        #       agent loop ends early at round 0 -> reward ~0).
        #   max_steps covers the whole episode's tool calls.
        return self._create_agent(
            self.model, max_steps=self.max_steps,
            completion_kwargs={
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "extra_body": {"return_token_ids": True},
            })

    def _tasks(self, seeds):
        from env import market_discovery
        from tasks import SYSTEM_PROMPT
        return [market_discovery(prompt=SYSTEM_PROMPT, seed=s) for s in seeds]

    async def rollout_group(self, seed, n, temperature):
        from hud.eval import rollout
        agent = self._agent(temperature)
        tasks = self._tasks([seed] * n)            # n rollouts of the same world
        return list(await asyncio.gather(
            *(rollout(t, agent, runtime=self.runtime) for t in tasks)))

    async def eval(self, seeds, temperature=0.0):
        from hud.eval import rollout
        agent = self._agent(temperature)
        runs = await asyncio.gather(
            *(rollout(t, agent, runtime=self.runtime) for t in self._tasks(seeds)))
        return mean(r.reward for r in runs)

    async def train_step(self, runs, lr, loss_fn, group_size):
        _check_groups(len(runs), group_size)
        # Train from trace_id REFERENCES, not inline token blobs. With KV-cache
        # continuation each turn's Sample carries the full prompt token ids, so an
        # inline batch is O(turns^2) per episode -> hundreds of MB -> HTTP 413. The
        # service resolves the recorded tokens + reward from the trace_id (~KB body).
        # Micro-batch ONE whole GRPO group per call (advantages normalize within the
        # group) and accumulate gradients; a single optim_step applies the sum and
        # checkpoints+promotes (gateway then serves the new weights).
        inputs = [r.trace_id for r in runs]
        if any(t is None for t in inputs):
            raise ValueError("a rollout produced no trace_id; cannot train from it")
        total_datums = 0
        for i in range(0, len(inputs), group_size):
            chunk = inputs[i:i + group_size]
            res = await _with_retry(
                lambda c=chunk: self.client.forward_backward(
                    c, loss_fn=loss_fn, group_size=group_size),
                what="forward_backward")
            total_datums += res.num_datums
        if total_datums == 0:
            raise RuntimeError(
                "forward_backward resolved 0 training datums from trace_ids — the "
                "gateway trace has no token-level data to train on")
        step = await _with_retry(
            lambda: self.client.optim_step(learning_rate=lr), what="optim_step")
        head = await self.client.head()
        return {"mean_reward": mean(r.reward for r in runs),
                "step": step.step, "checkpoint": step.checkpoint_id,
                "num_datums": total_datums,
                "head_mean_reward": getattr(head, "mean_reward", None)}


# ============================== the RL loop ==============================

async def rl_loop(backend, *, train_seeds, eval_seeds, steps, group_size, lr,
                  loss_fn, temperature, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # references (local, deterministic): oracle = ceiling (~1.0), scripted = strong
    # heuristic, naive = floor. disc_eff is already normalized so these read as fractions.
    oracle = local_ref(lambda w, s: OraclePolicy(w), eval_seeds)
    scripted = local_ref(lambda w, s: ScriptedExperimenter(w, s), eval_seeds)
    naive = local_ref(lambda w, s: NaivePolicy(w, s), eval_seeds)
    print(f"\nReference (disc_eff)  oracle={oracle:.3f}  scripted={scripted:.3f}  "
          f"naive={naive:.3f}\n")

    curve = []
    base = await backend.eval(eval_seeds)
    curve.append({"step": 0, "model": "base", "eval": round(base, 4)})
    print(f"[step 0] BASE  eval disc_eff={base:.3f}")

    for step in range(1, steps + 1):
        runs = []
        for s in train_seeds:                       # per-seed groups, kept contiguous
            runs += await backend.rollout_group(s, group_size, temperature)
        metrics = await backend.train_step(runs, lr, loss_fn, group_size)
        ev = await backend.eval(eval_seeds)
        point = {"step": step, "eval": round(ev, 4),
                 "train_mean_reward": round(metrics["mean_reward"], 4),
                 "n_rollouts": len(runs)}
        for k in ("skill", "checkpoint", "num_datums", "head_mean_reward"):
            if k in metrics:
                point[k] = metrics[k]
        curve.append(point)
        extra = f" skill={metrics['skill']}" if "skill" in metrics else (
            f" ckpt={metrics.get('checkpoint')}" if metrics.get("checkpoint") else "")
        print(f"[step {step}] train_reward={metrics['mean_reward']:.3f}  "
              f"eval disc_eff={ev:.3f}{extra}")

    # --- the money shot: does the curve bend toward the oracle? ---
    print("\n" + "=" * 60)
    print("HUD RL CURVE (mean held-out disc_eff by step)")
    print("=" * 60)
    span = max(1e-6, oracle)
    for c in curve:
        r = c["eval"]
        print(f"  step {c['step']:>2}  {r:>6.3f}  {'#' * int(40 * max(0.0, r) / span)}")
    print(f"  {'oracle':>7}  {oracle:>6.3f}  (ceiling)")
    base_r, final_r = curve[0]["eval"], curve[-1]["eval"]
    delta = final_r - base_r
    print("-" * 60)
    print(f"BASE -> RL: {base_r:.3f} -> {final_r:.3f}   "
          f"({'+' if delta >= 0 else ''}{delta:.3f}, "
          f"{(delta / base_r * 100) if base_r else float('inf'):+.0f}%)")
    print("=" * 60)

    with open(os.path.join(out_dir, "curve.json"), "w") as f:
        json.dump({"curve": curve,
                   "references": {"oracle": oracle, "scripted": scripted, "naive": naive},
                   "config": {"steps": steps, "group_size": group_size, "lr": lr,
                              "loss_fn": loss_fn, "train_seeds": train_seeds,
                              "eval_seeds": eval_seeds}},
                  f, indent=2)
    return curve


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="FirmBench HUD-native on-policy RL (GRPO)")
    ap.add_argument("--selftest", action="store_true",
                    help="offline dry run with a mock gateway+trainer (no HUD, no keys)")
    ap.add_argument("--run", action="store_true",
                    help="real run: rollouts + training through HUD (needs `hud login`)")
    ap.add_argument("--steps", type=int, default=3, help="RL optimizer steps")
    ap.add_argument("--group-size", type=int, default=8,
                    help="rollouts per world (the GRPO group size)")
    ap.add_argument("--train-seeds", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loss", default=DEFAULT_LOSS,
                    help="built-in loss: importance_sampling | ppo | cispo | dro")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="rollout sampling temperature (eval is greedy at 0.0)")
    ap.add_argument("--max-steps", type=int, default=80,
                    help="agent tool-call budget per episode (~16 rounds x a few calls)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="per-turn token budget (reasoning models need room for <think> + tool call)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="forked trainable gateway slug (required for --run)")
    ap.add_argument("--out", default="rft_hud_out")
    args = ap.parse_args()

    if not args.selftest and not args.run:
        print("Choose a mode: --selftest (offline) or --run (real HUD). "
              "See `python3 rft_hud.py -h`.")
        return

    train_seeds = list(range(1, 1 + args.train_seeds))
    eval_seeds = list(range(100, 100 + args.eval_seeds))   # disjoint held-out seeds

    print("=" * 60)
    print("FirmBench — HUD-native on-policy RL")
    print(f"mode={'selftest' if args.selftest else 'run'}  "
          f"model={args.model or '(mock)'}  loss={args.loss}")
    print(f"steps={args.steps}  group_size={args.group_size}  "
          f"train_worlds={len(train_seeds)}  eval_worlds={len(eval_seeds)}")
    print("=" * 60)

    if args.run:
        from hud.settings import settings
        if not settings.api_key:
            print("\nERROR: --run needs a HUD API key. Run `hud login` "
                  "(or set HUD_API_KEY).")
            return
        if not args.model:
            print("\nERROR: --run needs a trainable model. Create one with "
                  "`hud models fork <base-model>` and pass --model <slug>.")
            return
        backend = HudBackend(args.model, max_steps=args.max_steps,
                             max_tokens=args.max_tokens)
    else:
        backend = MockBackend()

    asyncio.run(rl_loop(
        backend,
        train_seeds=train_seeds, eval_seeds=eval_seeds,
        steps=args.steps, group_size=args.group_size, lr=args.lr,
        loss_fn=args.loss, temperature=args.temperature, out_dir=args.out))


if __name__ == "__main__":
    main()
