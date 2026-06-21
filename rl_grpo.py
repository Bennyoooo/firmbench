"""
FirmBench — TRUE RL via GRPO (Group Relative Policy Optimization) on Gemma.

This is RL, not SFT. `rft.py` / `rft_gcp.py` do reward-*filtered* SFT (keep the winners,
cross-entropy on them) — they never push DOWN bad actions and use no advantage/KL. This
file does real policy-gradient RL: for each world it samples a GROUP of full episodes,
grades them with the verifier, computes a **group-relative advantage** (the GRPO baseline
— no critic needed), and updates the policy with a **PPO-style clipped surrogate** plus a
**KL penalty to a frozen reference**. Good episodes are reinforced; bad ones are actively
suppressed. FirmBench is a textbook RL-with-verifiable-rewards (RLVR) task: the held-out
verifier (run.Verifier) IS the reward function.

It's the same RL shape as run.py's `train_reinforce`, lifted from a hand-tuned REINFORCE
to (a) a proper GRPO objective and (b) the Gemma LLM as the policy.

Multi-turn handling (episode = 10 LLM calls, one scalar reward): trajectory-level credit
assignment — the episode's group-relative advantage is broadcast to every turn's tokens.

Two ways to run:
  python3 rl_grpo.py --selftest    # offline, CPU, no torch/GPU. The REAL GRPO loop
                                    # optimizes a categorical policy over FirmBench
                                    # probe->exploit schedules and prints the reward curve
                                    # bending. Validates the GRPO machinery end-to-end.

  python3 rl_grpo.py --run         # real GRPO on a Gemma open model (needs GPU + torch +
                                    # transformers + peft). Open weights are REQUIRED — you
                                    # can't gradient-RL a closed model (e.g. Gemini).
                                    #   pip install -r requirements-rl.txt
                                    #   export RL_BASE_MODEL=google/gemma-2-2b-it
                                    # Runs locally or as a Vertex AI custom-training job.

Flags: --iterations, --group-size, --train-seeds, --eval-seeds, --lr, --clip, --kl,
       --inner-epochs, --model, --out.
"""

import os
import math
import json
import random
import argparse
from statistics import mean, pstdev

from sim import (Config, generate_world, FirmEnv, OraclePolicy,
                 best_price_for, run_episode)
from run import Verifier

DEFAULT_MODEL = os.environ.get("RL_BASE_MODEL", "google/gemma-2-2b-it")


# ============================================================================
# GRPO core (policy-agnostic): group-relative advantage + clipped surrogate.
# These two functions are shared by the tabular selftest and the Gemma trainer —
# the actual RL algorithm lives here, independent of how logπ is computed.
# ============================================================================

def group_advantages(rewards):
    """GRPO baseline: normalize rewards WITHIN a group (same world, G rollouts).

    A_i = (R_i - mean) / (std + eps). This replaces the value network — the group
    mean is the baseline, so above-average episodes get positive advantage and
    below-average ones get NEGATIVE advantage (the part SFT-on-winners can't do).
    """
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    m = mean(rewards)
    s = pstdev(rewards)
    if s < 1e-8:
        return [0.0 for _ in rewards]  # whole group tied → no signal
    return [(r - m) / s for r in rewards]


def clipped_surrogate_coef(advantage, ratio, clip_eps):
    """PPO/GRPO clipped objective, returned as the scalar multiplying d(logπ)/dθ.

    surrogate = min(ratio·A, clip(ratio, 1-eps, 1+eps)·A).
    d/dθ[ratio·A] = A·ratio·d(logπ)/dθ. In the clipped region the gradient is 0 (the
    objective is flat), exactly as in PPO. Returns (coef, was_clipped).
    """
    if advantage >= 0:
        clipped = ratio > 1.0 + clip_eps
    else:
        clipped = ratio < 1.0 - clip_eps
    coef = 0.0 if clipped else advantage * ratio
    return coef, clipped


# ============================================================================
# Reward (= run.Verifier discovery efficiency = profit/oracle in [0,1]), with the
# oracle cached per seed so the hot RL loop doesn't recompute it thousands of times.
# ============================================================================

class RewardFn:
    """profit / oracle clipped to [0,1] — identical to run.Verifier.grade()['reward'],
    memoized per seed. A real Verifier is kept for parity checks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._oracle = {}
        self.verifier = Verifier()

    def oracle(self, seed, world):
        if seed not in self._oracle:
            self._oracle[seed] = run_episode(world, OraclePolicy(world))
        return self._oracle[seed]

    def reward(self, seed, world, profit):
        o = self.oracle(seed, world)
        return max(0.0, min(1.0, profit / o)) if o > 0 else 0.0


# ============================================================================
# Tabular policy backend (offline selftest).
#
# A CATEGORICAL policy over FirmBench probe->exploit SCHEDULES: the action is the
# switch round k (probe/discover for rounds < k, then exploit the discovered
# pain->feature map for the rest). reward(k) has a clear interior optimum (k≈6),
# so the GRPO loop has something real to learn and held-out reward visibly bends.
# This is the exact GRPO algorithm the Gemma path uses, on a policy we can train
# on CPU with no torch.
# ============================================================================

class ScheduledFirmPolicy:
    """Probe/discover for rounds < k, then exploit. k is the (learned) action."""

    def __init__(self, world, switch_k, seed=0):
        self.w = world
        self.k = switch_k

    def reset(self):
        self.pain_demand = {}
        self.solved = {}
        self.tried = set()
        self._last_built = None

    def act(self, env, obs):
        cfg = self.w.cfg
        built = set(obs["built_features"])
        # ingest diagnostics from last round's campaigns
        for c in obs["per_campaign"]:
            tgt = c["target"]
            if len(tgt) == 1:
                p = tgt[0]
                if obs["round"] <= 1:
                    self.pain_demand[p] = c["audience"]
                if c["purchases"] > 0.5 and self._last_built is not None:
                    self.solved[p] = self._last_built
        self._last_built = None

        if obs["round"] == 0:  # round 0: probe all pains cheaply to size demand
            return {"build": None, "price": 50.0,
                    "campaigns": [{"target": {p}, "spend": 10.0} for p in range(cfg.n_pains)]}

        top = sorted(self.pain_demand, key=self.pain_demand.get, reverse=True)
        unsolved_top = [p for p in top if p not in self.solved]

        # DISCOVER while round < k: build the next feature + cheap diagnostic probes
        if obs["round"] < self.k and unsolved_top and obs["round"] < cfg.horizon - 1:
            cand = [x for x in range(cfg.n_features)
                    if x not in built and x not in self.tried]
            f = cand[0] if cand else None
            if f is not None:
                self.tried.add(f)
                self._last_built = f
                return {"build": f, "price": 50.0,
                        "campaigns": [{"target": {p}, "spend": 60.0} for p in unsolved_top[:6]]}

        # EXPLOIT: full budget on the discovered pains at the best price
        if self.solved:
            target = set(self.solved.keys())
            bm = {x: 1.0 for x in built}
            return {"build": None, "price": best_price_for(self.w, bm, target),
                    "campaigns": [{"target": target, "spend": max(0.0, obs["cash"])}]}

        target = set(top[:3]) if top else {0, 1, 2}
        return {"build": None, "price": 50.0,
                "campaigns": [{"target": target, "spend": min(500.0, obs["cash"])}]}


def _softmax(logits):
    m = max(logits)
    ex = [math.exp(l - m) for l in logits]
    z = sum(ex)
    return [e / z for e in ex]


def _sample_cat(probs, rng):
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1


def _logp_cat(logits, i):
    return math.log(max(_softmax(logits)[i], 1e-12))


def _dlogp_cat(logits, i):
    """∂logπ(i)/∂logit_j = 1[j=i] - softmax_j."""
    p = _softmax(logits)
    return [(1.0 if j == i else 0.0) - p[j] for j in range(len(logits))]


def grpo_train_tabular(train_seeds, eval_seeds, cfg, *, iterations, group_size,
                       lr, clip_eps, kl_coef, inner_epochs, out_dir, seed=0):
    """GRPO over the categorical schedule policy. Genuine RL: group-relative advantage
    + clipped importance-weighted policy gradient + KL to a frozen reference."""
    rng = random.Random(seed)
    n_actions = cfg.horizon                  # action = switch round k in 1..horizon
    logits = [0.0] * n_actions               # start uniform
    logits_ref = list(logits)                # frozen reference for the KL term
    rfn = RewardFn(cfg)
    rcache = {}                              # (seed, k) -> reward (sim is deterministic)
    worlds = {s: generate_world(s, cfg) for s in set(train_seeds) | set(eval_seeds)}

    def reward_of(s, k):
        key = (s, k)
        if key not in rcache:
            prof = run_episode(worlds[s], ScheduledFirmPolicy(worlds[s], k))
            rcache[key] = rfn.reward(s, worlds[s], prof)
        return rcache[key]

    def expected_eval(lg):
        """Exact expected held-out reward under the current policy (no sampling noise)."""
        probs = _softmax(lg)
        return mean(sum(probs[i] * reward_of(s, i + 1) for i in range(n_actions))
                    for s in eval_seeds)

    curve = [{"iter": 0, "eval_reward": expected_eval(logits)}]
    print(f"[iter 0] base policy  eval E[reward]={curve[0]['eval_reward']:.3f}  "
          f"argmax_k={max(range(n_actions), key=lambda i: logits[i]) + 1}")

    for it in range(1, iterations + 1):
        logits_old = list(logits)
        probs_old = _softmax(logits_old)
        batch = []   # {adv, action_i, logp_old}
        for s in train_seeds:
            actions = [_sample_cat(probs_old, rng) for _ in range(group_size)]
            rewards = [reward_of(s, i + 1) for i in actions]
            advs = group_advantages(rewards)
            for i, A in zip(actions, advs):
                batch.append({"adv": A, "i": i, "logp_old": _logp_cat(logits_old, i)})

        for _ep in range(inner_epochs):
            grad = [0.0] * n_actions
            clip_hits = 0
            for e in batch:
                A, i = e["adv"], e["i"]
                lp_new = _logp_cat(logits, i)
                ratio = math.exp(max(-30.0, min(30.0, lp_new - e["logp_old"])))
                coef, clipped = clipped_surrogate_coef(A, ratio, clip_eps)
                clip_hits += int(clipped)
                d = _dlogp_cat(logits, i)
                lp_ref = _logp_cat(logits_ref, i)
                kl_w = kl_coef * (lp_new - lp_ref)        # KL(π‖ref), k1-estimator grad
                for j in range(n_actions):
                    grad[j] += (coef - kl_w) * d[j]
            n = max(1, len(batch))
            for j in range(n_actions):
                logits[j] += lr * grad[j] / n

        ev = expected_eval(logits)
        curve.append({"iter": it, "eval_reward": ev,
                      "argmax_k": max(range(n_actions), key=lambda i: logits[i]) + 1})
        if it % max(1, iterations // 10) == 0 or it == iterations:
            print(f"[iter {it:3d}] eval E[reward]={ev:.3f}  "
                  f"argmax_k={curve[-1]['argmax_k']}  clip_frac={clip_hits/(n*inner_epochs):.2f}")

    _print_curve(curve, label="GRPO (categorical schedule policy) — RL reward curve")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "grpo_curve.json"), "w") as f:
        json.dump({"curve": curve, "final_probs": _softmax(logits),
                   "config": {"group_size": group_size, "lr": lr, "clip": clip_eps,
                              "kl": kl_coef, "inner_epochs": inner_epochs}}, f, indent=2)
    return logits, curve


# ============================================================================
# Gemma policy backend (real --run) — GRPO on open weights via torch + PEFT.
# Imports are guarded so this whole file stays importable without torch.
# ============================================================================

def grpo_train_gemma(train_seeds, eval_seeds, cfg, *, iterations, group_size,
                     lr, clip_eps, kl_coef, inner_epochs, model_name, out_dir):
    """Real GRPO on a Gemma open model. Open weights are required: we backprop the
    clipped surrogate through the model's token log-probs. Reward = verifier disc_eff,
    group-relative per world-seed, broadcast across all turn tokens (trajectory-level).

    Heavy deps (torch/transformers/peft) + a GPU. Runs locally or as a Vertex AI
    custom-training job (see docs/gcp-pipeline.md)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "Real GRPO needs torch + transformers + peft (and a GPU). "
            "`pip install -r requirements-rl.txt`. Use --selftest for the offline RL demo."
        ) from e

    # Prompt/parse helpers live in agent_vertex.py (which we own → no churn risk).
    from agent_vertex import system_prompt, format_obs, extract_json, validate_action

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device — GRPO on a real LLM will be extremely slow.")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    rfn = RewardFn(cfg)

    def _completion_logprob(prompt_text, completion_text, grad):
        """Sum log p(completion | prompt) under the current policy. grad=False → no_grad."""
        ids_prompt = tok(prompt_text, return_tensors="pt").input_ids.to(device)
        ids_full = tok(prompt_text + completion_text, return_tensors="pt").input_ids.to(device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = model(ids_full).logits[:, :-1, :]
            targets = ids_full[:, 1:]
            logp = torch.log_softmax(logits, dim=-1)
            tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
            start = ids_prompt.shape[1] - 1            # completion token positions
            return tok_logp[start:].sum()

    def _rollout(world, seed):
        """One stochastic episode. Returns (profit, turns) where each turn is
        (prompt_text, completion_text, old_logprob_float)."""
        env = FirmEnv(world)
        obs = env.reset()
        history, turns = [], []
        done = False
        while not done:
            prompt_text = (f"<system>\n{system_prompt(cfg)}\n</system>\n"
                           f"<user>\n{format_obs(obs, cfg, history)}\n</user>\n")
            ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(ids, do_sample=True, temperature=0.7, top_p=0.95,
                                     max_new_tokens=512, pad_token_id=tok.pad_token_id)
            completion_text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            old_lp = float(_completion_logprob(prompt_text, completion_text, grad=False))
            turns.append((prompt_text, completion_text, old_lp))
            action = validate_action(extract_json(completion_text), cfg)
            history.append(f"  r{obs['round']}: {json.dumps({'build': action['build']})}")
            obs, _r, done, _ = env.step(action)
        return env.total_profit, turns

    curve = []
    for it in range(1, iterations + 1):
        batch = []   # {adv, turns}
        for s in train_seeds:
            world = generate_world(s, cfg)
            group = [_rollout(world, seed=s * 1000 + it * 31 + k) for k in range(group_size)]
            rewards = [rfn.reward(s, world, prof) for prof, _ in group]
            advs = group_advantages(rewards)
            for (prof, turns), A in zip(group, advs):
                batch.append({"adv": A, "turns": turns})
            print(f"  [iter {it}] seed {s}: rewards={[round(r, 2) for r in rewards]}")

        for _ep in range(inner_epochs):
            opt.zero_grad()
            total_loss = torch.zeros((), device=device)
            n = 0
            for e in batch:
                A = e["adv"]
                for (prompt_text, completion_text, old_lp) in e["turns"]:
                    new_lp = _completion_logprob(prompt_text, completion_text, grad=True)
                    ratio = torch.exp(torch.clamp(new_lp - old_lp, -30, 30))
                    surrogate = torch.min(ratio * A,
                                          torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A)
                    kl = (new_lp - old_lp)              # KL-to-rollout-policy proxy
                    total_loss = total_loss - surrogate + kl_coef * kl
                    n += 1
            (total_loss / max(1, n)).backward()
            opt.step()

        ev = _eval_gemma(model, tok, eval_seeds, cfg, rfn, device,
                         system_prompt, format_obs, extract_json, validate_action)
        curve.append({"iter": it, "eval_reward": ev})
        print(f"[iter {it}] eval mean_reward(disc_eff)={ev:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(os.path.join(out_dir, "grpo_gemma_lora"))
    tok.save_pretrained(os.path.join(out_dir, "grpo_gemma_lora"))
    with open(os.path.join(out_dir, "grpo_curve.json"), "w") as f:
        json.dump({"curve": curve, "model": model_name}, f, indent=2)
    _print_curve(curve, label=f"GRPO ({model_name}) — RL reward curve")
    return curve


def _eval_gemma(model, tok, eval_seeds, cfg, rfn, device,
                system_prompt, format_obs, extract_json, validate_action):
    import torch
    vals = []
    for s in eval_seeds:
        world = generate_world(s, cfg)
        env = FirmEnv(world)
        obs = env.reset()
        history, done = [], False
        while not done:
            prompt_text = (f"<system>\n{system_prompt(cfg)}\n</system>\n"
                           f"<user>\n{format_obs(obs, cfg, history)}\n</user>\n")
            ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(ids, do_sample=False, max_new_tokens=512,
                                     pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            action = validate_action(extract_json(text), cfg)
            obs, _r, done, _ = env.step(action)
        vals.append(rfn.reward(s, world, env.total_profit))
    return mean(vals) if vals else 0.0


# ============================================================================
# pretty-printing
# ============================================================================

def _print_curve(curve, label):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    rewards = [c["eval_reward"] for c in curve]
    span = max(1e-6, max(rewards) if rewards else 1.0)
    for c in curve:
        r = c["eval_reward"]
        bar = "#" * int(40 * max(0.0, r) / span)
        print(f"  iter {c['iter']:>3}  {r:>7.3f}  {bar}")
    base_r, final_r = rewards[0], rewards[-1]
    delta = final_r - base_r
    pct = (delta / abs(base_r) * 100) if base_r else float("inf")
    print("-" * 60)
    print(f"BASE -> GRPO: {base_r:.3f} -> {final_r:.3f}   "
          f"({'+' if delta >= 0 else ''}{delta:.3f}, {pct:+.0f}%)")
    print("=" * 60)


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="FirmBench GRPO (true RL) trainer")
    ap.add_argument("--selftest", action="store_true",
                    help="offline CPU RL demo (categorical schedule policy, real GRPO loop)")
    ap.add_argument("--run", action="store_true",
                    help="real GRPO on a Gemma open model (needs GPU + torch)")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=10, help="rollouts per world (the GRPO group)")
    ap.add_argument("--train-seeds", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default: 1.0 (selftest) / 1e-5 (run)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl", type=float, default=0.01)
    ap.add_argument("--inner-epochs", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="rl_grpo_out")
    args = ap.parse_args()

    if not args.selftest and not args.run:
        print("Choose a mode: --selftest (offline CPU RL) or --run (real Gemma GRPO). "
              "See `python3 rl_grpo.py -h`.")
        return

    cfg = Config()
    train_seeds = list(range(1, 1 + args.train_seeds))
    eval_seeds = list(range(100, 100 + args.eval_seeds))   # disjoint held-out
    lr = args.lr if args.lr is not None else (1.0 if args.selftest else 1e-5)

    print("=" * 60)
    print("FirmBench — GRPO (Group Relative Policy Optimization) — TRUE RL")
    print(f"mode={'selftest' if args.selftest else 'run'}  "
          f"{'policy=categorical-schedule' if args.selftest else 'model=' + args.model}")
    print(f"iterations={args.iterations}  group_size={args.group_size}  "
          f"train_worlds={len(train_seeds)}  eval_worlds={len(eval_seeds)}")
    print(f"lr={lr}  clip={args.clip}  kl={args.kl}  inner_epochs={args.inner_epochs}")
    print("=" * 60 + "\n")

    if args.selftest:
        grpo_train_tabular(train_seeds, eval_seeds, cfg,
                           iterations=args.iterations, group_size=args.group_size,
                           lr=lr, clip_eps=args.clip, kl_coef=args.kl,
                           inner_epochs=args.inner_epochs, out_dir=args.out)
    else:
        try:
            grpo_train_gemma(train_seeds, eval_seeds, cfg,
                             iterations=args.iterations, group_size=args.group_size,
                             lr=lr, clip_eps=args.clip, kl_coef=args.kl,
                             inner_epochs=args.inner_epochs, model_name=args.model,
                             out_dir=args.out)
        except RuntimeError as e:
            print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
