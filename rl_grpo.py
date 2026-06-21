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

It's the same RL shape as run.py's `train_reinforce`, lifted from a 2-param toy policy to
(a) a richer GRPO objective and (b) the Gemma LLM as the policy.

Multi-turn handling (episode = 10 LLM calls, one scalar reward): trajectory-level credit
assignment — the episode's group-relative advantage is broadcast to every turn's tokens.

Two ways to run:
  python3 rl_grpo.py --selftest    # offline, CPU, no torch/GPU. A tabular probe-vs-exploit
                                    # policy is optimized by the REAL GRPO loop; prints the
                                    # reward curve bending. Validates the RL machinery.

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
from run import Verifier, RLProbeExploitPolicy

DEFAULT_MODEL = os.environ.get("RL_BASE_MODEL", "google/gemma-2-2b-it")


# ============================================================================
# GRPO core (policy-agnostic): group-relative advantage + clipped surrogate.
# These functions are shared by the tabular selftest and the Gemma trainer.
# ============================================================================

def group_advantages(rewards):
    """GRPO baseline: normalize rewards WITHIN a group (same world, G rollouts).

    A_i = (R_i - mean) / (std + eps). This replaces the value network — the group
    mean is the baseline, so above-average episodes get positive advantage and
    below-average ones get negative advantage (the part SFT-on-winners can't do).
    """
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    m = mean(rewards)
    s = pstdev(rewards)
    if s < 1e-8:
        return [0.0 for _ in rewards]  # no signal to separate them
    return [(r - m) / s for r in rewards]


def clipped_surrogate_coef(advantage, ratio, clip_eps):
    """PPO/GRPO clipped objective, returned as the scalar multiplying d(logp)/dθ.

    surrogate = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).
    d/dθ[ratio * A] = A * ratio * d(logp)/dθ. In the clipped region the gradient is 0
    (the objective is flat), exactly as in PPO. Returns (coef, was_clipped).
    """
    if advantage >= 0:
        clipped = ratio > 1.0 + clip_eps
    else:
        clipped = ratio < 1.0 - clip_eps
    coef = 0.0 if clipped else advantage * ratio
    return coef, clipped


# ============================================================================
# Reward (= run.Verifier discovery efficiency), with oracle cached per seed so
# the hot training loop doesn't recompute the oracle thousands of times.
# ============================================================================

class RewardFn:
    """profit / oracle clipped to [0,1] — identical to run.Verifier.grade()['reward'],
    but memoized per seed. We keep a real Verifier around for parity checks at eval."""

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
# Tabular policy backend (offline selftest) — a stochastic probe-vs-exploit
# policy whose per-round Bernoulli decisions expose log-probs, so the GRPO loop
# can optimize its 2 parameters. Same decision structure as run.RLProbeExploitPolicy.
# ============================================================================

def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class GRPOProbeExploitPolicy(RLProbeExploitPolicy):
    """Like RLProbeExploitPolicy but records, per decision, the (action, feature) pair
    needed to RE-evaluate its log-prob under updated θ (for the importance ratio).

    decision: probe ~ Bernoulli(p), p = sigmoid(θ0 + θ1 * unsolved_frac).
    We store (a in {0,1}, x = unsolved_frac) each round so logπ_θ(a|x) is recomputable.
    """

    def reset(self):
        super().reset()
        self.decisions = []  # list of (a, x)

    def act(self, env, obs):
        # Mirror the parent's decision points but capture (a, x). We reimplement the
        # probe branch so we can record x; everything else delegates to the parent's
        # ingest/exploit logic by calling super().act after fixing the RNG draw.
        cfg = self.w.cfg
        # ingest diagnostics + round-0 probe are identical to parent; reuse by peeking.
        if obs["round"] == 0:
            return super().act(env, obs)

        # recompute the feature x the parent would use, and the decision, recording both
        n_pains = cfg.n_pains
        unsolved = max(0, n_pains - len(self.solved))
        x = unsolved / n_pains
        p = _sigmoid(self.theta[0] + self.theta[1] * x)
        a = 1 if self.rng.random() < p else 0
        self.decisions.append((a, x))
        # now drive the parent's machinery with this exact decision: temporarily force
        # its RNG so its internal `probe = rng.random() < p_probe` matches `a`.
        self._forced_probe = (a == 1)
        return self._act_with_forced_probe(env, obs)

    def _act_with_forced_probe(self, env, obs):
        """Parent's act() body but using self._forced_probe instead of an RNG draw."""
        cfg = self.w.cfg
        built = set(obs["built_features"])
        for c in obs["per_campaign"]:
            tgt = c["target"]
            if len(tgt) == 1:
                p = tgt[0]
                if obs["round"] <= 1:
                    self.pain_demand[p] = c["audience"]
                if c["purchases"] > 0.5 and self._last_built is not None:
                    self.solved[p] = self._last_built
        self._last_built = None

        probe = self._forced_probe
        top_pains = sorted(self.pain_demand, key=self.pain_demand.get, reverse=True)
        unsolved_top = [p for p in top_pains if p not in self.solved]

        if probe and unsolved_top and obs["round"] < cfg.horizon - 1:
            candidates = [x for x in range(cfg.n_features)
                          if x not in built and x not in self.tried_features]
            f = candidates[0] if candidates else None
            if f is not None:
                self.tried_features.add(f)
                self._last_built = f
                # Pure DISCOVERY this round: small diagnostic probes, NO big exploit
                # campaign. This makes explore vs exploit a real tradeoff — so a
                # probe-early / exploit-late SCHEDULE strictly beats always-probe, giving
                # GRPO a learnable gap that shows up in greedy held-out eval.
                test_targets = unsolved_top[:6]
                campaigns = [{"target": {p}, "spend": 60.0} for p in test_targets]
                return {"build": f, "price": 50.0, "campaigns": campaigns}

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


def _rollout_tabular(world, theta, seed):
    """Run one stochastic episode; return (profit, decisions=[(a,x)...])."""
    pol = GRPOProbeExploitPolicy(world, theta=list(theta), seed=seed)
    profit = run_episode(world, pol)
    return profit, pol.decisions


def _logp_tabular(theta, a, x):
    p = _sigmoid(theta[0] + theta[1] * x)
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p) if a == 1 else math.log(1 - p)


def _dlogp_tabular(theta, a, x):
    """∂logπ_θ(a|x)/∂θ for a Bernoulli-sigmoid head: (a - p) and (a - p)·x."""
    p = _sigmoid(theta[0] + theta[1] * x)
    g = (a - p)
    return [g, g * x]


def grpo_train_tabular(train_seeds, eval_seeds, cfg, *, iterations, group_size,
                       lr, clip_eps, kl_coef, inner_epochs, out_dir, seed=0):
    """The GRPO loop over the tabular policy. This is genuine RL: group-relative
    advantage + clipped importance-weighted policy gradient + KL to a frozen reference."""
    rng = random.Random(seed)
    theta = [0.0, 0.0]
    theta_ref = list(theta)          # frozen reference for the KL penalty
    rfn = RewardFn(cfg)
    curve = []

    base = eval_tabular(theta, eval_seeds, cfg, rfn)
    curve.append({"iter": 0, "eval_reward": base})
    print(f"[iter 0] base policy  eval mean_reward(disc_eff)={base:.3f}  theta={_fmt(theta)}")

    for it in range(1, iterations + 1):
        theta_old = list(theta)      # behavior policy for this batch (ratio denominator)
        batch = []                   # list of episodes: {adv, decisions, logp_old}

        for s in train_seeds:
            world = generate_world(s, cfg)
            group = [_rollout_tabular(world, theta_old, seed=s * 1000 + it * 31 + k)
                     for k in range(group_size)]
            rewards = [rfn.reward(s, world, prof) for prof, _ in group]
            advs = group_advantages(rewards)
            for (prof, decisions), A in zip(group, advs):
                logp_old = [_logp_tabular(theta_old, a, x) for (a, x) in decisions]
                batch.append({"adv": A, "decisions": decisions, "logp_old": logp_old})

        # K inner epochs of clipped policy-gradient ascent on the collected batch
        n_terms = max(1, sum(len(e["decisions"]) for e in batch))
        for _ep in range(inner_epochs):
            grad = [0.0, 0.0]
            clip_hits = 0
            for e in batch:
                A = e["adv"]
                for (a, x), lp_old in zip(e["decisions"], e["logp_old"]):
                    lp_new = _logp_tabular(theta, a, x)
                    ratio = math.exp(max(-30.0, min(30.0, lp_new - lp_old)))
                    coef, clipped = clipped_surrogate_coef(A, ratio, clip_eps)
                    clip_hits += int(clipped)
                    d = _dlogp_tabular(theta, a, x)
                    grad[0] += coef * d[0]
                    grad[1] += coef * d[1]
                    # KL(π_θ || π_ref) penalty, k1 estimator gradient: -kl·(logπ-logπ_ref)·∇logπ
                    lp_ref = _logp_tabular(theta_ref, a, x)
                    kl_w = kl_coef * (lp_new - lp_ref)
                    grad[0] -= kl_w * d[0]
                    grad[1] -= kl_w * d[1]
            theta[0] += lr * grad[0] / n_terms
            theta[1] += lr * grad[1] / n_terms

        ev = eval_tabular(theta, eval_seeds, cfg, rfn)
        curve.append({"iter": it, "eval_reward": ev, "theta": list(theta)})
        if it % max(1, iterations // 10) == 0 or it == iterations:
            print(f"[iter {it:3d}] eval mean_reward(disc_eff)={ev:.3f}  theta={_fmt(theta)}  "
                  f"clip_frac={clip_hits/(n_terms*inner_epochs):.2f}")

    _print_curve(curve, label="GRPO (tabular policy) — RL reward curve")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "grpo_curve.json"), "w") as f:
        json.dump({"curve": curve, "theta": theta,
                   "config": {"group_size": group_size, "lr": lr, "clip": clip_eps,
                              "kl": kl_coef, "inner_epochs": inner_epochs}}, f, indent=2)
    return theta, curve


def eval_tabular(theta, eval_seeds, cfg, rfn):
    """Greedy eval: probe iff p_probe >= 0.5. Mean disc_eff over held-out seeds."""
    vals = []
    for s in eval_seeds:
        world = generate_world(s, cfg)
        pol = _GreedyProbeExploit(world, theta=list(theta), seed=s)
        profit = run_episode(world, pol)
        vals.append(rfn.reward(s, world, profit))
    return mean(vals) if vals else 0.0


class _GreedyProbeExploit(GRPOProbeExploitPolicy):
    """Deterministic version for eval: decision = (p_probe >= 0.5)."""

    def act(self, env, obs):
        if obs["round"] == 0:
            return RLProbeExploitPolicy.act(self, env, obs)
        cfg = self.w.cfg
        x = max(0, cfg.n_pains - len(self.solved)) / cfg.n_pains
        p = _sigmoid(self.theta[0] + self.theta[1] * x)
        self._forced_probe = (p >= 0.5)
        return self._act_with_forced_probe(env, obs)


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
            sys_t = system_prompt(cfg)
            usr_t = format_obs(obs, cfg, history)
            prompt_text = f"<system>\n{sys_t}\n</system>\n<user>\n{usr_t}\n</user>\n"
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
            print(f"  [iter {it}] seed {s}: rewards={[round(r,2) for r in rewards]}")

        for _ep in range(inner_epochs):
            opt.zero_grad()
            total_loss = torch.zeros((), device=device)
            n = 0
            for e in batch:
                A = e["adv"]
                for (prompt_text, completion_text, old_lp) in e["turns"]:
                    new_lp = _completion_logprob(prompt_text, completion_text, grad=True)
                    ratio = torch.exp(torch.clamp(new_lp - old_lp, -30, 30))
                    unclipped = ratio * A
                    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
                    surrogate = torch.min(unclipped, clipped)
                    # KL-to-ref proxy: keep the policy near the rollout policy (old_lp).
                    kl = (new_lp - old_lp)
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

def _fmt(theta):
    return "[" + ", ".join(f"{t:+.3f}" for t in theta) + "]"


def _print_curve(curve, label):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    rewards = [c["eval_reward"] for c in curve]
    hi = max(rewards) if rewards else 1.0
    span = max(1e-6, hi)
    for c in curve:
        r = c["eval_reward"]
        bar = "#" * int(40 * max(0.0, r) / span)
        print(f"  iter {c['iter']:>3}  {r:>7.3f}  {bar}")
    base_r, final_r = rewards[0], rewards[-1]
    delta = final_r - base_r
    print("-" * 60)
    pct = (delta / abs(base_r) * 100) if base_r else float("inf")
    print(f"BASE -> GRPO: {base_r:.3f} -> {final_r:.3f}   "
          f"({'+' if delta >= 0 else ''}{delta:.3f}, {pct:+.0f}%)")
    print("=" * 60)


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="FirmBench GRPO (true RL) trainer")
    ap.add_argument("--selftest", action="store_true",
                    help="offline CPU RL demo (tabular policy, real GRPO loop)")
    ap.add_argument("--run", action="store_true",
                    help="real GRPO on a Gemma open model (needs GPU + torch)")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=8, help="rollouts per world (the GRPO group)")
    ap.add_argument("--train-seeds", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.3)
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

    print("=" * 60)
    print("FirmBench — GRPO (Group Relative Policy Optimization) — TRUE RL")
    print(f"mode={'selftest' if args.selftest else 'run'}  "
          f"{'policy=tabular' if args.selftest else 'model=' + args.model}")
    print(f"iterations={args.iterations}  group_size={args.group_size}  "
          f"train_worlds={len(train_seeds)}  eval_worlds={len(eval_seeds)}")
    print(f"lr={args.lr}  clip={args.clip}  kl={args.kl}  inner_epochs={args.inner_epochs}")
    print("=" * 60 + "\n")

    if args.selftest:
        grpo_train_tabular(train_seeds, eval_seeds, cfg,
                           iterations=args.iterations, group_size=args.group_size,
                           lr=args.lr, clip_eps=args.clip, kl_coef=args.kl,
                           inner_epochs=args.inner_epochs, out_dir=args.out)
    else:
        grpo_train_gemma(train_seeds, eval_seeds, cfg,
                         iterations=args.iterations, group_size=args.group_size,
                         lr=args.lr, clip_eps=args.clip, kl_coef=args.kl,
                         inner_epochs=args.inner_epochs, model_name=args.model,
                         out_dir=args.out)


if __name__ == "__main__":
    main()
