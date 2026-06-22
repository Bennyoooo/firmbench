# SaaSBench

### An RL gym where "make the number go up" literally means "build a better business."

An agent runs a SaaS company against a simulated population of ~5,000 real-world users —
discovering hidden demand, building the product, picking channels, and pricing for
retention. The reward is **profit**, computed exactly by a market simulator. No LLM judge,
nothing to fake. Then we trained an open 8B model with real RL and it ran the firm
**+260% better — on markets it had never seen.**

---

## The gap

RL has gyms for math, code, and games — domains with provable answers. It has **nothing**
for the most valuable agentic skill there is: **running a business.** Today you can only
grade "good strategy" with an LLM judge, which is noisy, expensive, and gameable. So nobody
can *train* it.

**Our bet: profit is the perfect reward.** It's a single number a simulator computes
exactly — impossible to argue with, and it only goes up if the agent *actually understands
the market.*

## What it is

Drop an agent into a market whose demand is hidden inside a 5,000-user simulation. Every
episode is a fresh randomized world. Over ~16 rounds the agent runs experiments — probe
demand, build features, choose channels, set price, launch campaigns — to reverse-engineer:

- which customer pains are biggest (hidden personas/segments),
- which feature solves which pain (a hidden mapping),
- which marketing channel reaches which segment,
- price elasticity, a quality bar, and **churn**.

Because it's a **subscription business**: customers pay every round and leave if you
overprice or under-deliver. The agent has to optimize **lifetime value** — acquire, retain,
and not torch cash on a finite market — not chase a one-shot sale. A deterministic funnel
(reach → try → buy → subscribe → churn) turns its actions into profit. That's the reward.

## Why it's a real benchmark, not a vibe

- **No LLM judge.** A 5,000-user funnel computes profit. There's nothing to persuade.
- **Honest [0,1].** reward = profit ÷ a **computed theoretical ceiling** — an optimistic
  true upper bound, not "a policy we hope is optimal." No clipping, no "beat the oracle."
- **Domain-randomized.** New hidden world every episode; generalization is measured on
  held-out seeds, not a within-episode trick.
- **Matched-quad design.** Every hidden latent has an action to exploit it, an observation
  to discover it, *and* a reward channel — proven by an **ablation gate** (`naive < scripted
  < oracle` holds for every latent, 6/7 configs; it fails loudly if a latent ever becomes
  unobservable).
- **Reward-hacking reviewed and closed** — e.g. a subscriber-lifecycle model bounds the
  customer base by the real population, killing a campaign-flood exploit.

## The proof — real RL, and the curve bends

We fine-tuned an **open 8B model (Qwen3-8B)** with **GRPO** — genuine policy-gradient RL,
not SFT — using our profit verifier as the reward (8 candidates/prompt, KL-regularized):

- **In-distribution:** reward 0.193 → 0.367 across 3 epochs (monotonic, +91%).
- **Held-out generalization: 0.147 → 0.529 — +260% on worlds it never trained on.**

It learned a *transferable* market-strategy skill, not memorized seeds.

## It's hard — wide-open headroom

On held-out worlds, a disciplined scripted expert reaches only **~7%** of the theoretical
ceiling. Frontier models separate cleanly (% of ceiling, seed 42):

| GPT-5.5 | GPT-5 | Gemini 3.5 | **Qwen3-8B · our RL** | Claude Sonnet | Opus 4.8 |
|---|---|---|---|---|---|
| 47.7% | 28.9% | 11.2% | **5.0%** | 4.7% | 1.8% |

Our RL'd 8B leaps from its base (**0.0%**) past Claude Sonnet. The benchmark is far from
solved — exactly where you want a frontier RL environment.

## Shipped — runnable today

- **Live on HUD** with 6 MCP tools: `hud eval tasks.py claude`.
- A **replay viewer** that steps through any episode (ad campaigns, features built, profit
  curve) with a model leaderboard, plus a **world explorer** for the hidden market.
- Built on the sponsor stack: **HUD** (host + on-policy RL via `hud.train`), **Fireworks**
  (inference + GRPO fine-tuning), OpenAI · Anthropic · Google (frontier leaderboard).

## Where it goes — a firm of agents

The same verifiable profit reward extends straight to **multi-agent**: split the firm into
**Builder / Marketer / Pricer / Coordinator**, each seeing only its slice and forced to
communicate to win. We grade a **coordination tax** — team profit vs a single-agent,
full-info oracle — so cooperation becomes measurable without an LLM judge. One shared,
role-conditioned checkpoint (parameter sharing). Designed and spec'd as Phase D.

---

**SaaSBench: verifiable, trainable, and far from solved.**
`github.com/Bennyoooo/firmbench` · pitch deck at `/pitch` · replay at `/replay`
