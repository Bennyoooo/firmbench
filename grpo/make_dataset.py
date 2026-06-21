"""Generate firmbench_prompts.jsonl: one strategy prompt per world seed, with the
oracle profit baked into ground_truth for reward normalization."""
import sys
from eval_protocol.models import EvaluationRow, Message
from sim import generate_world, OraclePolicy, run_episode
from test_firmbench_grpo import build_prompt, cfg  # share the same tight-economy cfg

TRAIN_SEEDS = list(range(1, 25))        # 24 training worlds
EVAL_SEEDS = list(range(100, 116))      # 16 disjoint held-out worlds (generalization)


def write(out, seeds):
    rows = []
    for s in seeds:
        world = generate_world(s, cfg)
        oracle = run_episode(world, OraclePolicy(world))
        rows.append(EvaluationRow(
            messages=[
                Message(role="system", content="You are a sharp startup operator. Reason briefly, then output the JSON plan."),
                Message(role="user", content=build_prompt(world)),
            ],
            ground_truth={"seed": s, "oracle": round(oracle, 1)},
        ))
    with open(out, "w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    print(f"wrote {len(rows)} rows -> {out}; oracle "
          f"{min(r.ground_truth['oracle'] for r in rows):.0f}..{max(r.ground_truth['oracle'] for r in rows):.0f}")


if __name__ == "__main__":
    write("firmbench_prompts.jsonl", TRAIN_SEEDS)
    write("firmbench_eval.jsonl", EVAL_SEEDS)
