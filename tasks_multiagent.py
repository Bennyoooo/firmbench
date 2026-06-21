"""FirmBench — Phase D multi-agent HUD tasks (pattern A: Coordinator-dispatch).

Kept in its own module (separate from the single-agent tasks.py) so the two HUD
Environments don't collide when `hud eval` loads a task file. Role prompts live in
tasks.py (ROLE_PROMPTS / MULTIAGENT_SYSTEM_PROMPT); this file binds the Coordinator-dispatch
prompt to the multi-agent template for seeds 42 / 123 / 7.

    hud eval tasks_multiagent.py claude \\
        --task-ids multiagent_market_discovery_seed42 -y --max-steps 120
"""

from env_multiagent import env, multiagent_market_discovery  # noqa: F401
from tasks import MULTIAGENT_SYSTEM_PROMPT, ROLE_PROMPTS  # noqa: F401

_t1 = multiagent_market_discovery(prompt=MULTIAGENT_SYSTEM_PROMPT, seed=42)
_t1.slug = "multiagent_market_discovery_seed42"

_t2 = multiagent_market_discovery(prompt=MULTIAGENT_SYSTEM_PROMPT, seed=123)
_t2.slug = "multiagent_market_discovery_seed123"

_t3 = multiagent_market_discovery(prompt=MULTIAGENT_SYSTEM_PROMPT, seed=7)
_t3.slug = "multiagent_market_discovery_seed7"

multiagent_tasks = [_t1, _t2, _t3]
tasks = multiagent_tasks   # default list for `hud eval tasks_multiagent.py`
