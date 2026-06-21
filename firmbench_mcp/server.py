#!/usr/bin/env python3
"""FirmBench MCP-Gym server (eval-protocol multi-turn RL).

Launches the FirmBench MCP-Gym so the eval-protocol rollout (local pytest, or Fireworks RFT
via --mcp-server) can run full interactive episodes. Mirrors the FrozenLake server.

    python firmbench_mcp/server.py --port 9007 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from firmbench_mcp import FirmBenchMcp


def main():
    ap = argparse.ArgumentParser(description="FirmBench MCP-Gym Server")
    ap.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    server = FirmBenchMcp(seed=args.seed)
    print(f"🚀 FirmBench MCP-Gym on port {args.port} (seed={args.seed}, transport={args.transport})")
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
