#!/usr/bin/env python3
"""
Convenience wrapper for standardized local evaluation presets.

Usage:
  python eval_presets.py smoke
  python eval_presets.py dev --tag my-branch
  python eval_presets.py gate --opponents random ./agents/v0_baseline.py:agent
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PRESETS: Dict[str, Dict[str, str]] = {
    "smoke": {
        "games": "10",
        "players": "2",
        "max_retries": "0",
    },
    "dev": {
        "games": "100",
        "players": "2",
        "max_retries": "1",
    },
    "gate": {
        "games": "400",
        "players": "2",
        "max_retries": "1",
    },
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run standard Orbit Wars eval presets.")
    p.add_argument("preset", choices=sorted(PRESETS.keys()), help="Preset to execute.")
    p.add_argument(
        "--opponents",
        nargs="+",
        default=["random"],
        help="Opponent list passed through to local_eval.py.",
    )
    p.add_argument("--agent-path", default="main.py", help="Path to our agent.")
    p.add_argument("--agent-fn", default="agent", help="Agent callable name.")
    p.add_argument("--tag", default="", help="Optional run tag suffix.")
    p.add_argument("--seed", type=int, default=1337, help="Base seed.")
    p.add_argument("--seed-stride", type=int, default=9973, help="Seed stride.")
    p.add_argument("--debug", action="store_true", help="Enable environment debug mode.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated command without executing.",
    )
    return p


def _build_command(args: argparse.Namespace) -> List[str]:
    root = Path(__file__).resolve().parent
    local_eval = root / "local_eval.py"
    preset_cfg = PRESETS[args.preset]
    tag = f"{args.preset}-{args.tag}" if args.tag else args.preset

    cmd = [
        sys.executable,
        str(local_eval),
        "--agent-path",
        args.agent_path,
        "--agent-fn",
        args.agent_fn,
        "--games",
        preset_cfg["games"],
        "--players",
        preset_cfg["players"],
        "--max-retries",
        preset_cfg["max_retries"],
        "--seed",
        str(args.seed),
        "--seed-stride",
        str(args.seed_stride),
        "--shuffle-seating",
        "--tag",
        tag,
        "--opponents",
        *args.opponents,
    ]
    if args.debug:
        cmd.append("--debug")
    return cmd


def main() -> int:
    args = _parser().parse_args()
    cmd = _build_command(args)
    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
