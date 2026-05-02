#!/usr/bin/env python3
"""Summarize downloaded Kaggle replays for iteration (win rate, length, ship trajectories).

Agent log JSON from Kaggle is usually timing-only; replays contain full state per step.
Use this after ``download_submission_replays_logs.py`` on a ``logs/<subdir>/replays`` folder.

Examples::

    uv run python scripts/analyze_replays.py logs/submission1/replays --me PVR
    uv run python scripts/analyze_replays.py logs/submission2/replays --me PVR --snapshots 0,80,160,240
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def ship_totals_from_obs(obs: dict[str, Any]) -> dict[int, int]:
    totals: dict[int, int] = defaultdict(int)
    for p in obs.get("planets", []) or []:
        if len(p) >= 7:
            owner, ships = int(p[1]), int(p[5])
            if owner >= 0:
                totals[owner] += ships
    for f in obs.get("fleets", []) or []:
        if len(f) >= 7:
            owner, ships = int(f[1]), int(f[6])
            totals[owner] += ships
    return dict(totals)


def planets_owned(obs: dict[str, Any]) -> dict[int, int]:
    owned: dict[int, int] = defaultdict(int)
    for p in obs.get("planets", []) or []:
        if len(p) >= 7:
            o = int(p[1])
            if o >= 0:
                owned[o] += 1
    return dict(owned)


def player_index(team_names: list[str], me: str) -> int | None:
    hits = [i for i, t in enumerate(team_names) if t == me]
    if not hits:
        return None
    if len(hits) > 1:
        # Rare duplicate display names; first seat is a best-effort default.
        return hits[0]
    return hits[0]


def load_step_obs(steps: list[Any], step_idx: int) -> dict[str, Any] | None:
    """Observation from player-0 view at environment step ``step_idx`` (search backward)."""
    if not steps:
        return None
    # steps[0] is often bootstrap; play is steps[1:]
    for t in range(len(steps) - 1, -1, -1):
        row = steps[t]
        if not row:
            continue
        obs = row[0].get("observation")
        if not isinstance(obs, dict):
            continue
        if int(obs.get("step", -1)) <= step_idx:
            return obs
    return steps[-1][0].get("observation") if steps[-1] else None


def analyze_replay(path: Path, me: str, snapshot_steps: list[int]) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    teams = data.get("info", {}).get("TeamNames") or []
    rewards = data.get("rewards") or []
    steps = data.get("steps") or []
    idx = player_index(teams, me)
    out: dict[str, Any] = {
        "file": path.name,
        "players": len(teams),
        "me_index": idx,
        "me_reward": rewards[idx] if idx is not None and idx < len(rewards) else None,
    }
    if idx is None or not steps:
        return out
    final_obs = steps[-1][0].get("observation") or {}
    out["final_step"] = int(final_obs.get("step", len(steps) - 2))
    out["episode_cap"] = int(data.get("configuration", {}).get("episodeSteps", 500))

    totals_end = ship_totals_from_obs(final_obs)
    total_ships = sum(totals_end.values()) or 1
    out["me_share_end"] = totals_end.get(idx, 0) / total_ships
    out["me_planets_end"] = planets_owned(final_obs).get(idx, 0)

    snaps: dict[int, dict[str, float]] = {}
    for s in snapshot_steps:
        obs = load_step_obs(steps, s)
        if not obs:
            continue
        tot = ship_totals_from_obs(obs)
        tsum = sum(tot.values()) or 1
        snaps[s] = {
            "me_share": tot.get(idx, 0) / tsum,
            "me_planets": float(planets_owned(obs).get(idx, 0)),
        }
    out["snapshots"] = snaps
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate stats from Orbit Wars replay JSON files.")
    p.add_argument("replay_dir", type=Path, help="Directory containing *-replay.json files")
    p.add_argument("--me", default="PVR", help="Your TeamNames label to track")
    p.add_argument(
        "--snapshots",
        default="0,60,120,180,240,300",
        help="Comma-separated env steps to sample ship-share and planet count",
    )
    args = p.parse_args()
    snap_steps = sorted({int(x.strip()) for x in args.snapshots.split(",") if x.strip()})

    paths = sorted(args.replay_dir.glob("*replay.json"))
    if not paths:
        print(f"No replay files under {args.replay_dir}")
        return

    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in paths:
        row = analyze_replay(path, args.me, snap_steps)
        if row.get("me_index") is None:
            skipped += 1
            continue
        rows.append(row)

    n = len(rows)
    wins = sum(1 for r in rows if (r.get("me_reward") or 0) > 0)
    losses = sum(1 for r in rows if (r.get("me_reward") or 0) < 0)
    mean_r = statistics.mean((r["me_reward"] for r in rows if r.get("me_reward") is not None)) if rows else 0.0

    print(f"Directory: {args.replay_dir}")
    print(f"Replays: {len(paths)}  parsed as {args.me}: {n}  skipped (name not found): {skipped}")
    print(f"Wins: {wins}  Losses: {losses}  Win rate: {wins / n:.3f}" if n else "")
    print(f"Mean reward: {mean_r:.3f}")
    print()

    def bucket_stats(label: str, subset: list[dict[str, Any]]) -> None:
        if not subset:
            print(f"{label}: (none)")
            return
        lengths = [r["final_step"] for r in subset]
        shares = [r["me_share_end"] for r in subset]
        print(f"{label}  n={len(subset)}")
        print(f"  final_step: mean={statistics.mean(lengths):.0f}  median={statistics.median(lengths):.0f}")
        print(f"  me ship-share at end: mean={statistics.mean(shares):.3f}")
        for s in snap_steps:
            key = str(s)
            vals = [r["snapshots"][s]["me_share"] for r in subset if s in r.get("snapshots", {})]
            if vals:
                print(f"  me_share @ step<={s}: mean={statistics.mean(vals):.3f}")

    wins_r = [r for r in rows if (r.get("me_reward") or 0) > 0]
    los_r = [r for r in rows if (r.get("me_reward") or 0) < 0]
    print("--- Wins ---")
    bucket_stats("wins", wins_r)
    print()
    print("--- Losses ---")
    bucket_stats("losses", los_r)


if __name__ == "__main__":
    main()
