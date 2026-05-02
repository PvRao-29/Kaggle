#!/usr/bin/env python3
"""
Robust local evaluation harness for Orbit Wars agents.

Features:
- Deterministic seed sweeps (best effort; depends on environment support)
- Multiple opponent specs in one run
- 2-player and 4-player matches
- Crash-safe execution with retries and detailed error accounting
- Rich aggregate metrics + per-game JSONL output
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from kaggle_environments import make

AgentCallable = Callable[[dict], List[List[Union[int, float]]]]


@dataclass
class AgentSpec:
    name: str
    source: str
    implementation: Union[str, AgentCallable]


@dataclass
class GameResult:
    game_id: int
    seed: int
    scenario: str
    players: int
    our_slot: int
    reward: float
    rank: int
    status: str
    steps: int
    did_win: bool
    did_survive: bool
    duration_sec: float
    error: Optional[str] = None


def _default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robust local Orbit Wars evaluations.")
    parser.add_argument("--agent-path", default="main.py", help="Path to our agent file.")
    parser.add_argument("--agent-fn", default="agent", help="Callable name in --agent-path.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random"],
        help=(
            "Opponent specs. Built-ins: random. "
            "File agents: /path/to/file.py[:fn]. "
            "Repeat values to increase representation."
        ),
    )
    parser.add_argument("--games", type=int, default=100, help="Total games to run.")
    parser.add_argument("--players", type=int, default=2, choices=[2, 4], help="Match size.")
    parser.add_argument("--episode-steps", type=int, default=500, help="episodeSteps override.")
    parser.add_argument("--ship-speed", type=float, default=6.0, help="shipSpeed override.")
    parser.add_argument("--comet-speed", type=float, default=4.0, help="cometSpeed override.")
    parser.add_argument("--sun-radius", type=float, default=10.0, help="sunRadius override.")
    parser.add_argument("--board-size", type=float, default=100.0, help="boardSize override.")
    parser.add_argument("--seed", type=int, default=1337, help="Base seed.")
    parser.add_argument("--seed-stride", type=int, default=9973, help="Stride between game seeds.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per failed game.")
    parser.add_argument(
        "--shuffle-seating",
        action="store_true",
        help="Rotate our agent across player slots uniformly.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable environment debug mode.")
    parser.add_argument("--out-dir", default="eval_runs", help="Output directory.")
    parser.add_argument("--tag", default="", help="Optional label for this run.")
    return parser


def _load_agent_from_path(path_spec: str, default_fn: str) -> AgentSpec:
    path_str, fn_name = (path_spec.split(":", 1) + [default_fn])[:2]
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Agent file must be .py: {path}")

    module_name = f"_orbit_eval_{path.stem}_{abs(hash((str(path), fn_name))) % 10_000_000}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, fn_name):
        raise AttributeError(f"Function '{fn_name}' not found in {path}")
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"'{fn_name}' in {path} is not callable")
    return AgentSpec(name=f"{path.name}:{fn_name}", source=str(path), implementation=fn)


def _parse_opponent_spec(raw: str) -> AgentSpec:
    if raw == "random":
        return AgentSpec(name="random", source="builtin", implementation="random")
    loaded = _load_agent_from_path(raw, "agent")
    loaded.name = f"opponent:{loaded.name}"
    return loaded


def _build_agents_for_game(
    our_agent: AgentSpec,
    opponents: Sequence[AgentSpec],
    players: int,
    our_slot: int,
    game_rng: random.Random,
) -> Tuple[List[Union[str, AgentCallable]], str]:
    if players < 2:
        raise ValueError("players must be >= 2")
    if not opponents:
        raise ValueError("At least one opponent must be provided")

    sampled = [opponents[game_rng.randrange(len(opponents))] for _ in range(players - 1)]
    labels: List[str] = []
    agents: List[Union[str, AgentCallable]] = []
    for slot in range(players):
        if slot == our_slot:
            agents.append(our_agent.implementation)
            labels.append("ours")
        else:
            opp = sampled.pop(0)
            agents.append(opp.implementation)
            labels.append(opp.name)
    scenario = " | ".join(f"p{idx}:{label}" for idx, label in enumerate(labels))
    return agents, scenario


def _rank_from_rewards(rewards: List[float], idx: int) -> int:
    # Competition ranking: 1 = best. Ties share rank.
    sorted_unique = sorted(set(rewards), reverse=True)
    return sorted_unique.index(rewards[idx]) + 1


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return default


def _run_single_game(
    game_id: int,
    seed: int,
    our_agent: AgentSpec,
    opponents: Sequence[AgentSpec],
    players: int,
    our_slot: int,
    config: Dict[str, Union[int, float]],
    debug: bool,
    game_rng: random.Random,
) -> GameResult:
    agents, scenario = _build_agents_for_game(our_agent, opponents, players, our_slot, game_rng)
    t0 = time.perf_counter()

    env = make(
        "orbit_wars",
        debug=debug,
        configuration={
            **config,
            # Best effort deterministic request; unsupported keys are ignored.
            "randomSeed": seed,
            "seed": seed,
        },
    )
    env.run(agents)
    duration = time.perf_counter() - t0

    final_step = env.steps[-1]
    if our_slot >= len(final_step):
        raise RuntimeError(f"Invalid final step length {len(final_step)} for slot {our_slot}")

    ours = final_step[our_slot]
    rewards = [_safe_float(s.reward, default=0.0) for s in final_step]
    our_reward = rewards[our_slot]
    our_rank = _rank_from_rewards(rewards, our_slot)
    status = str(getattr(ours, "status", "UNKNOWN"))
    steps = max(0, len(env.steps) - 1)

    return GameResult(
        game_id=game_id,
        seed=seed,
        scenario=scenario,
        players=players,
        our_slot=our_slot,
        reward=our_reward,
        rank=our_rank,
        status=status,
        steps=steps,
        did_win=(our_rank == 1),
        did_survive=(status.upper() == "DONE"),
        duration_sec=duration,
    )


def _aggregate(results: Sequence[GameResult]) -> Dict[str, Union[int, float]]:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    rewards = [r.reward for r in ok]
    ranks = [r.rank for r in ok]
    durations = [r.duration_sec for r in ok]
    steps = [r.steps for r in ok]

    if ok:
        mean_reward = statistics.fmean(rewards)
        std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        sem_reward = std_reward / math.sqrt(len(rewards)) if len(rewards) > 0 else 0.0
        ci95 = 1.96 * sem_reward
        mean_rank = statistics.fmean(ranks)
        win_rate = sum(r.did_win for r in ok) / len(ok)
        survival_rate = sum(r.did_survive for r in ok) / len(ok)
    else:
        mean_reward = 0.0
        std_reward = 0.0
        ci95 = 0.0
        mean_rank = 0.0
        win_rate = 0.0
        survival_rate = 0.0

    return {
        "games_total": len(results),
        "games_succeeded": len(ok),
        "games_failed": len(failed),
        "failure_rate": (len(failed) / len(results)) if results else 0.0,
        "mean_reward": mean_reward,
        "reward_stddev": std_reward,
        "reward_ci95_half_width": ci95,
        "mean_rank": mean_rank,
        "win_rate": win_rate,
        "survival_rate": survival_rate,
        "mean_steps": statistics.fmean(steps) if steps else 0.0,
        "mean_duration_sec": statistics.fmean(durations) if durations else 0.0,
        "p95_duration_sec": statistics.quantiles(durations, n=20)[-1] if len(durations) >= 20 else (max(durations) if durations else 0.0),
    }


def _result_to_dict(r: GameResult) -> Dict[str, Union[int, float, str, bool, None]]:
    return {
        "game_id": r.game_id,
        "seed": r.seed,
        "scenario": r.scenario,
        "players": r.players,
        "our_slot": r.our_slot,
        "reward": r.reward,
        "rank": r.rank,
        "status": r.status,
        "steps": r.steps,
        "did_win": r.did_win,
        "did_survive": r.did_survive,
        "duration_sec": r.duration_sec,
        "error": r.error,
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Union[int, float, str, bool, None]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _print_summary(summary: Dict[str, Union[int, float]], out_dir: Path) -> None:
    print("\n=== Orbit Wars Local Eval Summary ===")
    print(f"Output dir: {out_dir}")
    print(f"Games: {summary['games_succeeded']}/{summary['games_total']} succeeded")
    print(f"Failure rate: {summary['failure_rate']:.2%}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Survival rate: {summary['survival_rate']:.2%}")
    print(f"Mean rank: {summary['mean_rank']:.3f}")
    print(
        "Mean reward: "
        f"{summary['mean_reward']:.4f} +/- {summary['reward_ci95_half_width']:.4f} (95% CI half-width)"
    )
    print(f"Reward stddev: {summary['reward_stddev']:.4f}")
    print(f"Mean steps: {summary['mean_steps']:.2f}")
    print(f"Mean duration: {summary['mean_duration_sec']:.3f}s, p95: {summary['p95_duration_sec']:.3f}s")


def main() -> int:
    args = _default_parser().parse_args()
    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.seed_stride == 0:
        raise ValueError("--seed-stride must be non-zero")

    our_agent = _load_agent_from_path(args.agent_path, args.agent_fn)
    opponents = [_parse_opponent_spec(o) for o in args.opponents]

    run_id = time.strftime("%Y%m%d-%H%M%S")
    safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.tag).strip("_")
    run_name = f"{run_id}-{safe_tag}" if safe_tag else run_id
    out_dir = Path(args.out_dir).expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "episodeSteps": int(args.episode_steps),
        "shipSpeed": float(args.ship_speed),
        "cometSpeed": float(args.comet_speed),
        "sunRadius": float(args.sun_radius),
        "boardSize": float(args.board_size),
    }
    run_meta = {
        "agent": {"name": our_agent.name, "source": our_agent.source, "fn": args.agent_fn},
        "opponents": [o.name for o in opponents],
        "games": int(args.games),
        "players": int(args.players),
        "seed": int(args.seed),
        "seed_stride": int(args.seed_stride),
        "max_retries": int(args.max_retries),
        "shuffle_seating": bool(args.shuffle_seating),
        "debug": bool(args.debug),
        "config": config,
        "started_at_unix": time.time(),
    }
    _write_json(out_dir / "run_config.json", run_meta)

    results: List[GameResult] = []
    base_rng = random.Random(args.seed)

    for game_id in range(args.games):
        seed = args.seed + game_id * args.seed_stride
        our_slot = game_id % args.players if args.shuffle_seating else 0
        attempt = 0
        game_result: Optional[GameResult] = None

        while attempt <= args.max_retries:
            attempt_seed = seed + attempt
            try:
                # Independent deterministic stream per game-attempt.
                game_rng = random.Random(base_rng.randrange(10**9) ^ attempt_seed)
                game_result = _run_single_game(
                    game_id=game_id,
                    seed=attempt_seed,
                    our_agent=our_agent,
                    opponents=opponents,
                    players=args.players,
                    our_slot=our_slot,
                    config=config,
                    debug=args.debug,
                    game_rng=game_rng,
                )
                break
            except Exception as exc:
                if attempt >= args.max_retries:
                    game_result = GameResult(
                        game_id=game_id,
                        seed=attempt_seed,
                        scenario="failed-before-scenario",
                        players=args.players,
                        our_slot=our_slot,
                        reward=0.0,
                        rank=args.players,
                        status="ERROR",
                        steps=0,
                        did_win=False,
                        did_survive=False,
                        duration_sec=0.0,
                        error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                    break
                attempt += 1

        assert game_result is not None
        results.append(game_result)

        if (game_id + 1) % max(1, min(25, args.games // 10 or 1)) == 0:
            ok_so_far = sum(1 for r in results if r.error is None)
            print(f"[{game_id + 1}/{args.games}] completed. successes={ok_so_far}, failures={len(results) - ok_so_far}")

    summary = _aggregate(results)
    _write_jsonl(out_dir / "games.jsonl", [_result_to_dict(r) for r in results])
    _write_json(out_dir / "summary.json", summary)
    _print_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
