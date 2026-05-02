"""
Orbit Wars - RL-ready policy scaffold with stronger baseline behavior.

This version keeps inference lightweight but mirrors RL abstractions:
- parse state
- generate candidate actions
- score actions (critic-like heuristic)
- pick best feasible launches with per-planet budgets

You can later replace `score_action()` with learned model inference.
"""

import math
from dataclasses import dataclass
from typing import Dict, List

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, Fleet, Planet


@dataclass
class CandidateAction:
    from_id: int
    to_id: int
    angle: float
    ships: int
    score: float


class OrbitWarsPolicy:
    def __init__(self):
        # Tuned constants for a stronger hand-crafted "Q-like" scorer.
        self.reserve_ratio = 0.35
        self.max_candidates_per_planet = 6
        self.max_launch_ratio = 0.75
        self.sun_radius = 10.0
        self.safety_horizon_turns = 14.0

    def _dist(self, a: Planet, b: Planet) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _fleet_speed(self, ships: int, max_speed: float) -> float:
        ships = max(1, int(ships))
        # Matches environment speed curve from README.
        return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5

    def _eta_turns(self, distance: float, ships: int, max_speed: float) -> float:
        return max(1.0, distance / max(1.0, self._fleet_speed(ships, max_speed)))

    def _segment_hits_sun(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        cx, cy = CENTER
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-9:
            return math.hypot(x1 - cx, y1 - cy) <= self.sun_radius
        t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        px = x1 + t * dx
        py = y1 + t * dy
        return math.hypot(px - cx, py - cy) <= self.sun_radius

    def _inbound_delta_before_eta(
        self,
        fleets: List[Fleet],
        player: int,
        target: Planet,
        eta_turns: float,
        max_speed: float,
    ) -> int:
        delta = 0
        for f in fleets:
            dist = math.hypot(target.x - f.x, target.y - f.y)
            arrival = dist / max(1.0, self._fleet_speed(f.ships, max_speed))
            if arrival > eta_turns:
                continue
            delta += f.ships if f.owner == player else -f.ships
        return delta

    def _defense_pressure(
        self,
        fleets: List[Fleet],
        player: int,
        mine: Planet,
        max_speed: float,
    ) -> int:
        enemy_inbound = 0
        friendly_inbound = 0
        for f in fleets:
            dist = math.hypot(mine.x - f.x, mine.y - f.y)
            arrival = dist / max(1.0, self._fleet_speed(f.ships, max_speed))
            if arrival > self.safety_horizon_turns:
                continue
            if f.owner == player:
                friendly_inbound += f.ships
            else:
                enemy_inbound += f.ships
        return max(0, enemy_inbound - friendly_inbound)

    def _ships_needed(
        self,
        player: int,
        fleets: List[Fleet],
        mine: Planet,
        target: Planet,
        dist: float,
        launch_ships_guess: int,
        max_speed: float,
    ) -> int:
        travel_turns = self._eta_turns(dist, launch_ships_guess, max_speed)
        growth = int(target.production * travel_turns) if target.owner != -1 else 0
        inbound_delta = self._inbound_delta_before_eta(fleets, player, target, travel_turns, max_speed)

        # If we own the target, negative value means reinforcements still needed.
        if target.owner == player:
            required = max(0, 1 - (target.ships + inbound_delta))
            return max(1, required)

        effective_defenders = target.ships + growth - inbound_delta
        return max(1, int(effective_defenders + 2))

    def _score_action(self, mine: Planet, target: Planet, dist: float, ships_needed: int) -> float:
        target_value = 2.5 * target.production + 0.5 * target.radius
        ownership_bonus = 3.0 if target.owner != -1 else 0.0
        distance_penalty = 0.14 * dist
        commitment_penalty = 0.06 * ships_needed

        # Prefer nearby high-prod planets, with a mild preference to attack enemies.
        return target_value + ownership_bonus - distance_penalty - commitment_penalty

    def choose_moves(self, obs) -> List[List[float]]:
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
        cfg = obs.get("configuration", {}) if isinstance(obs, dict) else {}
        max_speed = float(cfg.get("shipSpeed", 6.0)) if isinstance(cfg, dict) else 6.0
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]

        my_planets = [p for p in planets if p.owner == player]
        targets = [p for p in planets if p.owner != player]
        if not my_planets or not targets:
            return []

        committed_to_target: Dict[int, int] = {}
        moves: List[List[float]] = []

        for mine in sorted(my_planets, key=lambda p: p.ships, reverse=True):
            base_reserve = max(3, int(mine.ships * self.reserve_ratio))
            reserve = base_reserve + self._defense_pressure(fleets, player, mine, max_speed)
            launch_budget = min(
                mine.ships - reserve,
                int(mine.ships * self.max_launch_ratio),
            )
            if launch_budget <= 0:
                continue

            # Build top-k target candidates by distance first (cheap pre-filter).
            nearest_targets = sorted(targets, key=lambda t: self._dist(mine, t))[: self.max_candidates_per_planet]
            candidates: List[CandidateAction] = []
            for target in nearest_targets:
                dist = self._dist(mine, target)
                guess = max(1, min(launch_budget, target.ships + 2))
                ships_needed = self._ships_needed(player, fleets, mine, target, dist, guess, max_speed)
                already_committed = committed_to_target.get(target.id, 0)
                adjusted_needed = max(1, ships_needed - already_committed)
                if adjusted_needed > launch_budget:
                    continue

                angle = math.atan2(target.y - mine.y, target.x - mine.x)
                spawn_x = mine.x + math.cos(angle) * (mine.radius + 0.2)
                spawn_y = mine.y + math.sin(angle) * (mine.radius + 0.2)
                if self._segment_hits_sun(spawn_x, spawn_y, target.x, target.y):
                    continue
                score = self._score_action(mine, target, dist, adjusted_needed)
                candidates.append(
                    CandidateAction(
                        from_id=mine.id,
                        to_id=target.id,
                        angle=angle,
                        ships=adjusted_needed,
                        score=score,
                    )
                )

            if not candidates:
                continue

            best = max(candidates, key=lambda c: c.score)
            if best.score <= 0.0:
                continue

            moves.append([best.from_id, best.angle, best.ships])
            committed_to_target[best.to_id] = committed_to_target.get(best.to_id, 0) + best.ships

        return moves


_POLICY = OrbitWarsPolicy()


def agent(obs):
    return _POLICY.choose_moves(obs)
