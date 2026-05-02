"""Orbit Wars strategic policy with adaptive macro and tactical safety."""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
        self.base_reserve_ratio = 0.34
        self.max_candidates_per_planet = 10
        self.base_launch_ratio = 0.78
        self.sun_radius = 10.0
        self.safety_horizon_turns = 16.0
        self.border_radius = 20.0
        self.rotation_radius_limit = 50.0

    def _dist(self, a: Planet, b: Planet) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _fleet_speed(self, ships: int, max_speed: float) -> float:
        ships = max(1, int(ships))
        # Matches environment speed curve from README.
        return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5

    def _eta_turns(self, distance: float, ships: int, max_speed: float) -> float:
        return max(1.0, distance / max(1.0, self._fleet_speed(ships, max_speed)))

    def _sun_center(self) -> tuple[float, float]:
        # Kaggle environment versions differ: CENTER may be a scalar half-board size
        # or a 2-tuple coordinate. Handle both safely.
        if isinstance(CENTER, (tuple, list)) and len(CENTER) == 2:
            return float(CENTER[0]), float(CENTER[1])
        c = float(CENTER)
        return c, c

    def _rotate_point(self, x: float, y: float, angle: float) -> tuple[float, float]:
        cx, cy = self._sun_center()
        dx = x - cx
        dy = y - cy
        ca = math.cos(angle)
        sa = math.sin(angle)
        return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca

    def _planet_motion_models(self, obs, planets: List[Planet]) -> Dict[int, Tuple[str, object]]:
        """Build motion hints for orbiting planets and comets."""
        models: Dict[int, Tuple[str, object]] = {}
        if not isinstance(obs, dict):
            return models

        comet_ids = set(obs.get("comet_planet_ids", []) or [])
        comets = obs.get("comets", []) or []
        for group in comets:
            planet_ids = group.get("planet_ids", [])
            paths = group.get("paths", [])
            path_index = int(group.get("path_index", 0))
            for i, pid in enumerate(planet_ids):
                if i < len(paths):
                    models[int(pid)] = ("comet", (paths[i], path_index))

        initial_raw = obs.get("initial_planets", []) or []
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        initial_by_id = {int(p[0]): p for p in initial_raw if len(p) >= 7}
        for p in planets:
            if p.id in comet_ids or p.id in models:
                continue
            init = initial_by_id.get(p.id)
            if init is None:
                continue
            ix, iy = float(init[2]), float(init[3])
            orbital_radius = math.hypot(ix - self._sun_center()[0], iy - self._sun_center()[1])
            if orbital_radius + p.radius < self.rotation_radius_limit:
                models[p.id] = ("orbit", (ix, iy, angular_velocity))

        return models

    def _predict_target_pos(
        self,
        target: Planet,
        eta_turns: float,
        motion_models: Dict[int, Tuple[str, object]],
    ) -> tuple[float, float]:
        model = motion_models.get(target.id)
        if model is None:
            return target.x, target.y
        kind, payload = model
        if kind == "orbit":
            ix, iy, av = payload
            return self._rotate_point(ix, iy, av * eta_turns)
        if kind == "comet":
            path, path_index = payload
            if not path:
                return target.x, target.y
            idx = min(len(path) - 1, path_index + max(0, int(round(eta_turns))))
            if isinstance(path[idx], (list, tuple)) and len(path[idx]) >= 2:
                return float(path[idx][0]), float(path[idx][1])
        return target.x, target.y

    def _segment_hits_sun(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        cx, cy = self._sun_center()
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

    def _total_ships_by_player(self, planets: List[Planet], fleets: List[Fleet]) -> Dict[int, int]:
        totals: Dict[int, int] = {}
        for p in planets:
            if p.owner >= 0:
                totals[p.owner] = totals.get(p.owner, 0) + int(p.ships)
        for f in fleets:
            totals[f.owner] = totals.get(f.owner, 0) + int(f.ships)
        return totals

    def _phase(self, step: int, players: int) -> str:
        # 4-player games need a slightly longer conservative opening.
        if players >= 4:
            if step < 120:
                return "opening"
            if step < 280:
                return "mid"
            return "late"
        if step < 90:
            return "opening"
        if step < 250:
            return "mid"
        return "late"

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

    def _enemy_border_pressure(self, planets: List[Planet], player: int, mine: Planet) -> float:
        pressure = 0.0
        for p in planets:
            if p.owner < 0 or p.owner == player:
                continue
            d = self._dist(mine, p)
            if d > self.border_radius:
                continue
            pressure += max(0.0, (self.border_radius - d) / self.border_radius) * (p.ships + 6.0 * p.production)
        return pressure

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
        effective_defenders = target.ships + growth - inbound_delta
        return max(1, int(effective_defenders + 2))

    def _score_action(
        self,
        phase: str,
        players: int,
        strength_ratio: float,
        mine: Planet,
        target: Planet,
        dist: float,
        ships_needed: int,
        enemy_near_target: float,
    ) -> float:
        base_value = 3.0 * target.production + 0.35 * target.radius
        neutral_bonus = 2.4 if target.owner == -1 else 0.0
        enemy_bonus = 3.4 if target.owner >= 0 else 0.0
        distance_penalty = 0.13 * dist
        commitment_penalty = 0.07 * ships_needed
        contest_penalty = 0.018 * enemy_near_target

        if phase == "opening":
            neutral_bonus += 2.0
            enemy_bonus -= 1.4
            commitment_penalty += 0.02 * ships_needed
        elif phase == "late":
            enemy_bonus += 1.2
            neutral_bonus -= 0.8

        if players >= 4 and phase in ("opening", "mid") and target.owner >= 0:
            # Early FFA enemy attacks often create overextension.
            enemy_bonus -= 1.2
            contest_penalty *= 1.35

        if strength_ratio > 1.35:
            # Protect lead: de-risk expensive attacks.
            commitment_penalty += 0.02 * ships_needed
            enemy_bonus -= 0.5
        elif strength_ratio < 0.80 and target.owner >= 0:
            # If behind, we need more direct disruption.
            enemy_bonus += 1.4

        return base_value + neutral_bonus + enemy_bonus - distance_penalty - commitment_penalty - contest_penalty

    def choose_moves(self, obs) -> List[List[float]]:
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
        step = int(obs.get("step", 0)) if isinstance(obs, dict) else 0
        cfg = obs.get("configuration", {}) if isinstance(obs, dict) else {}
        max_speed = float(cfg.get("shipSpeed", 6.0)) if isinstance(cfg, dict) else 6.0
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        motion_models = self._planet_motion_models(obs, planets)

        my_planets = [p for p in planets if p.owner == player]
        targets = [p for p in planets if p.owner != player]
        if not my_planets or not targets:
            return []

        players = len({p.owner for p in planets if p.owner >= 0} | {f.owner for f in fleets})
        players = max(players, 2)
        phase = self._phase(step, players)
        totals = self._total_ships_by_player(planets, fleets)
        my_total = max(1, totals.get(player, 0))
        best_opp = max((v for k, v in totals.items() if k != player), default=1)
        strength_ratio = my_total / max(1, best_opp)

        reserve_ratio = self.base_reserve_ratio
        launch_ratio = self.base_launch_ratio
        if phase == "opening":
            reserve_ratio += 0.03
            launch_ratio -= 0.06
        if players >= 4:
            reserve_ratio += 0.03
            launch_ratio -= 0.05
        if strength_ratio > 1.25:
            reserve_ratio += 0.14
            launch_ratio -= 0.20
        elif strength_ratio < 0.80:
            reserve_ratio -= 0.05
            launch_ratio += 0.07
        reserve_ratio = max(0.20, min(0.65, reserve_ratio))
        launch_ratio = max(0.38, min(0.90, launch_ratio))

        committed_to_target: Dict[int, int] = {}
        moves: List[List[float]] = []

        for mine in sorted(my_planets, key=lambda p: p.ships, reverse=True):
            defense_pressure = self._defense_pressure(fleets, player, mine, max_speed)
            border_pressure = int(self._enemy_border_pressure(planets, player, mine) * 0.10)
            base_reserve = max(5, int(mine.ships * reserve_ratio))
            reserve = base_reserve + defense_pressure + border_pressure
            launch_budget = min(
                mine.ships - reserve,
                int(mine.ships * launch_ratio),
            )
            if launch_budget <= 0:
                continue

            # Build top-k target candidates by distance first (cheap pre-filter).
            nearest_targets = sorted(targets, key=lambda t: self._dist(mine, t))[: self.max_candidates_per_planet]
            candidates: List[CandidateAction] = []
            for target in nearest_targets:
                raw_dist = self._dist(mine, target)
                guess = max(1, min(launch_budget, target.ships + 2))
                eta_guess = self._eta_turns(raw_dist, guess, max_speed)
                tx, ty = self._predict_target_pos(target, eta_guess, motion_models)
                pred_dist = math.hypot(tx - mine.x, ty - mine.y)
                ships_needed = self._ships_needed(player, fleets, mine, target, pred_dist, guess, max_speed)
                already_committed = committed_to_target.get(target.id, 0)
                adjusted_needed = max(1, ships_needed - already_committed)
                if adjusted_needed > launch_budget:
                    continue

                eta_final = self._eta_turns(pred_dist, adjusted_needed, max_speed)
                tx, ty = self._predict_target_pos(target, eta_final, motion_models)
                final_dist = math.hypot(tx - mine.x, ty - mine.y)
                angle = math.atan2(ty - mine.y, tx - mine.x)
                spawn_x = mine.x + math.cos(angle) * (mine.radius + 0.2)
                spawn_y = mine.y + math.sin(angle) * (mine.radius + 0.2)
                if self._segment_hits_sun(spawn_x, spawn_y, tx, ty):
                    continue
                enemy_near_target = self._enemy_border_pressure(planets, player, target)
                score = self._score_action(
                    phase,
                    players,
                    strength_ratio,
                    mine,
                    target,
                    final_dist,
                    adjusted_needed,
                    enemy_near_target,
                )
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

            candidates.sort(key=lambda c: c.score, reverse=True)
            shots = 2 if (phase != "opening" and launch_budget >= 20 and strength_ratio < 1.8) else 1
            remaining = launch_budget
            launched = 0
            for cand in candidates:
                if launched >= shots:
                    break
                if cand.score <= 0.0 or cand.ships > remaining:
                    continue
                moves.append([cand.from_id, cand.angle, cand.ships])
                committed_to_target[cand.to_id] = committed_to_target.get(cand.to_id, 0) + cand.ships
                remaining -= cand.ships
                launched += 1

        return moves


_POLICY = OrbitWarsPolicy()


def agent(obs):
    return _POLICY.choose_moves(obs)
