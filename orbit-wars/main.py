"""Orbit Wars policy: fixed motion prediction, sun-safe angles, defense, and strong heuristics."""

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


@dataclass
class SnipeOpportunity:
    """Enemy fleet inbound to one of our planets — intercept in open space."""

    fleet: Fleet
    threatened: Planet
    urgency: float  # rough turns until fleet can engage our planet
    priority: float


class OrbitWarsPolicy:
    def __init__(self):
        self.base_reserve_ratio = 0.32
        self.max_candidates_per_planet = 14
        self.base_launch_ratio = 0.82
        self.sun_radius = 10.0
        self.safety_horizon_turns = 18.0
        self.border_radius = 22.0
        self.rotation_radius_limit = 50.0
        self.spawn_clearance = 0.12  # close to env's planet_radius + 0.1

    def _dist_xy(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    def _dist(self, a: Planet, b: Planet) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _fleet_speed(self, ships: int, max_speed: float) -> float:
        ships = max(1, int(ships))
        v = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
        return min(float(max_speed), max(1.0, v))

    def _eta_turns(self, distance: float, ships: int, max_speed: float) -> float:
        return max(1.0, distance / max(1.0, self._fleet_speed(ships, max_speed)))

    def _sun_center(self) -> tuple[float, float]:
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
                models[p.id] = ("orbit", float(angular_velocity))

        return models

    def _predict_target_pos(
        self,
        target: Planet,
        eta_turns: float,
        motion_models: Dict[int, Tuple[str, object]],
    ) -> tuple[float, float]:
        model = motion_models.get(target.id)
        if model is None:
            return float(target.x), float(target.y)
        kind, payload = model
        if kind == "orbit":
            av = float(payload)
            # Advance from *current* position by av * eta (matches env rotation model).
            return self._rotate_point(float(target.x), float(target.y), av * eta_turns)
        if kind == "comet":
            path, path_index = payload
            if not path:
                return float(target.x), float(target.y)
            base = max(0, path_index)
            idx = min(len(path) - 1, base + max(0, int(math.ceil(eta_turns))))
            pt = path[idx]
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                return float(pt[0]), float(pt[1])
        return float(target.x), float(target.y)

    def _predict_linear_fleet(self, fx: float, fy: float, angle: float, speed: float, t: float) -> tuple[float, float]:
        return fx + speed * math.cos(angle) * t, fy + speed * math.sin(angle) * t

    def _comet_steps_remaining(self, motion_models: Dict[int, Tuple[str, object]], target_id: int) -> int | None:
        m = motion_models.get(target_id)
        if not m or m[0] != "comet":
            return None
        path, path_index = m[1]
        if not path:
            return 0
        return max(0, len(path) - 1 - int(path_index))

    def _solve_intercept_coupled(
        self,
        sx: float,
        sy: float,
        ships: int,
        max_speed: float,
        motion_models: Dict[int, Tuple[str, object]],
        target: Planet,
        t_seed: float,
    ) -> tuple[float, float, float, float]:
        """Refine ETA for a moving target with speed depending on fleet size (3–4 iterations)."""
        t = max(1.0, t_seed)
        tx, ty = float(target.x), float(target.y)
        for _ in range(4):
            tx, ty = self._predict_target_pos(target, t, motion_models)
            dist = self._dist_xy(sx, sy, tx, ty)
            t_travel = self._eta_turns(dist, ships, max_speed)
            t = 0.5 * (t + t_travel)
        tx, ty = self._predict_target_pos(target, t, motion_models)
        dist = self._dist_xy(sx, sy, tx, ty)
        return tx, ty, dist, t

    def _solve_snipe_intercept(
        self,
        sx: float,
        sy: float,
        ships: int,
        max_speed: float,
        fx: float,
        fy: float,
        fang: float,
        vf: float,
        t_seed: float,
    ) -> tuple[float, float, float, float]:
        """Coupled refinement for linearly moving enemy fleet."""
        t = max(1.0, t_seed)
        for _ in range(4):
            px, py = self._predict_linear_fleet(fx, fy, fang, vf, t)
            dist = self._dist_xy(sx, sy, px, py)
            t_travel = self._eta_turns(dist, ships, max_speed)
            t = 0.5 * (t + t_travel)
        px, py = self._predict_linear_fleet(fx, fy, fang, vf, t)
        dist = self._dist_xy(sx, sy, px, py)
        return px, py, dist, t

    def _segment_hits_sun(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        cx, cy = self._sun_center()
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-12:
            return math.hypot(x1 - cx, y1 - cy) <= self.sun_radius
        t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        px = x1 + t * dx
        py = y1 + t * dy
        return math.hypot(px - cx, py - cy) <= self.sun_radius

    def _ray_hits_sun(self, x0: float, y0: float, ang: float, seg_len: float) -> bool:
        """Whether the segment from (x0,y0) along ``ang`` for length ``seg_len`` intersects the sun."""
        if seg_len <= 1e-9:
            return False
        x1 = x0 + math.cos(ang) * seg_len
        y1 = y0 + math.sin(ang) * seg_len
        return self._segment_hits_sun(x0, y0, x1, y1)

    def _wrap_pi(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _sun_forbidden_cone_from_launch(self, lx: float, ly: float) -> tuple[float, float] | None:
        """Directions from (lx,ly) that intersect the sun disk: center angle and half-width."""
        cx, cy = self._sun_center()
        dx, dy = cx - lx, cy - ly
        d = math.hypot(dx, dy)
        if d <= self.sun_radius + 1e-6:
            return None
        alpha = math.asin(min(0.999999, self.sun_radius / d))
        center = math.atan2(dy, dx)
        return center, alpha

    def _nearest_angle_outside_cone(self, want: float, center: float, alpha: float) -> float:
        """Pick one of the two cone boundary directions closest to ``want`` (outside forbidden arc)."""
        eps = 0.055
        c1 = self._wrap_pi(center - alpha - eps)
        c2 = self._wrap_pi(center + alpha + eps)
        d1 = abs(self._wrap_pi(want - c1))
        d2 = abs(self._wrap_pi(want - c2))
        return c1 if d1 <= d2 else c2

    def _launch_point(self, mine: Planet, ang: float) -> tuple[float, float]:
        r = mine.radius + self.spawn_clearance
        return mine.x + math.cos(ang) * r, mine.y + math.sin(ang) * r

    def _pick_sun_safe_angle(
        self,
        mine: Planet,
        tx: float,
        ty: float,
        base_angle: float,
    ) -> float | None:
        """Forbidden cone from spawn toward sun; snap the spawn→target ray direction until clear."""
        ang = base_angle
        for _ in range(5):
            lx, ly = self._launch_point(mine, ang)
            if not self._segment_hits_sun(lx, ly, tx, ty):
                return ang
            phi = math.atan2(ty - ly, tx - lx)
            cone = self._sun_forbidden_cone_from_launch(lx, ly)
            if cone is None:
                return None
            center, alpha = cone
            if abs(self._wrap_pi(phi - center)) <= alpha:
                phi = self._nearest_angle_outside_cone(phi, center, alpha)
            ang = phi
        lx, ly = self._launch_point(mine, ang)
        return ang if not self._segment_hits_sun(lx, ly, tx, ty) else None

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
        if players >= 4:
            if step < 110:
                return "opening"
            if step < 270:
                return "mid"
            return "late"
        if step < 85:
            return "opening"
        if step < 240:
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
            pressure += max(0.0, (self.border_radius - d) / self.border_radius) * (p.ships + 5.0 * p.production)
        return pressure

    def _nearest_enemy_planet_distance(self, mine: Planet, planets: List[Planet], player: int) -> float:
        best = 1e9
        for p in planets:
            if p.owner < 0 or p.owner == player:
                continue
            best = min(best, self._dist(mine, p))
        return best

    def _reserve_ratio_frontline_adjust(self, d_enemy: float) -> float:
        """Back-line planets hold less; front-line holds more (delta to add to base ratio)."""
        if d_enemy >= 40.0:
            return -0.10
        if d_enemy >= 28.0:
            return -0.04
        if d_enemy <= 11.0:
            return 0.18
        if d_enemy <= 18.0:
            return 0.10
        return 0.0

    def _snipe_opportunities(
        self,
        fleets: List[Fleet],
        my_planets: List[Planet],
        player: int,
        max_speed: float,
    ) -> List[SnipeOpportunity]:
        out: List[SnipeOpportunity] = []
        enemies = [f for f in fleets if f.owner != player and f.owner >= 0 and f.ships >= 14]
        for f in enemies:
            vf = self._fleet_speed(f.ships, max_speed)
            for p in my_planets:
                fx, fy = float(f.x), float(f.y)
                to_px, to_py = p.x - fx, p.y - fy
                d_plan = math.hypot(to_px, to_py)
                if d_plan < 4.0:
                    continue
                uhx, uhy = to_px / d_plan, to_py / d_plan
                vhx = vf * math.cos(f.angle)
                vhy = vf * math.sin(f.angle)
                vnm = math.hypot(vhx, vhy)
                if vnm < 0.35:
                    continue
                approach = (vhx * uhx + vhy * uhy) / vnm
                if approach < 0.32:
                    continue
                t_guess = (d_plan - p.radius - 3.0) / max(0.45, approach * vnm)
                if t_guess > 72.0 or t_guess < 4.5:
                    continue
                # Enemy path through the sun: fleet dies or intercept geometry is unreliable.
                enemy_path_len = vf * min(48.0, max(8.0, t_guess * 1.35))
                if self._ray_hits_sun(fx, fy, float(f.angle), enemy_path_len):
                    continue
                press = self._defense_pressure(fleets, player, p, max_speed)
                prio = float(f.ships) + 0.15 * press + 42.0 / max(7.0, t_guess)
                out.append(SnipeOpportunity(f, p, float(t_guess), prio))
        by_fleet: Dict[int, SnipeOpportunity] = {}
        for s in out:
            old = by_fleet.get(s.fleet.id)
            if old is None or s.priority > old.priority:
                by_fleet[s.fleet.id] = s
        ranked = sorted(by_fleet.values(), key=lambda s: -s.priority)
        return ranked[:18]

    def _snipe_moves(
        self,
        opportunities: List[SnipeOpportunity],
        my_planets: List[Planet],
        fleets: List[Fleet],
        player: int,
        max_speed: float,
        spent_from: Dict[int, int],
    ) -> List[List[float]]:
        moves: List[List[float]] = []
        for opp in opportunities:
            if len(moves) >= 3:
                break
            f = opp.fleet
            vf = self._fleet_speed(f.ships, max_speed)
            want_ships = min(220, int(f.ships + max(4, int(0.08 * f.ships))))
            donors = [q for q in my_planets if q.id != opp.threatened.id]
            donors.sort(key=lambda q: self._dist_xy(q.x, q.y, f.x, f.y))
            for mine in donors[:7]:
                already = spent_from.get(mine.id, 0)
                eff = mine.ships - already
                if eff < 18:
                    continue
                press_d = self._defense_pressure(fleets, player, mine, max_speed)
                base_res = max(5, int(eff * 0.36)) + press_d
                budget = eff - base_res
                if budget < want_ships:
                    continue
                send = min(budget, want_ships)
                if send < 16:
                    continue
                lx, ly = self._launch_point(mine, math.atan2(f.y - mine.y, f.x - mine.x))
                t0 = max(1.0, self._eta_turns(self._dist_xy(lx, ly, f.x, f.y), send, max_speed))
                px, py, pred_dist, _eta = self._solve_snipe_intercept(
                    lx, ly, send, max_speed, float(f.x), float(f.y), float(f.angle), vf, t0
                )
                # Intercept lies on the enemy's ray; if that ray crosses the sun before the meeting point, skip.
                dist_enemy_to_ip = self._dist_xy(float(f.x), float(f.y), px, py)
                if self._ray_hits_sun(float(f.x), float(f.y), float(f.angle), dist_enemy_to_ip + 0.5):
                    continue
                base_ang = math.atan2(py - mine.y, px - mine.x)
                ang = self._pick_sun_safe_angle(mine, px, py, base_ang)
                if ang is None:
                    continue
                spx, spy = self._launch_point(mine, ang)
                if self._segment_hits_sun(spx, spy, px, py):
                    continue
                moves.append([mine.id, ang, int(send)])
                spent_from[mine.id] = already + int(send)
                break
        return moves

    def _ships_needed_iter(
        self,
        player: int,
        fleets: List[Fleet],
        target: Planet,
        dist: float,
        launch_cap: int,
        max_speed: float,
        margin: int,
    ) -> int:
        s = max(1, min(launch_cap, int(target.ships + margin)))
        for _ in range(5):
            eta = self._eta_turns(dist, s, max_speed)
            growth = int(target.production * eta) if target.owner != -1 else 0
            inbound = self._inbound_delta_before_eta(fleets, player, target, eta, max_speed)
            eff = target.ships + growth - inbound
            need = max(1, int(eff + margin))
            if need == s:
                break
            s = min(launch_cap, need)
        return max(1, s)

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
        is_threat_leader: bool,
    ) -> float:
        cx, cy = self._sun_center()
        sun_d = math.hypot(target.x - cx, target.y - cy)
        centrality = max(0.0, 52.0 - sun_d) / 52.0 * 2.25

        if phase == "opening":
            # Production dominates early; high-prod planets outweigh fat low-prod bodies.
            base_value = 5.85 * (float(target.production) ** 1.42) + 0.32 * target.radius + centrality * 1.15
        else:
            base_value = 3.15 * target.production + 0.38 * target.radius + centrality * 0.95

        neutral_bonus = 2.8 if target.owner == -1 else 0.0
        enemy_bonus = 3.6 if target.owner >= 0 else 0.0
        distance_penalty = 0.118 * dist
        commitment_penalty = 0.065 * ships_needed
        contest_penalty = 0.017 * enemy_near_target

        if phase == "opening":
            neutral_bonus += 2.45
            enemy_bonus -= 1.05
            commitment_penalty += 0.016 * ships_needed
        elif phase == "late":
            enemy_bonus += 1.35
            neutral_bonus -= 0.9

        if players >= 4 and phase in ("opening", "mid") and target.owner >= 0:
            enemy_bonus -= 1.0
            contest_penalty *= 1.28

        if is_threat_leader and players >= 4:
            enemy_bonus *= 0.78
            contest_penalty *= 1.22
            commitment_penalty += 0.014 * ships_needed

        if strength_ratio > 1.4:
            commitment_penalty += 0.022 * ships_needed
            enemy_bonus -= 0.45
        elif strength_ratio < 0.78 and target.owner >= 0:
            enemy_bonus += 1.55

        return base_value + neutral_bonus + enemy_bonus - distance_penalty - commitment_penalty - contest_penalty

    def _target_pool(self, mine: Planet, targets: List[Planet], cap: int | None = None) -> List[Planet]:
        """Mix nearest planets with high production-per-distance value."""
        k = self.max_candidates_per_planet if cap is None else max(4, cap)
        by_dist = sorted(targets, key=lambda t: self._dist(mine, t))[: 4 + k // 2]
        by_value = sorted(
            targets,
            key=lambda t: -(t.production ** 1.55) / (self._dist(mine, t) + 10.0),
        )[:k]
        seen: set[int] = set()
        out: List[Planet] = []
        for t in by_dist + by_value:
            if t.id not in seen:
                seen.add(t.id)
                out.append(t)
        return out[:k]

    def _comet_evacuation_moves(
        self,
        obs,
        planets: List[Planet],
        my_planets: List[Planet],
        player: int,
        max_speed: float,
        motion_models: Dict[int, Tuple[str, object]],
    ) -> List[List[float]]:
        """Last few path steps on an owned comet: dump ships to nearest stable planet."""
        if not isinstance(obs, dict):
            return []
        comet_ids = set(int(x) for x in (obs.get("comet_planet_ids", []) or []))
        if not comet_ids:
            return []
        stable = [p for p in planets if p.id not in comet_ids]
        moves: List[List[float]] = []
        for mine in my_planets:
            if mine.id not in comet_ids:
                continue
            rem = self._comet_steps_remaining(motion_models, mine.id)
            if rem is None or rem > 5:
                continue
            eff = int(mine.ships)
            if eff < 2:
                continue
            ranked: List[tuple[int, float, Planet]] = []
            for p in stable:
                if p.id == mine.id:
                    continue
                d = self._dist(mine, p)
                if p.owner == player:
                    ranked.append((0, d, p))
                elif p.owner == -1:
                    ranked.append((1, d, p))
            if not ranked:
                continue
            ranked.sort(key=lambda x: (x[0], x[1]))
            dest = ranked[0][2]
            send = eff
            eta = self._eta_turns(self._dist(mine, dest), send, max_speed)
            mx, my = self._predict_target_pos(dest, eta, motion_models)
            base_ang = math.atan2(my - mine.y, mx - mine.x)
            ang = self._pick_sun_safe_angle(mine, mx, my, base_ang)
            if ang is None:
                continue
            sx = mine.x + math.cos(ang) * (mine.radius + self.spawn_clearance)
            sy = mine.y + math.sin(ang) * (mine.radius + self.spawn_clearance)
            if self._segment_hits_sun(sx, sy, mx, my):
                continue
            moves.append([mine.id, ang, send])
        return moves

    def _defense_reinforcement_moves(
        self,
        planets: List[Planet],
        fleets: List[Fleet],
        my_planets: List[Planet],
        player: int,
        max_speed: float,
        motion_models: Dict[int, Tuple[str, object]],
        pre_spent: Dict[int, int] | None = None,
    ) -> List[List[float]]:
        moves: List[List[float]] = []
        pre = pre_spent or {}
        if len(my_planets) < 2:
            return moves

        threatened: List[Tuple[Planet, int]] = []
        for p in my_planets:
            press = self._defense_pressure(fleets, player, p, max_speed)
            if press <= 0:
                continue
            gap = press - int(p.ships * 0.55)
            if gap < 10:
                continue
            threatened.append((p, min(gap + 12, 180)))

        threatened.sort(key=lambda x: -x[1])
        donor_used: Dict[int, int] = {}

        for mine, want in threatened[:4]:
            donors = [q for q in my_planets if q.id != mine.id]
            donors.sort(key=lambda q: self._dist(q, mine))
            for donor in donors:
                press_d = self._defense_pressure(fleets, player, donor, max_speed)
                base_res = max(4, int(donor.ships * 0.34)) + press_d
                already = donor_used.get(donor.id, 0) + int(pre.get(donor.id, 0))
                budget = donor.ships - base_res - already
                if budget < 12:
                    continue
                send = min(budget, want)
                if send < 10:
                    continue
                eta = self._eta_turns(self._dist(donor, mine), send, max_speed)
                mx, my = self._predict_target_pos(mine, eta, motion_models)
                base_ang = math.atan2(my - donor.y, mx - donor.x)
                ang = self._pick_sun_safe_angle(donor, mx, my, base_ang)
                if ang is None:
                    continue
                sx = donor.x + math.cos(ang) * (donor.radius + self.spawn_clearance)
                sy = donor.y + math.sin(ang) * (donor.radius + self.spawn_clearance)
                if self._segment_hits_sun(sx, sy, mx, my):
                    continue
                moves.append([donor.id, ang, int(send)])
                donor_used[donor.id] = already + int(send)
                break
        return moves

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
            reserve_ratio += 0.02
            launch_ratio -= 0.05
        if players >= 4:
            reserve_ratio += 0.025
            launch_ratio -= 0.04
        if strength_ratio > 1.28:
            reserve_ratio += 0.12
            launch_ratio -= 0.16
        elif strength_ratio < 0.78:
            reserve_ratio -= 0.04
            launch_ratio += 0.08

        mx_ship = max(totals.values()) if totals else 0
        n_at_max = sum(1 for v in totals.values() if v == mx_ship)
        is_threat_leader = players >= 4 and totals.get(player, 0) == mx_ship and n_at_max == 1
        if is_threat_leader:
            reserve_ratio += 0.11
            launch_ratio -= 0.09

        reserve_ratio = max(0.18, min(0.64, reserve_ratio))
        launch_ratio = max(0.38, min(0.90, launch_ratio))

        snipe_ops = self._snipe_opportunities(fleets, my_planets, player, max_speed)

        moves: List[List[float]] = []
        moves.extend(
            self._comet_evacuation_moves(obs, planets, my_planets, player, max_speed, motion_models)
        )
        evac_spent: Dict[int, int] = {}
        for mv in moves:
            if len(mv) >= 3:
                pid, _, n = int(mv[0]), mv[1], int(mv[2])
                evac_spent[pid] = evac_spent.get(pid, 0) + n
        moves.extend(
            self._defense_reinforcement_moves(
                planets, fleets, my_planets, player, max_speed, motion_models, pre_spent=evac_spent
            )
        )

        spent_from: Dict[int, int] = {}
        for mv in moves:
            if len(mv) >= 3:
                pid, _, n = int(mv[0]), mv[1], int(mv[2])
                spent_from[pid] = spent_from.get(pid, 0) + n

        n_before_snipe = len(moves)
        moves.extend(self._snipe_moves(snipe_ops, my_planets, fleets, player, max_speed, spent_from))
        for mv in moves[n_before_snipe:]:
            if len(mv) >= 3:
                pid, _, n = int(mv[0]), mv[1], int(mv[2])
                spent_from[pid] = spent_from.get(pid, 0) + n

        committed_to_target: Dict[int, int] = {}

        border_w = 0.11 if is_threat_leader and players >= 4 else 0.09
        target_pool_cap = self.max_candidates_per_planet
        if is_threat_leader and players >= 4:
            target_pool_cap = min(target_pool_cap, 8)

        for mine in sorted(my_planets, key=lambda p: p.ships, reverse=True):
            already_out = spent_from.get(mine.id, 0)
            eff_ships = mine.ships - already_out
            if eff_ships <= 0:
                continue
            defense_pressure = self._defense_pressure(fleets, player, mine, max_speed)
            border_pressure = int(self._enemy_border_pressure(planets, player, mine) * border_w)
            d_enemy = self._nearest_enemy_planet_distance(mine, planets, player)
            front_adj = self._reserve_ratio_frontline_adjust(d_enemy)
            eff_reserve_ratio = max(0.12, min(0.72, reserve_ratio + front_adj))
            base_reserve = max(4, int(eff_ships * eff_reserve_ratio))
            reserve = base_reserve + defense_pressure + border_pressure
            launch_budget = min(
                eff_ships - reserve,
                int(eff_ships * launch_ratio),
            )
            if launch_budget <= 0:
                continue

            nearest_targets = self._target_pool(mine, targets, target_pool_cap)
            candidates: List[CandidateAction] = []
            for target in nearest_targets:
                comet_rem = self._comet_steps_remaining(motion_models, target.id)
                raw_dist = self._dist(mine, target)
                guess = max(1, min(launch_budget, target.ships + 3))
                t_seed = self._eta_turns(raw_dist, guess, max_speed)
                tx, ty, pred_dist, eta_c = self._solve_intercept_coupled(
                    mine.x, mine.y, guess, max_speed, motion_models, target, t_seed
                )
                if comet_rem is not None and comet_rem < int(math.ceil(eta_c)) + 3:
                    continue

                margin = 3 if target.owner == -1 else 2
                ships_needed = self._ships_needed_iter(
                    player, fleets, target, pred_dist, launch_budget, max_speed, margin
                )
                t_ref = self._eta_turns(pred_dist, ships_needed, max_speed)
                tx, ty, pred_dist, eta_c = self._solve_intercept_coupled(
                    mine.x, mine.y, ships_needed, max_speed, motion_models, target, t_ref
                )
                if comet_rem is not None and comet_rem < int(math.ceil(eta_c)) + 3:
                    continue

                already_committed = committed_to_target.get(target.id, 0)
                adjusted_needed = max(1, ships_needed - already_committed)
                if adjusted_needed > launch_budget:
                    continue

                t_fin = self._eta_turns(pred_dist, adjusted_needed, max_speed)
                tx, ty, pred_dist, eta_c = self._solve_intercept_coupled(
                    mine.x, mine.y, adjusted_needed, max_speed, motion_models, target, t_fin
                )
                if comet_rem is not None and comet_rem < int(math.ceil(eta_c)) + 3:
                    continue

                base_angle = math.atan2(ty - mine.y, tx - mine.x)
                angle = self._pick_sun_safe_angle(mine, tx, ty, base_angle)
                if angle is None:
                    continue
                spawn_x = mine.x + math.cos(angle) * (mine.radius + self.spawn_clearance)
                spawn_y = mine.y + math.sin(angle) * (mine.radius + self.spawn_clearance)
                if self._segment_hits_sun(spawn_x, spawn_y, tx, ty):
                    continue

                enemy_near_target = self._enemy_border_pressure(planets, player, target)
                score = self._score_action(
                    phase,
                    players,
                    strength_ratio,
                    mine,
                    target,
                    pred_dist,
                    adjusted_needed,
                    enemy_near_target,
                    is_threat_leader,
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
            shots = 1
            if phase != "opening" and launch_budget >= 28 and strength_ratio < 1.65:
                shots = 2
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
