"""MapWorld: static grid map, dynamic obstacles, robot, and collision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from core.robot import Robot

# Default map used when no external map is supplied.
# 0 = free, 1 = obstacle.  Surrounded by a solid border wall.
# Each logical cell below is expanded 4x4 (CELL_SCALE=4) so the actual
# grid stored in DEFAULT_MAP is 80x80 while the layout is defined here
# at 20x20 for readability.
_MAP_LAYOUT = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int8)

CELL_SCALE: int = 4
DEFAULT_MAP: np.ndarray = np.kron(_MAP_LAYOUT, np.ones((CELL_SCALE, CELL_SCALE), dtype=np.int8))


@dataclass
class DynamicObstacle:
    x: float
    y: float
    radius: float
    vx: float
    vy: float
    # Axis-aligned movement bbox (xmin, ymin, xmax, ymax) in world coords.
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class MapWorld:
    """
    Owns the static grid map, dynamic obstacles, and the robot.

    Coordinate convention
    ---------------------
    static_map[row, col] where row = int(y), col = int(x).
    0 = free, 1 = obstacle.
    """

    def __init__(
        self,
        static_map: np.ndarray,
        robot: Robot,
        enable_dynamic: bool = False,
        dynamic_radius: float = 0.4,
        dynamic_speed_range: Tuple[float, float] = (1.0, 3.0),
        dynamic_bboxes: Sequence[Sequence[float]] = (),
        rng_seed: int = 42,
    ) -> None:
        self.static_map = static_map.astype(np.int8)
        self.rows, self.cols = static_map.shape

        self.robot = robot

        self.enable_dynamic = enable_dynamic
        self.dynamic_radius = dynamic_radius
        self.dynamic_speed_range = tuple(dynamic_speed_range)
        self.dynamic_bboxes: List[Tuple[float, float, float, float]] = [
            (float(s[0]), float(s[1]), float(s[2]), float(s[3]))
            for s in dynamic_bboxes
        ]

        self.rng_seed = rng_seed

        self.dynamic_obstacles = []
        if self.enable_dynamic:
            self._spawn_dynamic()

    def _spawn_dynamic(self) -> None:
        rng = np.random.default_rng(self.rng_seed)
        speed_min, speed_max = self.dynamic_speed_range
        for x, y, hw, hh in self.dynamic_bboxes:
            xmin = max(0.0, x - hw)
            ymin = max(0.0, y - hh)
            xmax = min(float(self.cols), x + hw)
            ymax = min(float(self.rows), y + hh)
            speed = float(rng.uniform(speed_min, speed_max))
            angle = float(rng.uniform(0.0, 2 * math.pi))
            self.dynamic_obstacles.append(
                DynamicObstacle(
                    x=float(x), y=float(y), radius=self.dynamic_radius,
                    vx=speed * math.cos(angle),
                    vy=speed * math.sin(angle),
                    bbox=(xmin, ymin, xmax, ymax),
                )
            )

    def _update_obstacles(self, dt: float) -> None:
        if not self.enable_dynamic:
            return
        for obs in self.dynamic_obstacles:
            xmin, ymin, xmax, ymax = obs.bbox

            nx = obs.x + obs.vx * dt
            blocked = (
                nx - obs.radius < xmin or nx + obs.radius > xmax
                or self._hits_static(nx, obs.y, obs.radius)
            )
            if blocked:
                obs.vx = -obs.vx
            else:
                obs.x = nx

            ny = obs.y + obs.vy * dt
            blocked = (
                ny - obs.radius < ymin or ny + obs.radius > ymax
                or self._hits_static(obs.x, ny, obs.radius)
            )
            if blocked:
                obs.vy = -obs.vy
            else:
                obs.y = ny

            obs.x = float(np.clip(obs.x, xmin + obs.radius, xmax - obs.radius))
            obs.y = float(np.clip(obs.y, ymin + obs.radius, ymax - obs.radius))

    def _update_robot(self, cmd_vx: float, cmd_vy: float, dt: float) -> None:
        spd = math.hypot(cmd_vx, cmd_vy)
        if spd > self.robot.max_speed:
            cmd_vx *= self.robot.max_speed / spd
            cmd_vy *= self.robot.max_speed / spd

        dvx = cmd_vx - self.robot.vx
        dvy = cmd_vy - self.robot.vy
        dv = math.hypot(dvx, dvy)
        max_dv = self.robot.max_accel * dt
        if dv > max_dv and dv > 1e-9:
            dvx *= max_dv / dv
            dvy *= max_dv / dv
        self.robot.vx += dvx
        self.robot.vy += dvy

        nx = self.robot.x + self.robot.vx * dt
        ny = self.robot.y + self.robot.vy * dt
        if not self._check_collision(nx, ny, self.robot.radius):
            self.robot.x, self.robot.y = nx, ny
        elif not self._check_collision(nx, self.robot.y, self.robot.radius):
            self.robot.x = nx
            self.robot.vy = 0.0
        elif not self._check_collision(self.robot.x, ny, self.robot.radius):
            self.robot.y = ny
            self.robot.vx = 0.0
        else:
            self.robot.vx = 0.0
            self.robot.vy = 0.0

    def _hits_static(self, x: float, y: float, radius: float) -> bool:
        """Static-map-only collision."""
        if x - radius < 0 or x + radius > self.cols:
            return True
        if y - radius < 0 or y + radius > self.rows:
            return True
        c_lo = max(0, int(math.floor(x - radius)))
        c_hi = min(self.cols - 1, int(math.ceil(x + radius)))
        r_lo = max(0, int(math.floor(y - radius)))
        r_hi = min(self.rows - 1, int(math.ceil(y + radius)))
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if self.static_map[r, c] != 1:
                    continue
                nx = max(float(c), min(x, float(c + 1)))
                ny = max(float(r), min(y, float(r + 1)))
                if (x - nx) ** 2 + (y - ny) ** 2 < radius * radius:
                    return True
        return False

    def _check_collision(self, x: float, y: float, radius: float) -> bool:
        """Static map + dynamic obstacles collision (used for the robot)."""
        if self._hits_static(x, y, radius):
            return True
        for obs in self.dynamic_obstacles:
            if math.hypot(x - obs.x, y - obs.y) < radius + obs.radius:
                return True
        return False

    def reset(self, start_pos: Tuple[float, float]) -> None:
        """Reset robot to start_pos and re-spawn dynamic obstacles."""
        self.robot.set_pose(start_pos)

        self.dynamic_obstacles = []
        if self.enable_dynamic:
            self._spawn_dynamic()

    def update(self, cmd_vx: float, cmd_vy: float, dt: float) -> None:
        """Advance the world by dt: dynamic obstacles first, then robot."""
        self._update_obstacles(dt)
        self._update_robot(cmd_vx, cmd_vy, dt)
