"""Lidar sensor: 360° ray-marching scanner."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.world import MapWorld


class Lidar:
    """
    360° lidar sensor mounted on a robot.
    """

    def __init__(self, max_range: float, num_rays: int, step: float = 0.1) -> None:
        self.max_range = max_range
        self.num_rays = num_rays
        self.step = step
        self.angles: np.ndarray = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
        self.distances: np.ndarray = np.full(num_rays, max_range, dtype=np.float64)

    def scan(self, world: MapWorld):
        """Ray-march from world.robot's pose; write hit distances to self.distances."""

        self.distances.fill(self.max_range)
        n_steps = int(self.max_range / self.step) + 1

        for i, angle in enumerate(self.angles):
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            for k in range(1, n_steps):
                d = k * self.step
                rx = world.robot.x + d * cos_a
                ry = world.robot.y + d * sin_a
                r = int(ry)
                c = int(rx)

                if not (0 <= r < world.rows and 0 <= c < world.cols):
                    self.distances[i] = max(0.0, d - self.step)
                    break

                if world.static_map[r, c] == 1:
                    self.distances[i] = d
                    break

                hit = False
                for obs in world.dynamic_obstacles:
                    if math.hypot(rx - obs.x, ry - obs.y) < obs.radius:
                        self.distances[i] = d
                        hit = True
                        break
                if hit:
                    break