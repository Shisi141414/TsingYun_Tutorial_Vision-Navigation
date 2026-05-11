"""Omnidirectional circular robot with acceleration-limited dynamics."""

from __future__ import annotations

from typing import Tuple


class Robot:
    """
    State only: position (x, y) and velocity (vx, vy) in world coordinates.
    """

    def __init__(
        self,
        radius: float = 0.4,
        max_speed: float = 4.0,
        max_accel: float = 6.0,
    ) -> None:
        self.radius = radius
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def set_pose(self, pos: Tuple[float, float]) -> None:
        self.x, self.y = pos
        self.vx = 0.0
        self.vy = 0.0

    def get_pose(self) -> Tuple[float, float]:
        return (self.x, self.y)
