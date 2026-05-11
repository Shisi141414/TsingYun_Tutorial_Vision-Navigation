"""Task: owns start/goal, timer, mode, and the info dict shown in UI."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Deque, Tuple

if TYPE_CHECKING:
    from core.world import MapWorld


class Task:
    """
    One run = start_pos -> goal_pos. Two modes:

      - "timed": goal is the configured goal_pos. Timer runs until the robot
        is within goal_threshold AND its position has been identical for
        `stable_frames` consecutive ticks.
      - "debug": goal was set by a left-click. Timer is frozen.

    Owns: start_pos, goal_pos, goal_threshold, the current goal, timer state,
    stability detection, and the `info` dict for the UI. World is a pure env.
    """

    def __init__(
        self,
        world: MapWorld,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        goal_threshold: float = 1.0,
        pos_eps: float = 1e-3,
        stable_frames: int = 5,
    ) -> None:
        self.world = world
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.goal_threshold = goal_threshold
        self.pos_eps = pos_eps
        self.stable_frames = stable_frames

        self.pos_history: Deque[Tuple[float, float]] = deque(maxlen=stable_frames)
        self.info = {"goal": "", "timer": "", "status": ""}

        self.reset()

    def reset(self) -> None:
        """Reset world + robot pose, start a fresh timed run."""
        self.world.reset(self.start_pos)

        self.goal = self.goal_pos
        self.mode = "timed"
        self.timer_running = True
        self.elapsed = 0.0
        self.stable = False
        self.pos_history.clear()
        self._refresh_info("reset")

    def update(self, cmd_vx: float, cmd_vy: float, path: list, dt: float) -> None:
        """Advance timer and stability detection by one frame."""
        self.world.update(cmd_vx, cmd_vy, dt)

        status = f"path ({len(path)} pts)" if path else "no path"

        robot_pos = self.world.robot.get_pose()
        self.pos_history.append(robot_pos)
        if len(self.pos_history) == self.stable_frames:
            x0, y0 = self.pos_history[0]
            self.stable = all(
                abs(p[0] - x0) < self.pos_eps and abs(p[1] - y0) < self.pos_eps
                for p in self.pos_history
            )
        else:
            self.stable = False

        dist = math.hypot(robot_pos[0] - self.goal[0], robot_pos[1] - self.goal[1])
        if dist < self.goal_threshold:
            status = "reached!" if self.stable else "arriving..."
            if self.mode == "timed" and self.timer_running and self.stable:
                self.timer_running = False

        if self.timer_running:
            self.elapsed += dt

        self._refresh_info(status)

    def set_debug_goal(self, x: float, y: float) -> None:
        """Switch to debug mode at a new goal; freeze the timer."""
        self.goal = (x, y)
        self.mode = "debug"
        self.timer_running = False
        self.stable = False
        self.pos_history.clear()
        self._refresh_info("debug goal")

    def _fmt_timer(self) -> str:
        return f"{self.elapsed:6.2f}s  [{self.mode}]"

    def _refresh_info(self, status: str) -> None:
        self.info["goal"] = f"({self.goal[0]:.1f}, {self.goal[1]:.1f})"
        self.info["timer"] = self._fmt_timer()
        self.info["status"] = status
