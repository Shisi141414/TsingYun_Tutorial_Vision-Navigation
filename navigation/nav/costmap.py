"""Costmap generation: obstacle inflation and lidar-based dynamic costmap."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from scipy.ndimage import distance_transform_edt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.geometry import world_to_grid, grid_to_world

def compute_costmap(
    static_map: np.ndarray,
) -> np.ndarray:
    """
    Build the global costmap by inflating static obstacles.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        0 = free cell, 1 = obstacle cell.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell cost in [0, 255]:
        - obstacle cells get the maximum lethal value, so the planner
          treats them as impassable.
        - free cells near an obstacle get a non-zero cost that decays with
          distance, creating a "buffer" so the planned path keeps clear of
          walls instead of grazing them.
        - free cells far from any obstacle get cost 0.

    Notes
    -----
    - The classical recipe: compute the Euclidean distance from each free cell
      to the nearest obstacle (`scipy.ndimage.distance_transform_edt` does this
      in one call), then map distance → cost so that distance 0 is lethal and
      cost falls off smoothly out to some `inflation_radius`. Beyond that
      radius, cost should be 0.
    - The shape of the decay (linear, exponential, ...) and the magnitude of
      the inflation radius are tuning knobs. Pick something that visibly biases
      the path away from walls without making narrow passages impassable. The
      inflation radius that is too large will also cause the robot to take a
      longer route, wasting time.
    """
    # TODO: Implement a function to compute a costmap from the static map by inflating obstacles.
    distances = distance_transform_edt(static_map == 0)
    inflation_radius = 5.0  # tuning parameter: how far the cost should extend from obstacles
    costmap = np.clip(255 * (1 - distances / inflation_radius), 0, 255)
    costmap[static_map == 1] = 255  # lethal cost for obstacle cells
    return costmap.astype(np.uint8)


def update_local_costmap(
    static_map: np.ndarray,
    robot_pos: Tuple[float, float],
    lidar_scan: np.ndarray,
    lidar_range: float,
    lidar_num_rays: int,
) -> np.ndarray:
    """
    Produce the per-frame costmap by adding a dynamic layer on top of the
    static inflation.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        The same static map passed to `compute_costmap`. Re-inflate it (or
        cache the result) to get the static layer.
    robot_pos : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates. Lidar rays
        originate from this point.
    lidar_scan : np.ndarray, shape (lidar_num_rays,)
        Hit distance for each ray, in grid units. A value equal to `lidar_range`
        means the ray did not hit anything within range.
    lidar_range : float
        Maximum sensing distance of the lidar, in grid units.
    lidar_num_rays : int
        Number of rays in the scan; the i-th ray is at angle
        `2*pi * i / lidar_num_rays` measured from the +x axis.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Static-inflation layer merged with a dynamic layer that marks lidar
        hits as lethal and inflates them with a (smaller) buffer. Use a
        per-cell `max` to combine the two layers so the most conservative
        cost wins.

    Notes
    -----
    - Convert each ray hit `(angle_i, lidar_scan[i])` into a world point
      `(x + d*cos(a), y + d*sin(a))`, then to a grid cell. Mark that cell
      lethal and inflate it.
    - Skip rays where `lidar_scan[i] >= lidar_range` (no hit).
    - Optional but useful: skip hits that land on a cell that is *already*
      a static obstacle; otherwise the lidar's view of a wall keeps
      re-inflating the same area.
    """
    # TODO: Implement a function to update the global costmap with a local dynamic layer based on the lidar scan.
    costmap = compute_costmap(static_map).astype(np.float32)
    rows, cols = costmap.shape
    rx, ry = robot_pos

    dynamic_r = 3  # 动态障碍物膨胀半径

    for i in range(lidar_num_rays):
        d = lidar_scan[i]
        if d >= lidar_range:
            continue

        angle = 2.0 * np.pi * i / lidar_num_rays
        wx = rx + d * np.cos(angle)
        wy = ry + d * np.sin(angle)
        gy, gx = world_to_grid(wx, wy)

        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            continue
        if static_map[gy, gx] == 1:
            continue

        # 命中点标记致命
        costmap[gy, gx] = 255.0

        # 周围膨胀
        for dy in range(-dynamic_r, dynamic_r + 1):
            for dx in range(-dynamic_r, dynamic_r + 1):
                nx, ny = gx + dx, gy + dy
                if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
                    continue
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= dynamic_r:
                    cost = 255.0 * 0.6 * (1.0 - dist / dynamic_r)
                    costmap[ny, nx] = max(costmap[ny, nx], cost)

    return np.clip(costmap, 0, 255).astype(np.uint8)

