"""Global path planner: A* search on a costmap grid."""

from __future__ import annotations

from typing import List, Tuple

import heapq
import math
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.geometry import world_to_grid, grid_to_world

# _cache = {"start": None, "goal": None, "path": []}

def global_plan(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    costmap: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Run path search over `costmap` to find a path from `start` to `goal`.

    Parameters
    ----------
    start : Tuple[float, float], (x, y)
        Start position in world (grid-unit) coordinates. `costmap[int(y), int(x)]`
        is the cell containing this point.
    goal : Tuple[float, float], (x, y)
        Goal position in the same coordinate system.
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell traversal cost. Cells with large cost are treated as impassable
        (lethal). Otherwise the cell's cost is added to the step cost so the
        planner is biased away from inflated/dangerous areas.

    Returns
    -------
    path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal.
        World-coordinate waypoints from start to goal, inclusive of both ends.
        Returns [] if no path exists or if start/goal lie inside a lethal cell.

    Notes
    -----
    - Use 8-connectivity (N/S/E/W + 4 diagonals). Step cost between adjacent
      cells should be `dist + cell_cost`, where `dist` is 1.0 for cardinal moves
      and sqrt(2) for diagonals.
    - Use either a shortest path algorithm (like Dijkstra) or a heuristic search
      algorithm (like A*). If using A*, a good heuristic is the octile distance
      (diagonal distance) or Euclidean distance.
    """
    # TODO: Implement path search on the costmap grid to find a path from start to goal.
    
    # # 结果缓存：如果 start 和 goal 没变且之前计算过路径，则直接返回缓存的路径
    # start_key = (round(start[0], 1), round(start[1], 1))
    # goal_key = (round(goal[0], 1), round(goal[1], 1))
    # if _cache["start"] == start_key and _cache["goal"] == goal_key and _cache["path"]:
    #     return _cache["path"]

    rows, cols = costmap.shape
    # 世界坐标 → 网格坐标
    sy, sx = world_to_grid(start[0], start[1])
    gy, gx = world_to_grid(goal[0], goal[1])

    # 起点或终点在致命格子上 → 返回空
    if costmap[sy, sx] >= 250 or costmap[gy, gx] >= 250:
        return []

    # 8 连通邻格偏移：直线步长 1.0，对角线步长 √2
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    # 启发式：八分位距离（octile distance）
    def heuristic(x, y):
        dx = abs(x - gx)
        dy = abs(y - gy)
        return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)

    # 移动代价 = 步长 + 目标格代价
    def move_cost(dx, dy, dist, nx, ny):
        return dist + costmap[ny, nx]

    # A* 搜索
    open_set = []
    heapq.heappush(open_set, (0.0, sx, sy))
    came_from = {(sx, sy): None}
    g_score = {(sx, sy): 0.0}

    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == (gx, gy):
            break

        for dx, dy, dist in neighbors:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
                continue
            if costmap[ny, nx] >= 250:  # 致命格子不可走
                continue

            new_g = g_score[(cx, cy)] + move_cost(dx, dy, dist, nx, ny)
            if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_g
                f = new_g + heuristic(nx, ny)
                heapq.heappush(open_set, (f, nx, ny))
                came_from[(nx, ny)] = (cx, cy)

    # 终点不可达
    if (gx, gy) not in came_from:
        return []

    # 回溯路径：网格中心坐标
    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append(grid_to_world(cur[1], cur[0]))
        cur = came_from[cur]
    path.reverse()

    # # 更新缓存
    # _cache["start"] = start_key
    # _cache["goal"] = goal_key
    # _cache["path"] = path

    return path
