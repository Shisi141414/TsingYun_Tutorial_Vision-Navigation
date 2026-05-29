"""Local planner: Pure Pursuit controller."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import math
import time

def local_plan(
    current_pose: Tuple[float, float],
    max_speed: float,
    max_accel: float,
    global_path: List[Tuple[float, float]],
    costmap: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Convert the next chunk of the global path into a velocity command.

    Parameters
    ----------
    current_pose : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates.
    max_speed : float
        Maximum allowed speed magnitude (grid units / second). The returned
        command vector should not exceed this length.
    max_accel : float
        Maximum allowed acceleration. You may ignore this if the world's
        `step()` already enforces a ramp; otherwise use it to compute a
        feasible command from the current velocity.
    global_path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal
        Waypoints from the global planner, ordered from current pose to goal.
        May be empty if no path was found — in that case return `(0.0, 0.0)`.

    Returns
    -------
    cmd_vx, cmd_vy : float, float
        Desired world-frame velocity in grid units per second. The world step
        will clip this to `max_speed` and ramp toward it at `max_accel`, so
        returning a "pointing at the look-ahead" vector scaled to `max_speed`
        is usually the right move.

    Notes
    -----
    - Pure Pursuit recipe:
        1. Find the look-ahead point on `global_path`: walk forward from the
           closest waypoint to `current_pose` until the cumulative distance
           exceeds a look-ahead radius `Ld` (a tuning constant, e.g. 1.5-2.5
           grid units). If you reach the last waypoint first, use it.
        2. The command direction is `(look_ahead - current_pose)`, normalized.
        3. The command speed is `max_speed` (or a slowed value if the
           remaining path length is short, to ease into the goal).
    - Optional: More complex local programming methods (such as Dynamic
      Window Approach) can be used, or more complex model prediction methods
      (such as MPPI) can be tried.
    """
    # TODO: Implement Pure Pursuit controller.
    # 路径为空 → 停止
    if not global_path:
        return (0.0, 0.0)

    # 可调参数：前瞻半径
    Ld = 4.0  # 网格单位
    goal_Ld = 6.0  # 距离终点小于这个值就开始减速

    min_speed = 0.2  # 最小速度阈值，低于这个就停

    # 1. 找到离当前位置最近的路径点索引
    px, py = current_pose
    closest_idx = 0
    closest_dist = float('inf')
    for i, (wx, wy) in enumerate(global_path):
        d = math.hypot(wx - px, wy - py)
        # d使用欧氏距离计算，找到最近的路径点索引
        if d < closest_dist:
            closest_dist = d
            closest_idx = i

    # 2. 从前瞻点开始向前累计距离，找前瞻点
    look_ahead = global_path[-1]  # 默认终点
    accumulated = 0.0
    prev = global_path[closest_idx] # 从最近点开始累积距离
    for i in range(closest_idx + 1, len(global_path)):
        curr = global_path[i]
        seg_len = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        if accumulated + seg_len >= Ld:
            # 前瞻点在当前段内，线性插值
            remaining = Ld - accumulated
            t = remaining / seg_len if seg_len > 0 else 0.0
            lx = prev[0] + t * (curr[0] - prev[0])
            ly = prev[1] + t * (curr[1] - prev[1])
            look_ahead = (lx, ly)
            break
        accumulated += seg_len
        prev = curr

    # 3. 方向 = 前瞻点 - 当前位置
    dx = look_ahead[0] - px
    dy = look_ahead[1] - py
    dist_to_lookahead = math.hypot(dx, dy)

    if dist_to_lookahead < 1e-9:
        return (0.0, 0.0)

    # 归一化方向
    dir_x = dx / dist_to_lookahead
    dir_y = dy / dist_to_lookahead

    # 4. 速度大小：接近终点时减速
    # 计算当前位置到终点的剩余路径长度
    gx, gy = global_path[-1]
    dist_to_goal = math.hypot(gx - px, gy - py)

    if dist_to_goal < goal_Ld:
        # 距离终点小于前瞻半径 → 减速，避免冲过终点
        speed = max_speed * (dist_to_goal / goal_Ld)
    else:
        speed = max_speed

    # 速度太小就停
    if speed < min_speed:
        return (0.0, 0.0)

    # # 调试输出
    # if not hasattr(local_plan, "_start_time"):
    #     local_plan._start_time = time.time()
    # elapsed = time.time() - local_plan._start_time
    # print(f"[local_plan] elapsed={elapsed:.2f}s | Ld={Ld:.2f} goal_Ld={goal_Ld:.2f} min_speed={min_speed:.2f} | "
    #       f"dist_to_goal={dist_to_goal:.2f} speed={speed:.2f}")

    return (dir_x * speed, dir_y * speed)
