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

    # DWA :

    if not global_path:
        return (0.0, 0.0)

    px, py = current_pose
    rows, cols = costmap.shape if costmap is not None else (0, 0)

    # 可调参数
    dt = 0.5               # 预测时长（秒）
    num_samples = 30       # 轨迹采样数
    alpha = 1.0            # 朝向权重
    beta =  2.0             # 安全权重
    gamma = 1.0            # 速度权重

    # 找终点方向（用路径最后一个点）
    gx, gy = global_path[-1]
    goal_dist = math.hypot(gx - px, gy - py)
    goal_dir = math.atan2(gy - py, gx - px)

    best_score = -float('inf')
    best_cmd = (0.0, 0.0)

    for _ in range(num_samples):
        # 随机采样速度
        vx = np.random.uniform(-max_speed, max_speed)
        vy = np.random.uniform(-max_speed, max_speed)
        speed = math.hypot(vx, vy)
        if speed > max_speed:
            vx *= max_speed / speed
            vy *= max_speed / speed

        # 预测未来位置
        nx = px + vx * dt
        ny = py + vy * dt

        # 1. 朝向分数：速度和终点方向的夹角
        if speed > 0.1:
            heading_score = (math.cos(goal_dir) * vx + math.sin(goal_dir) * vy) / speed
        else:
            heading_score = 0.0

        # 2. 安全分数：轨迹末端离障碍物多远
        clearance_score = 0.0
        if costmap is not None:
            gx_cell = int(np.clip(nx, 0, cols - 1))
            gy_cell = int(np.clip(ny, 0, rows - 1))
            clearance_score = 1.0 - costmap[gy_cell, gx_cell] / 255.0

        # 3. 速度分数
        vel_score = speed / max_speed

        score = alpha * heading_score + beta * clearance_score + gamma * vel_score

        if score > best_score:
            best_score = score
            best_cmd = (vx, vy)

    return best_cmd


    # pure persuit controller:

    # # 路径为空 → 停止
    # if not global_path:
    #     return (0.0, 0.0)

    # # 可调参数：前瞻半径
    # Ld = 6.0  # 网格单位
    # goal_Ld = 6.0  # 距离终点小于这个值就开始减速
    # decel_slope = 1.0  # 减速曲线斜率，越大越激进

    # min_speed = 0.1  # 最小速度阈值，低于这个就停

    # # 1. 找到离当前位置最近的路径点索引
    # px, py = current_pose
    # closest_idx = 0
    # closest_dist = float('inf')
    # for i, (wx, wy) in enumerate(global_path):
    #     d = math.hypot(wx - px, wy - py)
    #     # d使用欧氏距离计算，找到最近的路径点索引
    #     if d < closest_dist:
    #         closest_dist = d
    #         closest_idx = i

    # # 1.5 检测前瞻范围内的急转弯，提前减速
    # turn_slowdown = 1.0  # 默认不减速
    # turn_angle_threshold = math.radians(45)  # 转弯角度阈值（弧度）
    # turn_lookahead = Ld * 0.4  # 提前检测范围

    # accumulated_turn = 0.0
    # prev_turn = global_path[closest_idx]
    # for i in range(closest_idx + 1, len(global_path)):
    #     curr_turn = global_path[i]
    #     seg_len = math.hypot(curr_turn[0] - prev_turn[0], curr_turn[1] - prev_turn[1])
    #     accumulated_turn += seg_len

    #     if i < len(global_path) - 1:
    #         next_pt = global_path[i + 1]
    #         # 计算相邻两段路径的转向角
    #         v1 = (curr_turn[0] - prev_turn[0], curr_turn[1] - prev_turn[1])
    #         v2 = (next_pt[0] - curr_turn[0], next_pt[1] - curr_turn[1])
    #         dot = v1[0]*v2[0] + v1[1]*v2[1]
    #         norm = math.hypot(*v1) * math.hypot(*v2)
    #         if norm > 1e-9:
    #             angle = math.acos(max(-1.0, min(1.0, dot / norm)))
    #             if angle > turn_angle_threshold and accumulated_turn < turn_lookahead:
    #                 # 在前瞻范围内有急转弯 → 减速比例
    #                 ratio = 1.0 - (accumulated_turn / turn_lookahead)
    #                 turn_slowdown = min(turn_slowdown, 0.3 + 0.7 * ratio)

    #     if accumulated_turn >= turn_lookahead:
    #         break
    #     prev_turn = curr_turn

    # # 2. 从前瞻点开始向前累计距离，找前瞻点
    # look_ahead = global_path[-1]  # 默认终点
    # accumulated = 0.0
    # prev = global_path[closest_idx] # 从最近点开始累积距离
    # for i in range(closest_idx + 1, len(global_path)):
    #     curr = global_path[i]
    #     seg_len = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
    #     if accumulated + seg_len >= Ld:
    #         # 前瞻点在当前段内，线性插值
    #         remaining = Ld - accumulated
    #         t = remaining / seg_len if seg_len > 0 else 0.0
    #         lx = prev[0] + t * (curr[0] - prev[0])
    #         ly = prev[1] + t * (curr[1] - prev[1])
    #         look_ahead = (lx, ly)
    #         break
    #     accumulated += seg_len
    #     prev = curr

    # # 3. 方向 = 前瞻点 - 当前位置
    # dx = look_ahead[0] - px
    # dy = look_ahead[1] - py
    # dist_to_lookahead = math.hypot(dx, dy)

    # if dist_to_lookahead < 1e-9:
    #     return (0.0, 0.0)

    # # 归一化方向
    # dir_x = dx / dist_to_lookahead
    # dir_y = dy / dist_to_lookahead

    # # 4. 速度大小：接近终点时减速
    # # 计算当前位置到终点的剩余路径长度
    # gx, gy = global_path[-1]
    # dist_to_goal = math.hypot(gx - px, gy - py)

    # if dist_to_goal < goal_Ld:
    #     # 距离终点小于前瞻半径 → 减速，避免冲过终点
    #     speed = max_speed * (dist_to_goal / goal_Ld) ** decel_slope
    #     # 速度太小就停
    #     if speed < min_speed:
    #         return (0.0, 0.0)
    # else:
    #     speed = max_speed

    # # 急转弯减速
    # speed *= turn_slowdown

    # # 调试输出
    # if not hasattr(local_plan, "_start_time"):
    #     local_plan._start_time = time.time()
    # elapsed = time.time() - local_plan._start_time
    # print(f"[local_plan] elapsed={elapsed:.2f}s | Ld={Ld:.2f} goal_Ld={goal_Ld:.2f} min_speed={min_speed:.2f} | "
    #       f"dist_to_goal={dist_to_goal:.2f} speed={speed:.2f}")




    return (dir_x * speed, dir_y * speed)
