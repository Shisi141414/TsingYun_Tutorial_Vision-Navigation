"""Coordinate conversion helpers shared across the simulator."""

from __future__ import annotations

from typing import Tuple


def world_to_screen(x: float, y: float, cell_size: int) -> Tuple[int, int]:
    """World (x, y) → screen pixel (px, py)."""
    return (int(x * cell_size), int(y * cell_size))


def screen_to_world(px: int, py: int, cell_size: int) -> Tuple[float, float]:
    """Screen pixel (px, py) → world (x, y)."""
    return (px / cell_size, py / cell_size)


def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """World (x, y) → grid (row, col)."""
    return (int(y), int(x))


def grid_to_world(row: int, col: int) -> Tuple[float, float]:
    """Grid cell centre → world (x, y)."""
    return (col + 0.5, row + 0.5)
