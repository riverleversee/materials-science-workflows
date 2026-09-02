"""Path helpers for pressure-dependent BNFF / DNTF study trees."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence


DEFAULT_PRESSURES_GPA = (0, 4, 10)
TRAJ_PREFIX = "anime_"
TRAJ_SUFFIX = ".xyz"


def resolve_data_dir(data_dir: str | None = None) -> str:
    """
    Resolve the on-disk study root.

    Priority:
      1. Explicit ``data_dir`` argument
      2. Environment variable ``NMA_DATA_DIR``
      3. ``BNFFanalysis/minpress`` next to this package (repo default layout)
    """
    if data_dir:
        return os.path.abspath(data_dir)
    env = os.environ.get("NMA_DATA_DIR")
    if env:
        return os.path.abspath(env)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_root, "BNFFanalysis", "minpress")


def animation_dir(data_dir: str, pressure_gpa: int) -> str:
    return os.path.join(data_dir, f"{pressure_gpa}GPa", "AnimationFiles")


def optimized_cell_path(data_dir: str, pressure_gpa: int) -> str:
    return os.path.join(data_dir, f"optimized_cell{pressure_gpa}GPa.cell")


def trajectory_path(data_dir: str, pressure_gpa: int, mode_index: int) -> str:
    return os.path.join(
        animation_dir(data_dir, pressure_gpa),
        f"{TRAJ_PREFIX}{mode_index}{TRAJ_SUFFIX}",
    )


def modes_in_dir(dirpath: str, traj_prefix: str = TRAJ_PREFIX, traj_suffix: str = TRAJ_SUFFIX) -> set[int]:
    if not os.path.isdir(dirpath):
        return set()
    modes: set[int] = set()
    for fname in os.listdir(dirpath):
        if fname.startswith("._"):
            continue
        if not (fname.startswith(traj_prefix) and fname.endswith(traj_suffix)):
            continue
        mid = fname[len(traj_prefix) : -len(traj_suffix)]
        if mid.isdigit():
            modes.add(int(mid))
    return modes


def discover_common_modes(
    animation_dirs: Sequence[str],
    traj_prefix: str = TRAJ_PREFIX,
    traj_suffix: str = TRAJ_SUFFIX,
) -> List[int]:
    """Return mode indices present in every pressure's AnimationFiles folder."""
    existing = [path for path in animation_dirs if os.path.isdir(path)]
    if not existing:
        return []
    common = set.intersection(*(modes_in_dir(path, traj_prefix, traj_suffix) for path in existing))
    return sorted(common)


def build_study_paths(
    data_dir: str | None = None,
    pressures_gpa: Iterable[int] | None = None,
    mode_indices: Sequence[int] | None = None,
) -> dict:
    """
    Build the path lists expected by the group-resolved analysis driver.

    Returns a dict with keys:
      data_dir, pressures, unit_cells, position_paths, trajectory_paths, mode_trajectory_grid
    """
    root = resolve_data_dir(data_dir)
    pressures = list(pressures_gpa or DEFAULT_PRESSURES_GPA)
    anim_dirs = [animation_dir(root, p) for p in pressures]
    modes = list(mode_indices) if mode_indices is not None else discover_common_modes(anim_dirs)

    unit_cells = [optimized_cell_path(root, p) for p in pressures]
    position_paths = [
        os.path.join(anim_dirs[i], f"{TRAJ_PREFIX}{modes[0]}{TRAJ_SUFFIX}") if modes else ""
        for i in range(len(pressures))
    ]
    mode_trajectory_grid = [
        [trajectory_path(root, pressure, mode) for mode in modes]
        for pressure in pressures
    ]

    return {
        "data_dir": root,
        "pressures": pressures,
        "unit_cells": unit_cells,
        "position_paths": position_paths,
        "animation_dirs": anim_dirs,
        "modes": modes,
        "mode_trajectory_grid": mode_trajectory_grid,
    }
