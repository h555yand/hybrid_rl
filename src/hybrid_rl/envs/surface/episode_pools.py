# Copyright 2026 [Aleksey Isaev]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episode pool generation and management.

Provides functions to generate, save, load, and cache episode pools
for reproducible training and evaluation.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from .lightweight_env import LightweightEnv, _is_reachable_by_surface

logger = logging.getLogger(__name__)


def _is_on_wall(env: LightweightEnv, point: np.ndarray) -> bool:
    """Check point is on vertical wall (not bottom/rim, not handle)."""
    _, _, face_id = env.mesh.nearest.on_surface([point[:3]])
    normal_at = env.mesh.face_normals[face_id[0]]
    if abs(float(np.dot(normal_at, env.up_direction))) > 0.5:
        return False

    center = np.array(env.mesh.centroid, dtype=float)
    delta = point[:3] - center
    delta[env.height_axis] = 0.0
    horiz_dist = float(np.linalg.norm(delta))

    extents = env.mesh.bounds[1] - env.mesh.bounds[0]
    horiz_extents = [
        extents[i] for i in range(3) if i != env.height_axis
    ]
    body_radius = min(horiz_extents) / 2.0

    return horiz_dist < body_radius * 1.15


def _get_handle_info(env: LightweightEnv) -> dict[str, Any] | None:
    """Get handle geometry info. Returns None if no handle detected."""
    height_axis = env.height_axis
    extents = env.mesh.bounds[1] - env.mesh.bounds[0]
    horiz_axes = [i for i in range(3) if i != height_axis]

    horiz_extents = [extents[i] for i in horiz_axes]
    min_horiz = min(horiz_extents)
    max_horiz = max(horiz_extents)

    if max_horiz < min_horiz * 1.3:
        return None

    handle_axis = (
        horiz_axes[0]
        if extents[horiz_axes[0]] > extents[horiz_axes[1]]
        else horiz_axes[1]
    )
    sym_axis = (
        horiz_axes[1]
        if handle_axis == horiz_axes[0]
        else horiz_axes[0]
    )

    body_radius = min_horiz / 2.0
    bbox_min_h = env.mesh.bounds[0][handle_axis]
    centroid_h = float(env.mesh.centroid[handle_axis])

    center_option_1 = bbox_min_h + body_radius
    center_option_2 = env.mesh.bounds[1][handle_axis] - body_radius
    if abs(centroid_h - center_option_1) < abs(centroid_h - center_option_2):
        center_h = center_option_1
    else:
        center_h = center_option_2

    center_s = float(env.mesh.centroid[sym_axis])

    return {
        "handle_axis": handle_axis,
        "sym_axis": sym_axis,
        "body_radius": body_radius,
        "center_h": center_h,
        "center_s": center_s,
    }


def _check_handle_between(
    env: LightweightEnv,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    handle_info: dict[str, Any],
) -> bool:
    """Check if start and goal are on opposite sides of handle.
    
    Both must be on wall, near handle axis, on opposite
    sides of symmetry axis.
    """
    if not _is_on_wall(env, start_pos):
        return False
    if not _is_on_wall(env, goal_pos):
        return False

    ha = handle_info["handle_axis"]
    ch = handle_info["center_h"]
    br = handle_info["body_radius"]
    sa = handle_info["sym_axis"]
    cs = handle_info["center_s"]

    # Both near handle side
    if start_pos[ha] < ch + br * 0.3:
        return False
    if goal_pos[ha] < ch + br * 0.3:
        return False

    # Opposite sides of sym axis
    if (start_pos[sa] - cs) * (goal_pos[sa] - cs) > 0:
        return False

    return True


def generate_episode_pools(
    mesh_path: str,
    episodes_per_level: int,
    seed: int,
    curriculum_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate episode pools with fixed start/goal
    positions.

    Supports filtering by same_side and path_blocked
    per curriculum level.

    Args:
        mesh_path: Path to the mesh file.
        episodes_per_level: Number of episodes per
            curriculum level.
        seed: Random seed for reproducibility.
        curriculum_config: Optional curriculum config
            with "levels" and "filters" keys.

    Returns:
        Dict with seed, episodes_per_level,
        curriculum_levels, and levels.
    """
    np.random.seed(seed)
    random.seed(seed)
    env = LightweightEnv(mesh_path, seed=seed)
    curr_levels = list(
        (curriculum_config or {}).get("levels", [])
    )
    curr_filters = list(
        (curriculum_config or {}).get("filters", [])
    )
    use_curriculum = bool(
        curriculum_config is not None
        and curr_levels
    )
    level_specs = (
        curr_levels
        if use_curriculum
        else [(None, None)]
    )
    levels: list[list[dict[str, Any]]] = []

    for level_idx, bounds in enumerate(level_specs):
        min_d, max_d = bounds
        level_pool: list[dict[str, Any]] = []

        # Get filter for this level
        level_filter = (
            curr_filters[level_idx]
            if level_idx < len(curr_filters)
            else {}
        )
        require_same_side = level_filter.get(
            "same_side", None
        )
        require_path_blocked = level_filter.get(
            "path_blocked", None
        )

        require_handle_between = level_filter.get(
            "handle_between", False
        )
        
        handle_info = None
        if require_handle_between:
            handle_info = _get_handle_info(env)
            if handle_info is None:
                logger.warning(
                    "Level %d: handle_between filter "
                    "requested but no handle detected",
                    level_idx,
                )
                require_handle_between = False

        max_attempts = (
            100
            if (
                require_same_side is not None
                or require_path_blocked is not None
                or require_handle_between
            )
            else 1
        )

        for item_idx in range(episodes_per_level):
            goal_pose = None
            final_start_pose = None

            for _attempt in range(max_attempts):
                env.reset()
                start_pose = env.get_pose()
                start_pos = start_pose[:3]

                candidate = (
                    env.get_random_surface_point(
                        reference_pos=start_pos,
                        min_dist=min_d,
                        max_dist=max_d,
                        max_attempts=2000,
                        mesh_sample=True,
                    )
                )
                
                # Check handle_between
                if require_handle_between:
                    if not _check_handle_between(
                        env, start_pos, candidate[:3],
                        handle_info,
                    ):
                        continue

                # Check same_side
                same_side = bool(
                    _is_reachable_by_surface(
                        env,
                        start_pos,
                        candidate[:3],
                    )
                )
                if (
                    require_same_side is not None
                    and same_side
                    != require_same_side
                ):
                    continue

                # Check path_blocked
                if (
                    require_path_blocked
                    is not None
                ):
                    env._current_goal = (
                        np.concatenate([
                            candidate[:3],
                            candidate[3:],
                        ])
                    )
                    sensor = env.get_sensor_data()
                    path_blocked = sensor.get(
                        "path_blocked", False
                    )
                    if (
                        path_blocked
                        != require_path_blocked
                    ):
                        continue

                goal_pose = candidate
                final_start_pose = start_pose
                break

            if goal_pose is None:
                goal_pose = candidate
                final_start_pose = start_pose
                logger.warning(
                    "Level %d ep %d: filter not "
                    "satisfied in %d attempts, "
                    "filter=%s",
                    level_idx,
                    item_idx,
                    max_attempts,
                    level_filter,
                )

            start_pos = final_start_pose[:3]

            same_side = bool(
                _is_reachable_by_surface(
                    env,
                    start_pos,
                    goal_pose[:3],
                )
            )
            env._current_goal = np.concatenate([
                goal_pose[:3],
                goal_pose[3:],
            ])
            sensor = env.get_sensor_data()
            path_blocked = bool(
                sensor.get(
                    "path_blocked", False
                )
            )

            target_distance = float(
                np.linalg.norm(
                    goal_pose[:3] - start_pos
                )
            )

            entry: dict[str, Any] = {
                "pool_index": int(item_idx),
                "start_pos": (
                    final_start_pose[:3].tolist()
                ),
                "start_rot": (
                    final_start_pose[3:].tolist()
                ),
                "goal_pos": (
                    goal_pose[:3].tolist()
                ),
                "goal_rot": (
                    goal_pose[3:].tolist()
                ),
                "target_distance": target_distance,
                "same_side": same_side,
                "path_blocked": path_blocked,
            }
            if use_curriculum:
                entry["curriculum_level"] = int(
                    level_idx
                )
                entry["curriculum_bounds"] = [
                    float(min_d), float(max_d)
                ]
            level_pool.append(entry)

        ss_count = sum(
            1
            for e in level_pool
            if e["same_side"]
        )
        pb_count = sum(
            1
            for e in level_pool
            if e.get("path_blocked", False)
        )
        logger.info(
            "[Pools] Level %d: %d episodes, "
            "same_side=%d (%.0f%%), "
            "path_blocked=%d (%.0f%%), "
            "filter=%s",
            level_idx,
            len(level_pool),
            ss_count,
            100.0 * ss_count
            / max(len(level_pool), 1),
            pb_count,
            100.0 * pb_count
            / max(len(level_pool), 1),
            level_filter,
        )

        levels.append(level_pool)

    return {
        "seed": int(seed),
        "episodes_per_level": int(
            episodes_per_level
        ),
        "curriculum_levels": [
            list(map(float, lvl))
            for lvl in curr_levels
        ],
        "levels": levels,
    }

def get_or_generate_pools(
    mesh_path: str,
    seeds: list[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: list[tuple[float, float]],
    regenerate: bool,
    prefix: str = "train",
    curriculum_filters: list[dict] | None = None,
) -> dict[int, dict[str, Any]]:
    """Get episode pools from cache or generate new ones.

    Args:
        mesh_path: Path to the mesh file.
        seeds: List of random seeds.
        episodes_per_level: Number of episodes per level.
        scripts_dir: Directory for saving/loading pool files.
        curriculum_levels: List of (min_dist, max_dist) tuples.
        regenerate: If True, always regenerate pools.
        prefix: Filename prefix for pool files.

    Returns:
        Dict mapping seed to pool data.
    """
    if not regenerate:
        loaded = _load_pools(
            seeds, scripts_dir, prefix
        )
        if loaded is not None:
            return loaded
        logger.info(
            "[Pools] Some %s files missing — "
            "regenerating...",
            prefix,
        )
    return _generate_and_save_pools(
        mesh_path=mesh_path,
        seeds=seeds,
        episodes_per_level=episodes_per_level,
        scripts_dir=scripts_dir,
        curriculum_levels=curriculum_levels,
        prefix=prefix,
        curriculum_filters=curriculum_filters,
    )


def _generate_and_save_pools(
    mesh_path: str,
    seeds: list[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: list[tuple[float, float]],
    prefix: str = "train",
    curriculum_filters: list[dict] | None = None,
) -> dict[int, dict[str, Any]]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    curriculum_config = (
        {
            "levels": curriculum_levels,
            "filters": curriculum_filters or [],
        }
        if curriculum_levels
        else None
    )
    pools: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        logger.info(
            "[Pools] Generating %s pools for seed=%d...", prefix, seed
        )
        pool = generate_episode_pools(
            mesh_path=mesh_path,
            episodes_per_level=episodes_per_level,
            seed=seed,
            curriculum_config=curriculum_config,
        )
        pools[seed] = pool
        pool_file = (
            scripts_dir / f"{prefix}_episode_pools_seed_{seed}.json"
        )
        with pool_file.open("w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2)
        logger.info("[Pools] Saved to %s", pool_file)
    return pools


def _load_pools(
    seeds: list[int],
    scripts_dir: Path,
    prefix: str = "train",
) -> dict[int, dict[str, Any]] | None:
    pools_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        pool_file = (
            scripts_dir / f"{prefix}_episode_pools_seed_{seed}.json"
        )
        if not pool_file.exists():
            logger.info("[Pools] File not found: %s", pool_file)
            return None
        with pool_file.open(encoding="utf-8") as f:
            pools_by_seed[seed] = json.load(f)
        level_sizes = [
            len(level)
            for level in pools_by_seed[seed].get("levels", [])
        ]
        logger.info(
            "[Pools] Loaded %s seed=%d levels=%s",
            prefix,
            seed,
            level_sizes,
        )
    return pools_by_seed
