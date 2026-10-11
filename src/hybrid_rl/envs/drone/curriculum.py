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

"""Curriculum manager for drone training.

Manages difficulty levels with reproducible goal generation via seeds.
Includes trap scenarios for teaching obstacle escape.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# Trap scenario definitions
# ═══════════════════════════════════════════════════════

TRAP_SCENARIOS = [
    {
        "name": "U_wall_open_back",
        "description": "U-shaped wall, open behind drone",
        "obstacles": [
            {"position": [3.0, -1.5, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.0, 1.5, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.0, 0.0, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [2.8, -0.75, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [2.8, 0.75, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.2, -0.75, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.2, 0.75, 1.0], "radius": 0.2, "height": 2.5},
        ],
        "start_offset": [1.5, 0.0, 0.0],
        "goal_offset": [6.0, 0.0, 0.0],
    },
    {
        "name": "corridor_dead_end",
        "description": "Narrow corridor with wall at end",
        "obstacles": [
            # Left wall
            {"position": [1.0, -0.8, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [2.0, -0.8, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.0, -0.8, 1.0], "radius": 0.2, "height": 2.5},
            # Right wall
            {"position": [1.0, 0.8, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [2.0, 0.8, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.0, 0.8, 1.0], "radius": 0.2, "height": 2.5},
            # End wall
            {"position": [3.5, 0.0, 1.0], "radius": 0.3, "height": 2.5},
        ],
        "start_offset": [0.0, 0.0, 0.0],
        "goal_offset": [5.0, 2.0, 0.0],
    },
    {
        "name": "L_corner",
        "description": "L-shaped obstacle requiring turn",
        "obstacles": [
            {"position": [2.5, 0.0, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [2.5, 0.5, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [2.5, 1.0, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [2.5, 1.5, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.0, 1.5, 1.0], "radius": 0.2, "height": 2.5},
            {"position": [3.5, 1.5, 1.0], "radius": 0.2, "height": 2.5},
        ],
        "start_offset": [0.0, 0.0, 0.0],
        "goal_offset": [4.0, 3.0, 0.0],
    },
]


class CurriculumManager:
    """Manages training difficulty progression."""

    def __init__(self, config: dict, seed: int = 42):
        cur_cfg = config.get("curriculum", {})
        self.enabled = cur_cfg.get("enabled", True)
        self.levels = cur_cfg.get("levels", [self._default_level()])
        self.promote_threshold = cur_cfg.get("promote_threshold", 0.7)
        self.demote_threshold = cur_cfg.get("demote_threshold", 0.3)
        self.min_episodes = cur_cfg.get("min_episodes_per_level", 30)

        # Trap injection rate
        self.trap_ratio = cur_cfg.get("trap_ratio", 0.15)
        self.trap_min_level = cur_cfg.get("trap_min_level", 2)

        self.current_level = 0
        self.episodes_at_level = 0
        self.recent_results: deque = deque(maxlen=50)

        self._base_seed = seed
        self._rng = np.random.RandomState(seed)
        self._episode_counter = 0
        self._last_was_trap = False
        self._trap_name = ""

        logger.info(
            "Curriculum: %d levels, seed=%d, enabled=%s, "
            "trap_ratio=%.0f%% (from level %d)",
            len(self.levels), seed, self.enabled,
            self.trap_ratio * 100, self.trap_min_level,
        )
        for i, lvl in enumerate(self.levels):
            logger.info("  L%d: %s", i, lvl["name"])

    @staticmethod
    def _default_level() -> dict:
        return {
            "name": "default",
            "min_dist": 2.0,
            "max_dist": 6.0,
            "min_altitude": 0.3,
            "max_altitude": 4.0,
            "obstacles": False,
            "num_obstacles": 0,
        }

    @property
    def level(self) -> dict:
        return self.levels[self.current_level]

    @property
    def level_name(self) -> str:
        if self._last_was_trap:
            return f"{self.level['name']}+TRAP({self._trap_name})"
        return self.level["name"]

    @property
    def success_rate(self) -> float:
        if len(self.recent_results) == 0:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def last_was_trap(self) -> bool:
        return self._last_was_trap

    def get_episode_seed(self, episode_idx: int) -> int:
        return self._base_seed + self.current_level * 100000 + episode_idx

    def _should_inject_trap(self, episode_idx: int) -> bool:
        """Decide if this episode should be a trap scenario."""
        if not self.enabled:
            return False
        if self.current_level < self.trap_min_level:
            return False
        if self.trap_ratio <= 0:
            return False

        rng = np.random.RandomState(
            self.get_episode_seed(episode_idx) + 777
        )
        return rng.random() < self.trap_ratio

    def _select_trap(self, episode_idx: int) -> dict:
        """Select a trap scenario deterministically."""
        rng = np.random.RandomState(
            self.get_episode_seed(episode_idx) + 888
        )
        idx = rng.randint(len(TRAP_SCENARIOS))
        return TRAP_SCENARIOS[idx]

    def generate_goal(
        self,
        episode_idx: int,
        reference_pos: np.ndarray,
        dome_size: float = 30.0,
    ) -> np.ndarray:
        """Generate reproducible goal for current level.

        If trap episode, goal is placed behind the trap.
        """
        self._last_was_trap = False
        self._trap_name = ""

        if self._should_inject_trap(episode_idx):
            trap = self._select_trap(episode_idx)
            self._last_was_trap = True
            self._trap_name = trap["name"]

            goal_offset = np.array(trap["goal_offset"], dtype=float)
            goal_pos = reference_pos + goal_offset
            goal_pos[2] = max(goal_pos[2], 0.3)
            return np.concatenate([goal_pos, np.zeros(3)])

        # Normal goal generation
        ep_seed = self.get_episode_seed(episode_idx)
        rng = np.random.RandomState(ep_seed)

        lvl = self.level
        min_dist = lvl["min_dist"]
        max_dist = lvl["max_dist"]
        min_alt = lvl["min_altitude"]
        max_alt = lvl["max_altitude"]
        dome_margin = dome_size * 0.8

        for _ in range(100):
            direction = rng.randn(3)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction /= norm

            dist = rng.uniform(min_dist, max_dist)
            point = reference_pos + direction * dist
            point[2] = np.clip(point[2], min_alt, max_alt)

            if np.linalg.norm(point) < dome_margin:
                return np.concatenate([point, np.zeros(3)])

        point = reference_pos + np.array([min_dist, 0.0, 0.5])
        point[2] = max(point[2], min_alt)
        return np.concatenate([point, np.zeros(3)])

    def generate_obstacles(
        self,
        episode_idx: int,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> list[dict]:
        """Generate obstacles. Trap episodes use predefined layouts."""
        if self._last_was_trap:
            trap = self._select_trap(episode_idx)
            obstacles = []
            for obs in trap["obstacles"]:
                pos = np.array(obs["position"], dtype=float)
                pos += start_pos
                obstacles.append({
                    "position": pos.tolist(),
                    "radius": obs["radius"],
                    "height": obs["height"],
                })
            return obstacles

        # Normal obstacle generation
        lvl = self.level
        if not lvl.get("obstacles", False):
            return []

        num_obs = lvl.get("num_obstacles", 3)
        ep_seed = self.get_episode_seed(episode_idx) + 999
        rng = np.random.RandomState(ep_seed)

        obstacles = []
        direction = goal_pos - start_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return []

        dir_norm = direction / dist

        if abs(dir_norm[2]) < 0.9:
            perp1 = np.cross(dir_norm, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(dir_norm, np.array([1, 0, 0]))
        perp1 /= np.linalg.norm(perp1) + 1e-12
        perp2 = np.cross(dir_norm, perp1)

        for i in range(num_obs):
            t = rng.uniform(0.2, 0.8)
            center = start_pos + dir_norm * dist * t

            offset1 = rng.uniform(-1.0, 1.0)
            offset2 = rng.uniform(-0.5, 0.5)
            pos = center + perp1 * offset1 + perp2 * offset2
            pos[2] = max(pos[2], 0.0)

            radius = rng.uniform(0.2, 0.5)
            height = rng.uniform(0.5, 2.0)

            obstacles.append({
                "position": pos.tolist(),
                "radius": round(float(radius), 3),
                "height": round(float(height), 3),
            })

        return obstacles

    def report_result(self, success: bool) -> Optional[str]:
        """Report episode result.

        Trap episodes don't count toward promotion/demotion
        to avoid penalizing hard scenarios.
        """
        if self._last_was_trap:
            # Track but don't affect curriculum progression
            return None

        self.recent_results.append(1 if success else 0)
        self.episodes_at_level += 1

        if not self.enabled:
            return None

        sr = self.success_rate

        if (
            sr >= self.promote_threshold
            and self.episodes_at_level >= self.min_episodes
            and self.current_level < len(self.levels) - 1
        ):
            old_name = self.level_name
            self.current_level += 1
            self.episodes_at_level = 0
            self.recent_results.clear()
            logger.info(
                "CURRICULUM PROMOTED: %s → %s (SR was %.0f%%)",
                old_name, self.level_name, sr * 100,
            )
            return "promoted"

        if (
            sr < self.demote_threshold
            and self.episodes_at_level >= self.min_episodes
            and self.current_level > 0
            and len(self.recent_results) >= 20
        ):
            old_name = self.level_name
            self.current_level -= 1
            self.episodes_at_level = 0
            self.recent_results.clear()
            logger.info(
                "CURRICULUM DEMOTED: %s → %s (SR was %.0f%%)",
                old_name, self.level_name, sr * 100,
            )
            return "demoted"

        return None

    def get_state(self) -> dict:
        return {
            "current_level": self.current_level,
            "level_name": self.level["name"],
            "episodes_at_level": self.episodes_at_level,
            "success_rate": round(self.success_rate, 3),
            "base_seed": self._base_seed,
        }
