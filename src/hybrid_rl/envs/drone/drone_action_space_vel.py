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

"""Drone action space for VelocityAviary.

Action = [dir_x, dir_y, dir_z, speed_fraction]
  - dir[0:3]: direction vector (auto-normalized by env)
  - speed_fraction: 0..1, multiplied by SPEED_LIMIT (0.25 m/s)
  - [0, 0, 0, 0] = hover

Axis mapping (world frame):
    dir_x+ → world X+
    dir_y+ → world Y+
    dir_z+ → world Z+ (up)
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class DroneActionInfo:
    index: int
    name: str
    dir_x: float
    dir_y: float
    dir_z: float
    speed: float


class DroneActionSpaceVel:
    """Discrete action space for VelocityAviary drone."""

    IDX_FORWARD = 0
    IDX_BACKWARD = 1
    IDX_LEFT = 2
    IDX_RIGHT = 3
    IDX_UP = 4
    IDX_DOWN = 5
    IDX_HOVER = 6
    IDX_FORWARD_UP = 7
    IDX_FORWARD_DOWN = 8
    IDX_FORWARD_LEFT = 9
    IDX_FORWARD_RIGHT = 10
    IDX_BACKWARD_LEFT = 11
    IDX_BACKWARD_RIGHT = 12

    NUM_ACTIONS = 13

    def __init__(
        self,
        cruise_speed: float = 1.0,
        slow_speed: float = 0.4,
    ):
        """
        Args:
            cruise_speed: speed_fraction for normal movement (0..1)
            slow_speed: speed_fraction for combined/careful moves
        """
        self.cruise_speed = cruise_speed
        self.slow_speed = slow_speed

        s = cruise_speed
        sl = slow_speed
        # Diagonal directions (normalized by env, but we keep unit-ish)
        d = 0.707  # 1/sqrt(2)

        self._actions: Dict[int, DroneActionInfo] = {
            self.IDX_FORWARD: DroneActionInfo(
                0, "forward", 1.0, 0.0, 0.0, s,
            ),
            self.IDX_BACKWARD: DroneActionInfo(
                1, "backward", -1.0, 0.0, 0.0, s,
            ),
            self.IDX_LEFT: DroneActionInfo(
                2, "left", 0.0, -1.0, 0.0, s,
            ),
            self.IDX_RIGHT: DroneActionInfo(
                3, "right", 0.0, 1.0, 0.0, s,
            ),
            self.IDX_UP: DroneActionInfo(
                4, "up", 0.0, 0.0, 1.0, s,
            ),
            self.IDX_DOWN: DroneActionInfo(
                5, "down", 0.0, 0.0, -1.0, s,
            ),
            self.IDX_HOVER: DroneActionInfo(
                6, "hover", 0.0, 0.0, 0.0, 0.0,
            ),
            self.IDX_FORWARD_UP: DroneActionInfo(
                7, "forward_up", d, 0.0, d, s,
            ),
            self.IDX_FORWARD_DOWN: DroneActionInfo(
                8, "forward_down", d, 0.0, -d, s,
            ),
            self.IDX_FORWARD_LEFT: DroneActionInfo(
                9, "forward_left", d, -d, 0.0, s,
            ),
            self.IDX_FORWARD_RIGHT: DroneActionInfo(
                10, "forward_right", d, d, 0.0, s,
            ),
            self.IDX_BACKWARD_LEFT: DroneActionInfo(
                11, "backward_left", -d, -d, 0.0, sl,
            ),
            self.IDX_BACKWARD_RIGHT: DroneActionInfo(
                12, "backward_right", -d, d, 0.0, sl,
            ),
        }

    def get_info(self, action_index: int) -> DroneActionInfo:
        return self._actions[action_index]

    def to_continuous(self, action_index: int) -> np.ndarray:
        """Convert discrete action to [dir_x, dir_y, dir_z, speed_frac]."""
        info = self._actions[action_index]
        return np.array(
            [info.dir_x, info.dir_y, info.dir_z, info.speed],
            dtype=np.float64,
        )
