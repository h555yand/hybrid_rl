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

"""PyBullet Drones environment adapter (VelocityAviary).

Uses gym-pybullet-drones VelocityAviary.
Action = [dir_x, dir_y, dir_z, speed_fraction]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    from gym_pybullet_drones.envs import VelocityAviary
    from gym_pybullet_drones.utils.enums import DroneModel
    HAS_PYBULLET_DRONES = True
except ImportError:
    HAS_PYBULLET_DRONES = False
    logger.warning(
        "gym-pybullet-drones not installed."
    )


class PyBulletDroneEnv:
    """Waypoint navigation env using VelocityAviary."""

    # Ray directions: 8 horizontal rays at 45° intervals
    _RAY_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(
        self,
        render: bool = False,
        flight_dome_size: float = 30.0,
        max_steps: int = 6000,
        goal_reach_distance: float = 0.3,
        initial_altitude: float = 1.0,
        ctrl_freq: int = 48,
        pyb_freq: int = 240,
        ray_count: int = 8,
        ray_max_dist: float = 3.0,
        seed: Optional[int] = None,
    ):
        if not HAS_PYBULLET_DRONES:
            raise ImportError("gym-pybullet-drones required.")

        self._render = render
        self._seed = seed
        self._dome_size = flight_dome_size
        self._max_steps = max_steps
        self._goal_reach_distance = goal_reach_distance
        self._initial_altitude = initial_altitude
        self._ray_count = ray_count
        self._ray_max_dist = ray_max_dist

        self._env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array(
                [[0.0, 0.0, initial_altitude]]
            ),
            gui=render,
            ctrl_freq=ctrl_freq,
            pyb_freq=pyb_freq,
        )

        self._obs: Optional[np.ndarray] = None
        self._info: Optional[dict] = None
        self._step_count: int = 0
        self._current_goal: Optional[np.ndarray] = None
        self._num_targets_reached: int = 0
        self._ground_z = 0.02

        # Visualization markers
        self._goal_marker_id = None
        self._goal_line_id = None
        self._goal_text_id = None
        self._cam_target = np.array([0.0, 0.0, initial_altitude])
        self._cam_smoothing = 0.1

        # Obstacles
        self._obstacle_ids: List[int] = []

        # Precompute ray directions (world frame, rotated per step)
        self._ray_angles_rad = np.deg2rad(self._RAY_ANGLES_DEG[:ray_count])
        self._raycast_interval = 4  # raycast каждые N шагов
        self._cached_ray_data = None

    # ═══════════════════════════════════════════════════
    # Observation parsing
    # ═══════════════════════════════════════════════════

    def _get_pos(self) -> np.ndarray:
        return self._obs[0, 0:3].copy()

    def _get_quat_xyzw(self) -> np.ndarray:
        return self._obs[0, 3:7].copy()

    def _get_vel_world(self) -> np.ndarray:
        return self._obs[0, 7:10].copy()

    def _get_ang_vel(self) -> np.ndarray:
        return self._obs[0, 10:13].copy()

    def _get_rotation(self) -> Rot:
        q = self._get_quat_xyzw()
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0])
        return Rot.from_quat(q)

    def _world_to_body(self, vec: np.ndarray) -> np.ndarray:
        return self._get_rotation().inv().apply(vec)

    # ═══════════════════════════════════════════════════
    # Raycasting
    # ═══════════════════════════════════════════════════

    def raycast_obstacles(self) -> dict:
        """Cast rays from drone position to detect obstacles.

        Returns dict with:
            ray_distances: [N] distances (clipped to ray_max_dist)
            ray_hits: [N] bool, True if hit something
            ray_directions_world: [N, 3] unit vectors
            nearest_obstacle_dist: float
            nearest_obstacle_dir_world: [3] direction to nearest
            path_blocked: bool (ray toward goal is blocked)
        """
        if self._obs is None:
            return self._empty_ray_data()

        pos = self._get_pos()
        rot = self._get_rotation()
        client = self._env.CLIENT
        drone_id = self._env.DRONE_IDS[0]
        max_dist = self._ray_max_dist

        # Get drone's forward direction (yaw only)
        euler = rot.as_euler("xyz")
        yaw = euler[2]

        ray_distances = np.full(self._ray_count, max_dist)
        ray_hits = np.zeros(self._ray_count, dtype=bool)
        ray_dirs_world = np.zeros((self._ray_count, 3))

        ray_from_list = []
        ray_to_list = []

        for i, angle in enumerate(self._ray_angles_rad):
            # Ray direction in world frame (horizontal plane)
            world_angle = yaw + angle
            direction = np.array([
                np.cos(world_angle),
                np.sin(world_angle),
                0.0,  # horizontal rays
            ])
            ray_dirs_world[i] = direction
            ray_from_list.append(pos.tolist())
            ray_to_list.append((pos + direction * max_dist).tolist())

        # Batch raycast (much faster than individual calls)
        results = p.rayTestBatch(
            ray_from_list, ray_to_list,
            physicsClientId=client,
        )

        for i, result in enumerate(results):
            hit_object_id = result[0]
            hit_fraction = result[2]

            # Ignore self-hits (drone body)
            if hit_object_id >= 0 and hit_object_id != drone_id:
                ray_distances[i] = hit_fraction * max_dist
                ray_hits[i] = True

        # Nearest obstacle
        if np.any(ray_hits):
            nearest_idx = np.argmin(ray_distances)
            nearest_dist = float(ray_distances[nearest_idx])
            nearest_dir_world = ray_dirs_world[nearest_idx]
        else:
            nearest_dist = max_dist
            nearest_dir_world = np.zeros(3)

        # Path blocked: cast ray toward goal
        path_blocked = False
        if self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            delta = goal_pos - pos
            goal_dist = np.linalg.norm(delta)
            if goal_dist > 0.1:
                goal_dir = delta / goal_dist
                ray_end = pos + goal_dir * min(goal_dist, max_dist)
                goal_ray = p.rayTest(
                    pos.tolist(), ray_end.tolist(),
                    physicsClientId=client,
                )
                if goal_ray[0][0] >= 0 and goal_ray[0][0] != drone_id:
                    path_blocked = True

        return {
            "ray_distances": ray_distances,
            "ray_hits": ray_hits,
            "ray_directions_world": ray_dirs_world,
            "nearest_obstacle_dist": nearest_dist,
            "nearest_obstacle_dir_world": nearest_dir_world,
            "path_blocked": path_blocked,
        }

    def _empty_ray_data(self) -> dict:
        max_dist = self._ray_max_dist
        return {
            "ray_distances": np.full(self._ray_count, max_dist),
            "ray_hits": np.zeros(self._ray_count, dtype=bool),
            "ray_directions_world": np.zeros((self._ray_count, 3)),
            "nearest_obstacle_dist": max_dist,
            "nearest_obstacle_dir_world": np.zeros(3),
            "path_blocked": False,
        }

    # ═══════════════════════════════════════════════════
    # Core API
    # ═══════════════════════════════════════════════════

    def reset(self, **kwargs) -> dict:
        if self._render:
            self._clear_markers()
        self._clear_obstacles()

        self._obs, self._info = self._env.reset()
        self._step_count = 0
        self._current_goal = None
        self._num_targets_reached = 0

        if self._render:
            self._update_camera()

        return self.get_sensor_data()

    def get_pose(self) -> np.ndarray:
        if self._obs is None:
            return np.zeros(6)
        pos = self._get_pos()
        euler = self._get_rotation().as_euler("xyz", degrees=False)
        return np.concatenate([pos, euler])

    def get_sensor_data(self) -> dict:
        if self._obs is None:
            return self._empty_sensor_data()

        pos = self._get_pos()
        vel_world = self._get_vel_world()
        ang_vel = self._get_ang_vel()
        quat = self._get_quat_xyzw()
        rot = self._get_rotation()

        # Goal-relative
        if self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            delta_world = goal_pos - pos
        else:
            delta_world = np.zeros(3)

        distance = float(np.linalg.norm(delta_world))
        delta_body = self._world_to_body(delta_world)
        vel_body = self._world_to_body(vel_world)

        if distance > 1e-8:
            goal_dir_body = delta_body / np.linalg.norm(delta_body)
        else:
            goal_dir_body = np.zeros(3)

        speed = float(np.linalg.norm(vel_world))

        if speed > 1e-4 and distance > 1e-4:
            vel_alignment = float(np.dot(
                vel_world / speed, delta_world / distance,
            ))
        else:
            vel_alignment = 0.0

        altitude = float(pos[2])

        # >>> Raycast ТОЛЬКО если есть препятствия <<<
        if self._obstacle_ids:
            if (self._step_count % self._raycast_interval == 0
                or self._cached_ray_data is None):
                self._cached_ray_data = self.raycast_obstacles()
        
            ray_data = self._cached_ray_data
       
            nearest_obs_dir_body = self._world_to_body(
                ray_data["nearest_obstacle_dir_world"]
            )
            ray_normalized = ray_data["ray_distances"] / self._ray_max_dist
            path_blocked = ray_data["path_blocked"]
            nearest_obs_dist = ray_data["nearest_obstacle_dist"]
            collision = (
                altitude < self._ground_z
                or self.check_obstacle_collision()
            )
        else:
            # Нет препятствий — пропускаем raycast
            nearest_obs_dir_body = np.zeros(3)
            ray_normalized = np.ones(self._ray_count)
            path_blocked = False
            nearest_obs_dist = self._ray_max_dist
            collision = altitude < self._ground_z

        out_of_bounds = np.linalg.norm(pos) > self._dome_size
        goal_reached = (
            self._current_goal is not None
            and distance < self._goal_reach_distance
        )
        if goal_reached:
            self._num_targets_reached += 1

        return {
            "goal_direction_body": goal_dir_body.tolist(),
            "goal_delta_body": delta_body.tolist(),
            "goal_delta_world": delta_world.tolist(),
            "distance_to_goal": distance,
            "angular_velocity": ang_vel.tolist(),
            "velocity_body": vel_body.tolist(),
            "velocity_world": vel_world.tolist(),
            "speed": speed,
            "vel_alignment": vel_alignment,
            "quaternion_xyzw": quat.tolist(),
            "position": pos.tolist(),
            "altitude": altitude,
            "path_blocked": path_blocked,
            "nearest_obstacle_dist": nearest_obs_dist,
            "nearest_obstacle_dir_body": nearest_obs_dir_body.tolist(),
            "obstacle_rays": ray_normalized.tolist(),
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "env_complete": goal_reached,
            "num_targets_reached": self._num_targets_reached,
        }

    def _empty_sensor_data(self) -> dict:
        n = self._ray_count
        return {
            "goal_direction_body": [0.0, 0.0, 0.0],
            "goal_delta_body": [0.0, 0.0, 0.0],
            "goal_delta_world": [0.0, 0.0, 0.0],
            "distance_to_goal": 0.0,
            "angular_velocity": [0.0, 0.0, 0.0],
            "velocity_body": [0.0, 0.0, 0.0],
            "velocity_world": [0.0, 0.0, 0.0],
            "speed": 0.0,
            "vel_alignment": 0.0,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "position": [0.0, 0.0, 0.0],
            "altitude": 0.0,
            "path_blocked": False,
            "nearest_obstacle_dist": self._ray_max_dist,
            "nearest_obstacle_dir_body": [0.0, 0.0, 0.0],
            "obstacle_rays": [1.0] * n,
            "collision": False,
            "out_of_bounds": False,
            "env_complete": False,
            "num_targets_reached": 0,
        }

    # ═══════════════════════════════════════════════════
    # Goal management
    # ═══════════════════════════════════════════════════

    def set_goal(self, goal_pose: np.ndarray) -> None:
        self._current_goal = np.array(goal_pose, dtype=float)
        if self._render:
            self._draw_goal_marker(self._current_goal[:3])

    def get_random_goal(self, **kwargs) -> np.ndarray:
        reference_pos = kwargs.get("reference_pos", None)
        if reference_pos is None:
            if self._obs is not None:
                reference_pos = self._get_pos()
            else:
                reference_pos = np.array(
                    [0.0, 0.0, self._initial_altitude]
                )

        min_dist = kwargs.get("min_dist", 2.0)
        max_dist = kwargs.get("max_dist", 6.0)
        min_alt = kwargs.get("min_altitude", 0.3)
        max_alt = kwargs.get("max_altitude", 4.0)
        dome_margin = self._dome_size * 0.8

        for _ in range(100):
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction) + 1e-12
            dist = np.random.uniform(min_dist, max_dist)
            point = reference_pos + direction * dist
            point[2] = np.clip(point[2], min_alt, max_alt)
            if np.linalg.norm(point) < dome_margin:
                return np.concatenate([point, np.zeros(3)])

        point = reference_pos + np.array([2.0, 0.0, 0.5])
        point[2] = max(point[2], min_alt)
        return np.concatenate([point, np.zeros(3)])

    get_random_surface_point = get_random_goal

    # ═══════════════════════════════════════════════════
    # Obstacles
    # ═══════════════════════════════════════════════════

    def spawn_obstacles(self, obstacles: list) -> None:
        """Spawn cylinder obstacles in PyBullet."""
        self._clear_obstacles()
        if not obstacles:
            return

        self._obstacle_ids = []
        for obs in obstacles:
            pos = obs["position"]
            radius = obs["radius"]
            height = obs["height"]

            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=self._env.CLIENT,
            )
            vis_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=[0.6, 0.3, 0.1, 0.8],
                physicsClientId=self._env.CLIENT,
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[pos[0], pos[1], pos[2] + height / 2],
                physicsClientId=self._env.CLIENT,
            )
            self._obstacle_ids.append(body_id)

    def _clear_obstacles(self) -> None:
        for oid in self._obstacle_ids:
            try:
                p.removeBody(oid, physicsClientId=self._env.CLIENT)
            except Exception:
                pass
        self._obstacle_ids = []

    def check_obstacle_collision(self) -> bool:
        if not self._obstacle_ids:
            return False
        drone_id = self._env.DRONE_IDS[0]
        for oid in self._obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=drone_id, bodyB=oid,
                physicsClientId=self._env.CLIENT,
            )
            if contacts:
                return True
        return False

    # ═══════════════════════════════════════════════════
    # Action execution
    # ═══════════════════════════════════════════════════

    def step_discrete(self, action_idx: int, action_space) -> dict:
        continuous = action_space.to_continuous(action_idx)
        return self._step_action(continuous)

    def step_continuous(
        self, action_type: int, action_params: np.ndarray
    ) -> dict:
        action = np.array(action_params[:4], dtype=np.float64)
        action[:3] = np.clip(action[:3], -1.0, 1.0)
        action[3] = np.clip(action[3], 0.0, 1.0)
        return self._step_action(action)

    def _step_action(self, action: np.ndarray) -> dict:
        action = np.array(action, dtype=np.float64)
        action[:3] = np.clip(action[:3], -1.0, 1.0)
        action[3] = np.clip(action[3], 0.0, 1.0)

        action_2d = action.reshape(1, 4)
        self._obs, reward, term, trunc, self._info = (
            self._env.step(action_2d)
        )
        self._step_count += 1

        if self._render and self._current_goal is not None:
            # self._draw_goal_line()
            self._update_camera()

        return self.get_sensor_data()

    # ═══════════════════════════════════════════════════
    # Visualization
    # ═══════════════════════════════════════════════════

    def _draw_goal_marker(self, goal_pos: np.ndarray) -> None:
        client = self._env.CLIENT
        if self._goal_marker_id is not None:
            p.removeBody(self._goal_marker_id, physicsClientId=client)
        if self._goal_text_id is not None:
            p.removeUserDebugItem(self._goal_text_id, physicsClientId=client)

        visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.25,
            rgbaColor=[1.0, 0.0, 0.0, 0.6],
            physicsClientId=client,
        )
        self._goal_marker_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=visual,
            basePosition=goal_pos.tolist(),
            physicsClientId=client,
        )
        self._goal_text_id = p.addUserDebugText(
            text="GOAL",
            textPosition=(goal_pos + np.array([0, 0, 0.4])).tolist(),
            textColorRGB=[1.0, 0.2, 0.2], textSize=1.5,
            physicsClientId=client,
        )

    def _draw_goal_line(self) -> None:
        client = self._env.CLIENT
        drone_pos = self._get_pos()
        goal_pos = self._current_goal[:3]
        if self._goal_line_id is not None:
            p.removeUserDebugItem(self._goal_line_id, physicsClientId=client)
        self._goal_line_id = p.addUserDebugLine(
            lineFromXYZ=drone_pos.tolist(),
            lineToXYZ=goal_pos.tolist(),
            lineColorRGB=[0.0, 1.0, 0.0], lineWidth=2.0,
            physicsClientId=client,
        )

    def _update_camera(self) -> None:
        if not self._render or self._obs is None:
            return
        drone_pos = self._get_pos()
        if self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            target = (drone_pos + goal_pos) / 2.0
            dist_to_goal = np.linalg.norm(goal_pos - drone_pos)
            cam_distance = max(3.0, dist_to_goal * 1.2)
        else:
            target = drone_pos
            cam_distance = 4.0
        alpha = self._cam_smoothing
        self._cam_target = alpha * self._cam_target + (1 - alpha) * target
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_distance, cameraYaw=45, cameraPitch=-35,
            cameraTargetPosition=self._cam_target.tolist(),
            physicsClientId=self._env.CLIENT,
        )

    def _clear_markers(self) -> None:
        client = self._env.CLIENT
        if self._goal_marker_id is not None:
            try:
                p.removeBody(self._goal_marker_id, physicsClientId=client)
            except Exception:
                pass
            self._goal_marker_id = None
        for attr in ('_goal_line_id', '_goal_text_id'):
            item_id = getattr(self, attr, None)
            if item_id is not None:
                try:
                    p.removeUserDebugItem(item_id, physicsClientId=client)
                except Exception:
                    pass
                setattr(self, attr, None)

    # ═══════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════

    @property
    def supports_continuous(self) -> bool:
        return True

    @property
    def supports_offline_retrain(self) -> bool:
        return True

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def ctrl_freq(self) -> int:
        return self._env.CTRL_FREQ

    @property
    def speed_limit(self) -> float:
        return self._env.SPEED_LIMIT

    # ═══════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
