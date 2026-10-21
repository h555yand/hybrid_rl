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

from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class LightweightEnv:
    """A lightweight simulator for training/testing RL without Habitat.
    Uses trimesh for:
    - Mesh loading (STL, OBJ, PLY)
    - Ray casting (depth, normal)
    - Collision detection
    Does not use:
    - Habitat
    - GPU
    """

    def __init__(self, mesh_path: str, seed=None):
        self.mesh = trimesh.load(mesh_path)
        # all numbers are mm
        # z+ is out of the page, x+ is "right", and y+ is "up"
        self.agent_pos = np.zeros(3)
        # euler angles
        self.agent_rot = np.zeros(3)
        # Pitch - turn up/down - rotation around the X axis
        # Yaw - turn left/right - rotation around the Y axis
        # Roll - turn head to shoulder - rotation around the Z axis
        # self.agent_rot = [Pitch, Yaw, Roll]
        # self.agent_rot = [X-angle, Y-angle, Z-angle]
        # Note: seed is set globally in train() => np.random.seed()
        # trimesh.sample() uses global np.random
        self._passed_through = False
        self._detach_had_collision = False
        self._compute_up_direction()
        self._mesh_centroid = np.array(self.mesh.centroid, dtype=float)
        self._wrong_side_outward = None
        self._edge_traversed = None
        self._last_snap_debug = ""
        self._wall_side_debug = ""

    @staticmethod
    def _normalize_euler(angles):
        return (np.array(angles, dtype=float) + 180.0) % 360.0 - 180.0

    def reset(self, position=None, rotation=None):
        """Place the agent in the starting position."""
        self._passed_through = False
        self._detach_had_collision = False
        self._wrong_side_outward = None
        if position is not None:
            self.agent_pos = np.array(position, dtype=float)
        else:
            # Random point on surface (using controlled RNG)
            points, face_ids = self.mesh.sample(1, return_index=True)

            normal = self.mesh.face_normals[face_ids[0]]
            self.agent_pos = points[0] + normal * 2.0  # 2 mm from the surface outwards

            # If rotation is not specified, look towards the surface
            if rotation is None:
                self.agent_rot = self._look_at_direction(-normal)

        if rotation is not None:
            self.agent_rot = np.array(rotation, dtype=float)
        elif position is not None:
            # If the position was set manually, but the rotation was not, you can leave it at 0
            self.agent_rot = np.zeros(3)

        self.agent_rot = self._normalize_euler(self.agent_rot)
        return self.get_sensor_data()

    def set_goal(self, goal_pose):
        self._current_goal = goal_pose

    def step(self, action_index, action_space):
        """Perform an action, return new sensor data.

        Args:
            action_index: action index (0-23)
            action_space: ActionSpace

        Returns:
            sensor_data: point_normal, on_object, depth
        """
        self._detach_had_collision = False
        self._edge_traversed = False

        action_info = action_space.get_info(action_index)

        if action_info.name in ("look_up", "look_down", "look_up_big", "look_down_big"):
            rot_before = R.from_euler("xyz", self.agent_rot, degrees=True)
            forward_before = rot_before.apply([0, 0, -1])

        if action_info.name == "move_tangentially":
            self._move_tangentially(
                action_info.direction_degrees,
                action_space.surface_step,
            )
        elif action_info.name == "free_forward":
            self._move_forward(action_space.free_step)
        elif action_info.name == "free_backward":
            self._move_forward(-action_space.free_step_backward)
        elif action_info.name == "look_up":
            self.agent_rot[0] += action_space.rotation_step
        elif action_info.name == "look_down":
            self.agent_rot[0] -= action_space.rotation_step
        elif action_info.name == "look_up_big":
            self.agent_rot[0] += action_space.rotation_step_big
        elif action_info.name == "look_down_big":
            self.agent_rot[0] -= action_space.rotation_step_big
        elif action_info.name == "rotate_sensor_+":
            self.agent_rot[2] += action_space.rotation_step
        elif action_info.name == "rotate_sensor_-":
            self.agent_rot[2] -= action_space.rotation_step
        elif action_info.name == "turn_left":
            self.agent_rot[1] += action_space.rotation_step
        elif action_info.name == "turn_right":
            self.agent_rot[1] -= action_space.rotation_step
        elif action_info.name == "turn_left_big":
            self.agent_rot[1] += action_space.rotation_step_big
        elif action_info.name == "turn_right_big":
            self.agent_rot[1] -= action_space.rotation_step_big
        elif action_info.name == "orient_horizontal":
            self._orient_horizontal(
                rotation_degrees=action_info.rotation_degrees,
                forward_distance=action_info.forward_distance,
                left_distance=action_info.left_distance,
            )
        elif action_info.name == "orient_vertical":
            self._orient_vertical(
                rotation_degrees=action_info.rotation_degrees,
                forward_distance=action_info.forward_distance,
                down_distance=action_info.down_distance,
            )
        elif action_info.name == "detach":
            if hasattr(self, "_current_goal") and self._current_goal is not None:
                self._detach_simple(
                    goal_pose=self._current_goal,
                    detach_distance=action_space.free_step * 3,
                )
        elif action_info.name == "free_forward_small":
            self._move_forward(action_space.free_step_small)

        if action_info.name in ("look_up", "look_down", "look_up_big", "look_down_big"):
            rot_after = R.from_euler("xyz", self.agent_rot, degrees=True)
            forward_after = rot_after.apply([0, 0, -1])
            height_axis = getattr(self, "height_axis", 2)
            logger.debug(
                f"LOOK_DEBUG: action={action_info.name}, "
                f"forward_z_before={forward_before[height_axis]:.3f}, "
                f"forward_z_after={forward_after[height_axis]:.3f}, "
                f"agent_rot={self.agent_rot}"
            )
        self.agent_rot = self._normalize_euler(self.agent_rot)
        return self.get_sensor_data()

    def get_sensor_data(self):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        ray_direction = rot.apply([0, 0, -1])
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[self.agent_pos],
            ray_directions=[ray_direction],
        )

        if len(locations) > 0:
            distances = np.linalg.norm(locations - self.agent_pos, axis=1)
            nearest_idx = np.argmin(distances)

            depth = distances[nearest_idx]
            face_idx = index_tri[nearest_idx]
            point_normal = self.mesh.face_normals[face_idx].tolist()
            on_object = bool(depth < 3.0)
        else:
            depth = 100.0
            point_normal = None
            on_object = False

        goal_normal = None
        if hasattr(self, "_current_goal") and self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            _, _, gf_id = self.mesh.nearest.on_surface([goal_pos])
            goal_normal = self.mesh.face_normals[gf_id[0]].tolist()

        # ═══ path_blocked (ray cast) ═══
        path_blocked = False
        if hasattr(self, "_current_goal") and self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            direction = goal_pos - self.agent_pos
            dist_to_goal = np.linalg.norm(direction)

            if dist_to_goal > 1e-8:
                direction_norm = direction / dist_to_goal
                bl_locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=[self.agent_pos],
                    ray_directions=[direction_norm],
                )
                if len(bl_locations) > 0:
                    hit_distances = np.linalg.norm(
                        bl_locations - self.agent_pos, axis=1
                    )
                    # Blocked if any intersection is closer than goal
                    # margin 2.0mm to avoid counting the goal surface itself
                    path_blocked = bool(np.min(hit_distances) < dist_to_goal - 2.0)

        # ═══ Curvature ═══
        curvature_data = self._estimate_curvature()

        # ═══ same_side ═══
        same_side = True
        if hasattr(self, "_current_goal") and self._current_goal is not None:
            on_object = bool(depth < 3.0)
            
            if on_object:
                # On surface: use nearest.on_surface (reliable)
                same_side = _is_reachable_by_surface(
                    self, self.agent_pos, self._current_goal[:3]
                )
            else:
                # In air: use ray-based inside/outside test
                same_side = self._same_side_air(
                    self.agent_pos, self._current_goal[:3],
                    point_normal,
                )

        return {
            "point_normal": point_normal,
            "k1": curvature_data["k1"],
            "k2": curvature_data["k2"],
            "principal_curvatures": [curvature_data["k1"], curvature_data["k2"]],
            "on_object": on_object,
            "depth": depth,
            "passed_through": getattr(self, "_passed_through", False),
            "goal_normal": goal_normal,
            "detach_had_collision": getattr(self, "_detach_had_collision", False),
            "detach_sub_steps": getattr(self, "_last_detach_sub_steps", 1),
            "path_blocked": path_blocked,
            "up_direction": self.up_direction.tolist(),
            "object_center": self.mesh.centroid.tolist(),
            "same_side": same_side,
            "object_extents": (self.mesh.bounds[1] - self.mesh.bounds[0]).tolist(),
            "edge_traversed": getattr(self, "_edge_traversed", False),
            "open_edge_height": self.open_edge_height,
            "snap_debug": getattr(self, "_last_snap_debug", ""),
        }

    def _same_side_air(self, agent_pos, goal_pos, agent_normal):
        """Determine same_side when agent is in air.
        
        Agent side: horizontal ray cast toward center (position-based).
        Goal side: normal-based (reliable for on-surface points).
        """
        center = np.array(self.mesh.centroid, dtype=float)
        height_axis = self.height_axis
        up = self.up_direction
        up_sign = self.up_sign

        # ═══ Agent: inside or outside? (position-based) ═══
        from_center = agent_pos - center
        from_center_horiz = from_center.copy()
        from_center_horiz[height_axis] = 0.0
        horiz_dist = float(np.linalg.norm(from_center_horiz))

        # Above rim = outside
        agent_height = agent_pos[height_axis]
        rim_height = self.open_edge_height
        margin = 5.0  # 5mm above rim before switching to "above" mode
        above_rim = (agent_height - rim_height) * up_sign > margin

        bbox_min = self.mesh.bounds[0]
        bbox_max = self.mesh.bounds[1]
        if up_sign > 0:
            bottom_height = bbox_min[height_axis]
        else:
            bottom_height = bbox_max[height_axis]
        below_bottom = (bottom_height - agent_height) * up_sign > 5.0

        if above_rim:
            agent_inside = False
        elif below_bottom:
            agent_inside = False
        elif horiz_dist < 1e-8:
            agent_inside = True  # at center, below rim
        else:
            direction = -from_center_horiz / horiz_dist
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=[agent_pos],
                ray_directions=[direction],
            )
            if len(locations) > 0:
                hit_distances = np.linalg.norm(locations - agent_pos, axis=1)
                hits_before_center = int(np.sum(hit_distances < horiz_dist + 5.0))
                agent_inside = hits_before_center == 0
            else:
                agent_inside = True  # no surface between agent and center

        agent_outward = not agent_inside

        # ═══ Goal side (normal-based, unchanged) ═══
        _, _, goal_face = self.mesh.nearest.on_surface([goal_pos])
        goal_normal = self.mesh.face_normals[goal_face[0]]

        goal_n_horiz = goal_normal.copy()
        goal_n_horiz[height_axis] = 0.0
        goal_from_center = goal_pos - center
        goal_from_center[height_axis] = 0.0

        if np.linalg.norm(goal_n_horiz) >= 0.3:
            goal_outward = float(np.dot(goal_n_horiz, goal_from_center)) > 0
        else:
            goal_outward = float(np.dot(goal_normal, up)) < 0

        return agent_outward == goal_outward

    def get_pose(self):
        """The agent's current pose."""
        return np.concatenate([self.agent_pos, self.agent_rot])

    def get_random_surface_point(
        self,
        reference_pos: np.ndarray = None,
        min_dist: float = None,
        max_dist: float = None,
        max_attempts: int = 200,
        mesh_sample=False,
        same_cube_side=False
    ) -> np.ndarray:
        """A random point on the surface with an optional distance limit.

        Args:
        reference_pos: The reference point (the agent's position). If None, no distance filter is applied.
        min_dist: Minimum distance from reference_pos (mm, inclusive).
        max_dist: Maximum distance from reference_pos (mm, inclusive).
        max_attempts: Limit of rejection sampling attempts. If exhausted,
        returns any point without filtering (fallback).

        Returns:
        pose [6D]: [x, y, z, rx, ry, rz] points on the surface.
        """
        use_filter = (
            reference_pos is not None
            and (min_dist is not None or max_dist is not None)
        )

        if mesh_sample:
            points, face_ids = self.mesh.sample(max_attempts, return_index=True)
            for i in range(max_attempts):
                normal = self.mesh.face_normals[face_ids[i]]
                position = points[i] + normal * 2.0

                if use_filter:
                    dist = float(np.linalg.norm(position - reference_pos))
                    if min_dist is not None and dist < min_dist:
                        continue
                    if max_dist is not None and dist > max_dist:
                        continue
                    if same_cube_side == True:
                        same_cube_side_check = is_on_same_cube_side(position, reference_pos)
                        if same_cube_side_check == False:
                            continue

                rotation = self._look_at_direction(-normal)
                return np.concatenate([position, rotation])
        else:
            for _ in range(max_attempts):
                points, face_ids = self.mesh.sample(1, return_index=True)
                normal = self.mesh.face_normals[face_ids[0]]
                position = points[0] + normal * 2.0

                if use_filter:
                    dist = float(np.linalg.norm(position - reference_pos))
                    if min_dist is not None and dist < min_dist:
                        continue
                    if max_dist is not None and dist > max_dist:
                        continue

                rotation = self._look_at_direction(-normal)
                return np.concatenate([position, rotation])

        # Fallback: no point was within the range for max_attempts.
        # Return a random point.
        points, face_ids = self.mesh.sample(1, return_index=True)
        normal = self.mesh.face_normals[face_ids[0]]
        position = points[0] + normal * 2.0
        rotation = self._look_at_direction(-normal)
        return np.concatenate([position, rotation])

    def _estimate_curvature(self):
        """Estimate principal curvatures k1, k2 at the point the agent is looking at.

        Uses discrete mean and Gaussian curvature to recover principal curvatures:
            k1 = H + sqrt(H² - K)
            k2 = H - sqrt(H² - K)
        where H = mean curvature, K = Gaussian curvature.

        Convention: |k1| >= |k2| (k1 is the curvature of maximum bending).

        Returns:
            dict with 'k1' and 'k2' (floats), or {'k1': 0.0, 'k2': 0.0}
            if no surface is visible.
        """
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        ray_direction = rot.apply([0, 0, -1])
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[self.agent_pos],
            ray_directions=[ray_direction],
        )
        if len(locations) == 0:
            return {"k1": 0.0, "k2": 0.0}

        distances = np.linalg.norm(locations - self.agent_pos, axis=1)
        nearest_idx = np.argmin(distances)
        face_idx = index_tri[nearest_idx]
        vertex_indices = self.mesh.faces[face_idx]

        if not hasattr(self, "_vertex_mean_curvature"):
            _curv_radius = 5.0
            _curv_area = np.pi * _curv_radius ** 2
            self._vertex_mean_curvature = (
                trimesh.curvature.discrete_mean_curvature_measure(
                    self.mesh, self.mesh.vertices, radius=_curv_radius
                ) / _curv_area
            )
            self._vertex_gaussian_curvature = (
                trimesh.curvature.discrete_gaussian_curvature_measure(
                    self.mesh, self.mesh.vertices, radius=_curv_radius
                ) / _curv_area
            )

        H = float(np.mean(self._vertex_mean_curvature[vertex_indices]))
        K = float(np.mean(self._vertex_gaussian_curvature[vertex_indices]))

        # Principal curvatures from H and K:
        # k1, k2 are roots of: t² - 2Ht + K = 0
        # k1 = H + sqrt(H² - K), k2 = H - sqrt(H² - K)
        discriminant = H * H - K
        if discriminant < 0:
            # Numerical noise — clamp to zero
            discriminant = 0.0

        sqrt_disc = np.sqrt(discriminant)
        k1 = H + sqrt_disc
        k2 = H - sqrt_disc

        # Convention: |k1| >= |k2|
        if abs(k1) < abs(k2):
            k1, k2 = k2, k1

        return {"k1": float(k1), "k2": float(k2)}

    def _estimate_curvature_mean_gaus(self):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        ray_direction = rot.apply([0, 0, -1])
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[self.agent_pos],
            ray_directions=[ray_direction],
        )
        if len(locations) == 0:
            return [0.0, 0.0]

        distances = np.linalg.norm(locations - self.agent_pos, axis=1)
        nearest_idx = np.argmin(distances)
        face_idx = index_tri[nearest_idx]
        vertex_indices = self.mesh.faces[face_idx]

        if not hasattr(self, "_vertex_mean_curvature"):
            _curv_radius = 5.0
            _curv_area = np.pi * _curv_radius ** 2
            self._vertex_mean_curvature = (
                trimesh.curvature.discrete_mean_curvature_measure(
                    self.mesh, self.mesh.vertices, radius=_curv_radius
                ) / _curv_area
            )
            self._vertex_gaussian_curvature = (
                trimesh.curvature.discrete_gaussian_curvature_measure(
                    self.mesh, self.mesh.vertices, radius=_curv_radius
                ) / _curv_area
            )

        mean_curv = float(np.mean(self._vertex_mean_curvature[vertex_indices]))
        gauss_curv = float(np.mean(self._vertex_gaussian_curvature[vertex_indices]))

        return [mean_curv, gauss_curv]

    def _look_at_direction(self, direction):
        """Return euler angles [rx, ry, rz] IN DEGREES such that:
        R.from_euler("xyz", euler, degrees=True).apply([0, 0, -1]) ≈ direction
        Convention: forward is -Z when euler = [0,0,0].
        """
        d = np.asarray(direction, dtype=float)
        d /= (np.linalg.norm(d) + 1e-12)

        forward = np.array([0.0, 0.0, -1.0])

        # align_vectors: find R such that R @ forward ≈ d
        # The first argument is the target vectors, the second is the source vectors
        rot, _ = R.align_vectors([d], [forward])

        return rot.as_euler("xyz", degrees=True)

    def _look_at_direction_complex(self, direction):
        """Calculates euler angles for looking in a direction.
        Builds a full rotation matrix so that -Z (forward) looks in
        the direction, and Y (up) remains as vertical as possible.
        Resistant to gimbal lock (vertical normals).
        """
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        # forward = -Z, so we need -Z → direction, i.e. Z → -direction
        forward = -direction
        # We select the up vector, avoiding parallelism with the forward vector.
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            # The normal is almost vertical - we take X as a fallback up
            world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(world_up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)
        # Rotation matrix: columns = right, up, forward (= -Z direction)
        rot_matrix = np.column_stack([right, up, forward])
        return R.from_matrix(rot_matrix).as_euler("xyz")

    def _move_tangentially(self, direction_degrees, step_size, snap_to_surface=True):
        """Move agent along surface tangent with snap."""
        
        # 1. Compute movement
        old_pos, old_normal, world_dir = self._compute_tangent_move(
            direction_degrees, step_size
        )
        if old_normal is None:
            return  # no surface, raw move already applied
        
        if not snap_to_surface:
            return

        # Get curvature at current point for _is_horizontal check
        curvature = self._estimate_curvature()
        curv_k1 = curvature.get("k1", 0.0)
        
        # 2. Find snap target
        snap = self._find_snap_target(
            old_pos, old_normal, world_dir, step_size,
            curvature_k1=curv_k1,
        )

        if snap is None:
            self.agent_pos = old_pos  # too far from surface
            return
        
        # 3. Validate: through-wall check
        snap = self._validate_snap(old_pos, old_normal, world_dir, snap, step_size)
        
        # 4. Apply
        if snap["valid"]:
            self.agent_pos = snap["position"]
            self.agent_rot = self._look_at_direction(-snap["normal"])
            self._edge_traversed = snap.get("edge_traversed", False)
            
            # Snap debug info
            center = np.array(self.mesh.centroid, dtype=float)
            from_c = snap["position"] - center
            from_c[self.height_axis] = 0
            dist_c = float(np.linalg.norm(from_c))
            move_d = float(np.linalg.norm(snap["position"] - old_pos))
            ws = getattr(self, '_wall_side_debug', '')
            ws_str = f"|ws={ws}" if ws else ""
            
            self._last_snap_debug = (
                f"ok|dc={dist_c:.1f}"
                f"|md={move_d:.1f}"
                f"|edge={int(snap.get('edge_traversed', False))}"
                f"{ws_str}"
            )
        else:
            self.agent_pos = old_pos
            self._last_snap_debug = f"blocked|ws={getattr(self, '_wall_side_debug', '')}"

    def _compute_tangent_move(self, direction_degrees, step_size):
        """Compute tangent direction and move agent.
        
        Returns:
            (old_pos, old_normal, world_dir) or (old_pos, None, None)
            if no surface normal available.
        """
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        sensor_data = self.get_sensor_data()
        
        if sensor_data["point_normal"] is None:
            # No surface — raw move
            angle_rad = np.radians(direction_degrees)
            local_dir = np.array([
                np.sin(angle_rad), 0.0, -np.cos(angle_rad)
            ])
            local_dir /= np.linalg.norm(local_dir) + 1e-12
            old_pos = self.agent_pos.copy()
            self.agent_pos += rot.apply(local_dir) * step_size
            return old_pos, None, None
        
        n = np.array(sensor_data["point_normal"], dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        
        # Build tangent basis
        world_dir = self._build_tangent_direction(rot, n, direction_degrees)
        
        old_pos = self.agent_pos.copy()
        self.agent_pos += world_dir * step_size
        
        return old_pos, n, world_dir

    def _build_tangent_direction(self, rot, normal, direction_degrees):
        """Build tangent direction from agent rotation and surface normal."""
        right = rot.apply([1.0, 0.0, 0.0])
        t1 = right - np.dot(right, normal) * normal
        t1_len = np.linalg.norm(t1)
        
        if t1_len < 1e-8:
            up = rot.apply([0.0, 1.0, 0.0])
            t1 = up - np.dot(up, normal) * normal
            t1_len = np.linalg.norm(t1)
        
        if t1_len < 1e-8:
            tmp = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(tmp, normal)) > 0.9:
                tmp = np.array([0.0, 0.0, 1.0])
            t1 = np.cross(normal, tmp)
            t1_len = np.linalg.norm(t1)
        
        t1 /= t1_len + 1e-12
        t2 = np.cross(normal, t1)
        t2 /= np.linalg.norm(t2) + 1e-12
        
        a = np.radians(direction_degrees)
        d = np.cos(a) * t1 + np.sin(a) * t2
        return d / (np.linalg.norm(d) + 1e-12)

    def _find_snap_target(self, old_pos, old_normal, world_dir, step_size, curvature_k1=None):
        """Find the best surface point to snap to."""
        self._wall_side_debug = ""
        
        closest, dist, face_id = self.mesh.nearest.on_surface(
            [self.agent_pos]
        )
        
        if dist[0] > 15.0:
            return None
        
        hit_n = self.mesh.face_normals[face_id[0]].copy()
        hit_n /= np.linalg.norm(hit_n) + 1e-12
        
        is_from_horiz = self._is_horizontal(old_normal, curvature_k1)
        
        # Compute curvature at snap point for is_new_horiz
        snap_k1 = None
        snap_vertex_indices = self.mesh.faces[face_id[0]]
        if hasattr(self, "_vertex_mean_curvature"):
            snap_H = float(np.mean(
                self._vertex_mean_curvature[snap_vertex_indices]
            ))
            snap_K = float(np.mean(
                self._vertex_gaussian_curvature[snap_vertex_indices]
            ))
            disc = max(snap_H * snap_H - snap_K, 0.0)
            snap_k1 = float(snap_H + np.sqrt(disc))
        
        is_new_horiz = self._is_horizontal(hit_n, snap_k1)
        
        hit_n = self._align_normal(
            hit_n, old_normal, is_from_horiz
        )
        
        can_transition = self._can_transition(
            hit_n, old_normal, is_from_horiz, is_new_horiz
        )
        
        edge_traversed = False
        
        if (
            can_transition
            and not is_from_horiz
            and not is_new_horiz
        ):
            edge_result = self._probe_for_edge(
                old_normal, world_dir, closest[0], step_size
            )
            if edge_result is not None:
                closest = [edge_result["point"]]
                hit_n = edge_result["normal"]
                edge_traversed = True
        
        if (
            not can_transition
            and not is_from_horiz
            and is_new_horiz
            and dist[0] < step_size * 2.0
        ):
            vert = abs(float(
                np.dot(world_dir, self.up_direction)
            ))
            if vert > 0.2:
                can_transition = True
                edge_traversed = True

        # DEBUG: before can_transition check
        self._wall_side_debug = (
            f"CT|ct={int(can_transition)}"
            f"|fh={int(is_from_horiz)}"
            f"|nh={int(is_new_horiz)}"
            f"|dot={float(np.dot(hit_n, old_normal)):.3f}"
            f"|hn={[round(x,2) for x in hit_n.tolist()]}"
            f"|on={[round(x,2) for x in old_normal.tolist()]}"
        )

        if not can_transition:
            edge_result = self._try_intermediate_edge(
                old_pos, old_normal, world_dir, step_size,
                is_from_horiz
            )
            if edge_result is not None:
                return edge_result
            return {
                "position": old_pos,
                "normal": old_normal,
                "valid": False,
            }
        
        # ═══ Correct side for hollow objects ═══
        if is_from_horiz and not is_new_horiz:
            if (
                hasattr(self, '_current_goal')
                and self._current_goal is not None
            ):
                hit_n_h = hit_n.copy()
                hit_n_h[self.height_axis] = 0.0
                hn_len = np.linalg.norm(hit_n_h)

                if hn_len > 1e-8:
                    old_h = old_pos[self.height_axis]
                    bmin_h = self.mesh.bounds[0][self.height_axis]
                    bmax_h = self.mesh.bounds[1][self.height_axis]

                    from_rim = abs(old_h - bmax_h) < 5.0
                    from_bottom = abs(old_h - bmin_h) < 5.0

                    snap_radial = closest[0] - self._mesh_centroid
                    snap_radial[self.height_axis] = 0.0
                    sr_len = float(np.linalg.norm(snap_radial))

                    need_flip = False
                    nf_info = "none"
                    gr_len = 0.0

                    if from_bottom:
                        from_inner_bottom = (
                            old_normal[self.height_axis]
                            * self.up_sign > 0
                        )
                        if sr_len > 1e-8:
                            normal_is_outward = (
                                float(np.dot(
                                    hit_n_h / hn_len,
                                    snap_radial / sr_len,
                                )) > 0
                            )
                            if from_inner_bottom:
                                need_flip = normal_is_outward
                            else:
                                need_flip = not normal_is_outward
                        nf_info = (
                            f"bot|ib={int(from_inner_bottom)}"
                            f"|nio={int(normal_is_outward if sr_len > 1e-8 else -1)}"
                        )

                    elif from_rim:
                        agent_radial = old_pos - self._mesh_centroid
                        agent_radial[self.height_axis] = 0.0
                        ar_len = float(np.linalg.norm(agent_radial))
                        
                        if sr_len > 1e-8 and ar_len > 1e-8:
                            normal_is_outward = (
                                float(np.dot(
                                    hit_n_h / hn_len,
                                    snap_radial / sr_len,
                                )) > 0
                            )
                            
                            radial_dir = agent_radial / ar_len
                            
                            # ═══ Probe OUTWARD to find outer wall at rim level ═══
                            probe_out = old_pos.copy()
                            probe_out += radial_dir * 5.0
                            cl_out, _, _ = self.mesh.nearest.on_surface([probe_out])
                            out_radial = cl_out[0] - self._mesh_centroid
                            out_radial[self.height_axis] = 0.0
                            outer_radius = float(np.linalg.norm(out_radial))
                            
                            # ═══ Probe INWARD to find inner wall at rim level ═══
                            probe_in = old_pos.copy()
                            probe_in -= radial_dir * 5.0
                            cl_in, _, _ = self.mesh.nearest.on_surface([probe_in])
                            in_radial = cl_in[0] - self._mesh_centroid
                            in_radial[self.height_axis] = 0.0
                            inner_radius = float(np.linalg.norm(in_radial))
                            
                            # Wall midline at rim height
                            wall_midline = (outer_radius + inner_radius) / 2.0
                            
                            # Agent on inner edge if closer to center than midline
                            agent_on_inner_edge = ar_len < wall_midline
                            
                            # Goal analysis (for debug, not used in decision)
                            goal_radial = (
                                self._current_goal[:3] - self._mesh_centroid
                            )
                            goal_radial[self.height_axis] = 0.0
                            gr_len = float(np.linalg.norm(goal_radial))
                            
                            # ═══ KEY LOGIC ═══
                            # Agent crawled UP to rim from one side,
                            # now falls DOWN the OTHER side.
                            # inner edge → came from inside → wants OUTSIDE
                            # outer edge → came from outside → wants INSIDE
                            
                            if agent_on_inner_edge:
                                # Came from inside → want outer wall
                                need_flip = not normal_is_outward
                            else:
                                # Came from outside → want inner wall  
                                need_flip = normal_is_outward
                            
                            nf_info = (
                                f"rim|ar={ar_len:.1f}"
                                f"|sr={sr_len:.1f}"
                                f"|or={outer_radius:.1f}"
                                f"|ir={inner_radius:.1f}"
                                f"|mid={wall_midline:.1f}"
                                f"|inner={int(agent_on_inner_edge)}"
                                f"|nio={int(normal_is_outward)}"
                                f"|flip={int(need_flip)}"
                            )
                        else:
                            nf_info = (
                                f"rim|skip"
                                f"|sr={sr_len:.1f}"
                                f"|ar={ar_len:.1f}"
                            )

                    # Debug
                    gdc_str = f"|gdc={gr_len:.1f}" if from_rim else ""
                    self._wall_side_debug = (
                        f"nf={int(need_flip)}"
                        f"|{nf_info}"
                        f"|sdc={sr_len:.1f}"
                        f"{gdc_str}"
                    )

                    if need_flip:
                        opp_pos = (
                            closest[0] - hit_n * 5.0
                        )
                        cl_opp, d_opp, f_opp = (
                            self.mesh.nearest.on_surface(
                                [opp_pos]
                            )
                        )
                        n_opp = self.mesh.face_normals[
                            f_opp[0]
                        ].copy()
                        n_opp /= (
                            np.linalg.norm(n_opp)
                            + 1e-12
                        )

                        if not self._is_horizontal(
                            n_opp
                        ):
                            n_opp_h = n_opp.copy()
                            n_opp_h[
                                self.height_axis
                            ] = 0.0
                            nol = np.linalg.norm(
                                n_opp_h
                            )
                            if nol > 1e-8:
                                opp_r = (
                                    cl_opp[0]
                                    - self._mesh_centroid
                                )
                                opp_r[
                                    self.height_axis
                                ] = 0.0
                                opp_sr = float(
                                    np.linalg.norm(
                                        opp_r
                                    )
                                )
                                if opp_sr > 1e-8:
                                    opp_outward = (
                                        float(
                                            np.dot(
                                                n_opp_h
                                                / nol,
                                                opp_r
                                                / opp_sr,
                                            )
                                        )
                                        > 0
                                    )
                                    if from_bottom:
                                        if (
                                            from_inner_bottom
                                        ):
                                            ok = (
                                                not opp_outward
                                            )
                                        else:
                                            ok = (
                                                opp_outward
                                            )
                                    else:
                                        # from_rim: crossing
                                        # to other side
                                        if (
                                            agent_on_inner_edge
                                        ):
                                            ok = (
                                                opp_outward
                                            )
                                        else:
                                            ok = (
                                                not opp_outward
                                            )
                                    if ok:
                                        hit_n = n_opp
                                        closest = (
                                            cl_opp
                                        )
                else:
                    self._wall_side_debug = "hn0"
            else:
                self._wall_side_debug = "nogoal"


        # Recompute is_new_horiz after edge traversal
        # (edge traversal may have changed hit_n)
        is_new_horiz = self._is_horizontal(hit_n)
        # UNCONDITIONAL DEBUG
        self._wall_side_debug = (
            f"PRE|fh={int(is_from_horiz)}"
            f"|nh={int(is_new_horiz)}"
            f"|hn={[round(x,2) for x in hit_n.tolist()]}"
            f"|edge={int(edge_traversed)}"
        )

        # ═══ Side consistency: wall → horizontal ═══
        # If inner wall → outer bottom (crossed through),
        # find inner bottom instead.
        if not is_from_horiz and is_new_horiz:
            # Check by position, not by normal
            # (normal may be flipped by _probe_for_edge)
            snap_h = closest[0][self.height_axis]
            bmin_h = self.mesh.bounds[0][self.height_axis]
            bmax_h = self.mesh.bounds[1][self.height_axis]

            # DEBUG INSIDE
            self._wall_side_debug = (
                f"INSIDE_SC|sh={snap_h:.1f}"
                f"|bm={bmin_h:.1f}"
            )
            
            old_n_h = old_normal.copy()
            old_n_h[self.height_axis] = 0.0
            old_radial = old_pos - self._mesh_centroid
            old_radial[self.height_axis] = 0.0

            if (
                np.linalg.norm(old_n_h) > 0.3
                and np.linalg.norm(old_radial) > 1e-8
            ):
                was_inner = float(np.dot(old_n_h, old_radial)) < 0
                
                # Outer bottom: snap point near bbox_min
                # Inner bottom: snap point NOT near bbox_min
                # (inner bottom z ≈ bbox_min + wall_thickness)
                snap_near_bbox_min = abs(snap_h - bmin_h) < 3.0
                snap_near_bbox_max = abs(snap_h - bmax_h) < 3.0
                
                # Crossing: inner wall → outer bottom
                is_outer_bottom = snap_near_bbox_min and was_inner
                
                self._wall_side_debug = (
                    f"SC|sh={snap_h:.1f}"
                    f"|bm={bmin_h:.1f}"
                    f"|snbm={int(snap_near_bbox_min)}"
                    f"|wi={int(was_inner)}"
                    f"|ob={int(is_outer_bottom)}"
                )

                if is_outer_bottom:
                    # Probe UPWARD from outer bottom to find inner bottom
                    probe = closest[0] + self.up_direction * 5.0
                    cl_fb, d_fb, f_fb = (
                        self.mesh.nearest.on_surface([probe])
                    )
                    n_fb = self.mesh.face_normals[f_fb[0]].copy()
                    n_fb /= np.linalg.norm(n_fb) + 1e-12
                    
                    # Inner bottom: z > outer bottom z
                    # (inner bottom z ≈ bbox_min + wall_thickness)
                    fb_z = cl_fb[0][self.height_axis]
                    is_above_outer = (
                        (fb_z - bmin_h) * self.up_sign > 1.0
                    )
                    
                    if self._is_horizontal(n_fb) and is_above_outer:
                        # Align normal toward agent
                        agent_to_fb = old_pos - cl_fb[0]
                        if np.dot(n_fb, agent_to_fb) < 0:
                            n_fb = -n_fb
                        hit_n = n_fb
                        closest = cl_fb
                    else:
                        return {
                            "position": old_pos,
                            "normal": old_normal,
                            "valid": False,
                        }
        # ═══ Side consistency: wall → wall ═══
        # Use normal dot product instead of outward check.
        # Outward check fails on thin/curved objects where
        # normals relative to centroid are unreliable.
        # Dot product directly detects opposite-side snap:
        #   dot > 0: same side (normals agree)
        #   dot ≈ 0: perpendicular (edge transition, handled elsewhere)
        #   dot < -0.3: opposite side (normals oppose)
        if not is_from_horiz and not is_new_horiz:
            dot_normals = float(np.dot(old_normal, hit_n))
            if dot_normals < -0.5:
                return {
                    "position": old_pos,
                    "normal": old_normal,
                    "valid": False,
                }

        new_pos = closest[0] + hit_n * 2.0

        return {
            "position": new_pos,
            "normal": hit_n,
            "edge_traversed": edge_traversed,
            "valid": True,
        }
    
    def _is_horizontal_old(self, normal):
        return abs(float(np.dot(normal, self.up_direction))) > 0.85

    def _is_horizontal(self, normal, curvature_k1=None):
        """Check if surface is horizontal (rim, bottom, top).
        
        High curvature surfaces with vertical normals (e.g. torus
        handle top/bottom) are NOT horizontal — they are curved
        geometry that happens to have a vertical normal at that point.
        """
        return abs(float(np.dot(normal, self.up_direction))) > 0.85
        # needs improvements
        if abs(float(np.dot(normal, self.up_direction))) <= 0.85:
            return False
        if curvature_k1 is not None and abs(curvature_k1) > 0.1:
            return False
        return True

    def _align_normal(self, hit_n, old_normal, is_from_horiz):
        """Align snap normal with agent's current orientation."""
        if np.dot(hit_n, old_normal) < 0:
            if not is_from_horiz:
                hit_n = -hit_n
            elif self._is_horizontal(hit_n):
                # horizontal→horizontal: flip to match
                # (outer bottom snap finds inner bottom)
                hit_n = -hit_n
        return hit_n

    def _can_transition(self, hit_n, old_normal, is_from_horiz, is_new_horiz):
        """Check if transition to new face is allowed."""
        if is_from_horiz and not is_new_horiz:
            return True  # horizontal→vertical always OK
        return float(np.dot(hit_n, old_normal)) > -0.1

    def _probe_for_edge(self, old_normal, world_dir, snap_point, step_size):
        """Probe for adjacent perpendicular face near edges.
        
        Checks all 3 axes. For each axis where agent is near
        a bbox boundary and moving toward it, probes beyond
        the boundary for a face with different orientation.
        
        Returns:
            dict {point, normal} or None
        """
        bbox_min = self.mesh.bounds[0]
        bbox_max = self.mesh.bounds[1]
        
        best = None
        best_dist = float('inf')
        
        for axis in range(3):
            # Skip axis aligned with current normal
            if abs(old_normal[axis]) > 0.7:
                continue
            
            agent_coord = self.agent_pos[axis]
            move_comp = world_dir[axis]
            
            # Check both boundaries
            for boundary, sign_threshold in [
                (bbox_max[axis], 0.15),
                (bbox_min[axis], -0.15),
            ]:
                dist_to_boundary = abs(agent_coord - boundary)
                moving_toward = (
                    (boundary == bbox_max[axis] and move_comp > sign_threshold)
                    or (boundary == bbox_min[axis] and move_comp < -abs(sign_threshold))
                )
                
                if dist_to_boundary < step_size * 2.5 and moving_toward:
                    # Probe beyond boundary
                    probe = self.agent_pos.copy()
                    offset = 1.0 if boundary == bbox_max[axis] else -1.0
                    probe[axis] = boundary + offset
                    
                    cl, d, f = self.mesh.nearest.on_surface([probe])
                    n_probe = self.mesh.face_normals[f[0]].copy()
                    n_probe /= np.linalg.norm(n_probe) + 1e-12
                    
                    # Must be a different face orientation
                    dot_probe_old = float(
                        np.dot(n_probe, old_normal)
                    )
                    normals_different = (
                        abs(dot_probe_old) < 0.5
                        and dot_probe_old > -0.3
                    )

                    # Must be close to boundary
                    face_near_boundary = any(
                        abs(cl[0][ax] - bbox_max[ax]) < step_size
                        or abs(cl[0][ax] - bbox_min[ax]) < step_size
                        for ax in range(3) if abs(old_normal[ax]) < 0.7
                    )
                    
                    if (
                        normals_different
                        and d[0] < step_size * 3.0
                        and face_near_boundary
                        and dist_to_boundary < best_dist
                    ):
                        # Ensure normal points toward agent
                        agent_to = self.agent_pos - cl[0]
                        if np.dot(n_probe, agent_to) < 0:
                            n_probe = -n_probe
                        
                        best_dist = dist_to_boundary
                        best = {"point": cl[0], "normal": n_probe}
        
        return best

    def _validate_snap(self, old_pos, old_normal, world_dir, snap, step_size):
        """Validate snap doesn't pass through walls.
        
        If through-wall detected, tries same-side fallback.
        """
        if not snap["valid"]:
            return snap
        
        new_pos = snap["position"]
        
        if self._check_through_wall(old_pos, new_pos):
            # Try same-side fallback
            probe = old_pos + world_dir * step_size + old_normal * 3.0
            cl, d, f = self.mesh.nearest.on_surface([probe])
            n_fb = self.mesh.face_normals[f[0]].copy()
            n_fb /= np.linalg.norm(n_fb) + 1e-12
            
            if np.dot(n_fb, old_normal) < 0:
                n_fb = -n_fb
            
            fallback_pos = cl[0] + n_fb * 2.0
            
            if (
                not self._check_through_wall(old_pos, fallback_pos)
                and np.dot(n_fb, old_normal) > -0.1
                and d[0] < step_size * 3.0
            ):
                snap["position"] = fallback_pos
                snap["normal"] = n_fb
            else:
                snap["valid"] = False
        
        return snap


    def _check_through_wall(self, old_pos, new_pos, margin=2.5):
        """Check if path passes through mesh geometry."""
        move_vec = new_pos - old_pos
        move_dist = float(np.linalg.norm(move_vec))
        
        if move_dist <= margin * 2:
            return False
        
        move_dir = move_vec / move_dist
        locs, _, _ = self.mesh.ray.intersects_location(
            ray_origins=[old_pos],
            ray_directions=[move_dir],
        )
        if len(locs) == 0:
            return False
        
        dists = np.linalg.norm(locs - old_pos, axis=1)
        return int(np.sum(
            (dists > margin) & (dists < move_dist - margin)
        )) > 0

    def _correct_wall_side(self, old_pos, old_normal, snap_point, hit_n):
        """Ensure wall normal points to same side as agent was on.
        
        When transitioning horizontal→vertical (rim→wall),
        nearest.on_surface may find the wrong side of a thin wall.
        Check using agent position relative to object center.
        """
        height_ax = self.height_axis
        
        hit_n_h = hit_n.copy()
        hit_n_h[height_ax] = 0.0
        from_center = old_pos - self._mesh_centroid
        from_center[height_ax] = 0.0
        
        if (
            np.linalg.norm(hit_n_h) < 1e-8
            or np.linalg.norm(from_center) < 1e-8
        ):
            return hit_n
        
        # Agent was on outer side if old_normal pointed outward
        old_n_h = old_normal.copy()
        old_n_h[height_ax] = 0.0
        
        # For horizontal surfaces, use position-based side detection
        if np.linalg.norm(old_n_h) < 0.3:
            # On rim/bottom — determine side by distance from center
            # vs object radius
            dist_from_center = float(np.linalg.norm(from_center))
            extents_h = [
                (self.mesh.bounds[1][i] - self.mesh.bounds[0][i])
                for i in range(3) if i != height_ax
            ]
            approx_radius = max(extents_h) / 2.0
            was_outside = dist_from_center > approx_radius * 0.8
        else:
            was_outside = float(
                np.dot(old_n_h, from_center)
            ) > 0
        
        now_outside = float(
            np.dot(hit_n_h, from_center)
        ) > 0
        
        if was_outside != now_outside:
            # Wrong side — try opposite
            opp_pos = snap_point - hit_n * 5.0
            cl, _, f = self.mesh.nearest.on_surface([opp_pos])
            n_opp = self.mesh.face_normals[f[0]].copy()
            n_opp /= np.linalg.norm(n_opp) + 1e-12
            
            if not self._is_horizontal(n_opp):
                n_opp_h = n_opp.copy()
                n_opp_h[height_ax] = 0.0
                if np.linalg.norm(n_opp_h) > 1e-8:
                    opp_outside = float(
                        np.dot(n_opp_h, from_center)
                    ) > 0
                    if opp_outside == was_outside:
                        return n_opp
        
        return hit_n

    def _try_intermediate_edge(self, old_pos, old_normal, world_dir,
                            step_size, is_from_horiz):
        """Try edge traversal via intermediate half-step."""
        # Check ray blockage
        locs, _, _ = self.mesh.ray.intersects_location(
            ray_origins=[old_pos], ray_directions=[world_dir]
        )
        if len(locs) > 0:
            if np.min(np.linalg.norm(locs - old_pos, axis=1)) < step_size * 0.5:
                return None  # ray blocked
        
        half = old_pos + world_dir * step_size * 0.5
        cl_h, d_h, f_h = self.mesh.nearest.on_surface([half])
        n_h = self.mesh.face_normals[f_h[0]].copy()
        n_h /= np.linalg.norm(n_h) + 1e-12
        
        if np.dot(n_h, old_normal) < 0:
            if not is_from_horiz:
                n_h = -n_h
        
        if np.dot(n_h, old_normal) <= -0.1 or d_h[0] >= step_size * 2.0:
            return None
        
        inter = cl_h[0] + n_h * 2.0
        full = inter + world_dir * step_size * 0.5
        cl_f, d_f, f_f = self.mesh.nearest.on_surface([full])
        n_f = self.mesh.face_normals[f_f[0]].copy()
        n_f /= np.linalg.norm(n_f) + 1e-12
        
        if np.dot(n_f, n_h) < 0:
            n_f = -n_f
        
        if np.dot(n_f, n_h) <= -0.1 or d_f[0] >= step_size * 2.0:
            return None
        
        new_pos = cl_f[0] + n_f * 2.0
        
        # Through-wall check
        if self._check_through_wall(old_pos, new_pos):
            return None

        # ═══ Side consistency: wall → horizontal ═══
        new_is_h = self._is_horizontal(n_f)
        if not is_from_horiz and new_is_h:
            old_n_h = old_normal.copy()
            old_n_h[self.height_axis] = 0.0
            old_rad = old_pos - self._mesh_centroid
            old_rad[self.height_axis] = 0.0

            if (
                np.linalg.norm(old_n_h) > 0.3
                and np.linalg.norm(old_rad) > 1e-8
            ):
                was_inner = (
                    float(np.dot(old_n_h, old_rad)) < 0
                )
                bottom_outer = (
                    n_f[self.height_axis]
                    * self.up_sign
                    < 0
                )

                if was_inner and bottom_outer:
                    # Crossed through bottom — find inner bottom
                    probe = cl_f[0] + old_normal * 5.0
                    cl_fb, _, f_fb = (
                        self.mesh.nearest.on_surface(
                            [probe]
                        )
                    )
                    n_fb = self.mesh.face_normals[
                        f_fb[0]
                    ].copy()
                    n_fb /= (
                        np.linalg.norm(n_fb) + 1e-12
                    )

                    if (
                        self._is_horizontal(n_fb)
                        and n_fb[self.height_axis]
                        * self.up_sign
                        > 0
                    ):
                        return {
                            "position": cl_fb[0]
                            + n_fb * 2.0,
                            "normal": n_fb,
                            "edge_traversed": True,
                            "valid": True,
                        }
                    return None
        return {
            "position": new_pos,
            "normal": n_f,
            "edge_traversed": True,
            "valid": True,
        }

    def _move_forward(self, step_size):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        forward = rot.apply([0, 0, -1])
        old_pos = self.agent_pos.copy()
        self.agent_pos += forward * step_size
        self._passed_through = False

        if abs(step_size) > 0.5:
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=[old_pos],
                ray_directions=[forward * np.sign(step_size)],
            )
            if len(locations) > 0:
                distances = np.linalg.norm(locations - old_pos, axis=1)
                if np.min(distances) < abs(step_size):
                    self._passed_through = True

            closest, dist_to_mesh, _ = self.mesh.nearest.on_surface([self.agent_pos])
            # Adaptive proximity threshold based on step size
            proximity_threshold = min(1.0, abs(step_size) * 0.25)
            if dist_to_mesh[0] < proximity_threshold:
                self._passed_through = True
                
    def _orient_horizontal(self, rotation_degrees, forward_distance, left_distance):
        """Moves the agent forward and sideways (left/right), projects onto surface, and yaws.
        
        Args:
            rotation_degrees: additional yaw rotation (e.g. +45 for right turn)
            forward_distance: mm to move forward (+) or backward (-)
            left_distance: mm to move left (+) or right (-)
        """
        # Local movement direction
        local_dir = np.array([
            -left_distance,        # +left → -X (left), -left → +X (right)
            0.0,
            -forward_distance      # +forward → -Z, -forward → +Z
        ])

        if np.linalg.norm(local_dir) < 1e-8:
            local_dir = np.array([0.0, 0.0, -1.0])  # default: forward
        else:
            local_dir = local_dir / np.linalg.norm(local_dir)

        # Rotate to world
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        world_dir = rot.apply(local_dir)

        # Project onto tangent plane
        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is not None:
            normal = np.array(sensor_data["point_normal"])
            world_dir = world_dir - np.dot(world_dir, normal) * normal
            norm = np.linalg.norm(world_dir)
            if norm > 1e-8:
                world_dir /= norm

        # Total step distance in mm
        step = np.linalg.norm([forward_distance, left_distance])
        self.agent_pos += world_dir * step

        # Apply yaw rotation
        self.agent_rot[1] += rotation_degrees  # Y-axis = yaw

    def _orient_vertical(self, rotation_degrees, forward_distance, down_distance):
        """Moves the agent forward and down/up, projects onto surface, and pitches.
        
        Args:
            rotation_degrees: additional pitch rotation (e.g. +10 for look down)
            forward_distance: mm to move forward (+) or backward (-)
            down_distance: mm to move down (+) or up (-)
        """
        # Local movement: X=0, Y=-down, Z=-forward
        local_dir = np.array([
            0.0,
            -down_distance,        # +down → -Y, -down → +Y (up)
            -forward_distance      # +forward → -Z
        ])

        if np.linalg.norm(local_dir) < 1e-8:
            local_dir = np.array([0.0, 0.0, -1.0])  # default: forward
        else:
            local_dir = local_dir / np.linalg.norm(local_dir)

        # Rotate to world
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        world_dir = rot.apply(local_dir)

        # Project onto tangent plane
        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is not None:
            normal = np.array(sensor_data["point_normal"])
            world_dir = world_dir - np.dot(world_dir, normal) * normal
            norm = np.linalg.norm(world_dir)
            if norm > 1e-8:
                world_dir /= norm

        # Total step distance
        step = np.linalg.norm([forward_distance, down_distance])
        self.agent_pos += world_dir * step

        # Apply pitch rotation (around X axis)
        self.agent_rot[0] += rotation_degrees  # X-axis = pitch

    def _compute_up_direction(self):
        """Determine the 'up' direction for the object.

        Strategy:
        1. For objects with clear asymmetry (hollow objects like mugs,
           vases): use ray-based asymmetry detection. The open side
           has fewer ray hits → that's 'up'.
        2. For symmetric objects (cubes, flat squares): fall back to
           the thinnest axis as height. This correctly identifies
           the thin dimension of flat objects.

        The up_direction is used for:
        - same_side detection (which side of object agent is on)
        - crawl_to_edge direction (which way to crawl toward rim)
        - detach fly direction (which way to fly after detaching)
        - horizontal surface detection (is_horizontal check)
        """
        bbox_min = self.mesh.bounds[0]
        bbox_max = self.mesh.bounds[1]
        extents = bbox_max - bbox_min

        # Use centroid — resistant to handles and other protrusions
        center = np.array(self.mesh.centroid, dtype=float)

        n_probes = 25
        best_axis = 0
        best_asymmetry = -1
        best_up_sign = 1.0
        axis_info = {}

        for axis in range(3):
            horiz_axes = [i for i in range(3) if i != axis]

            probe_points = []
            for i in range(5):
                for j in range(5):
                    pt = center.copy()
                    r0 = (
                        min(
                            extents[horiz_axes[0]],
                            extents[horiz_axes[1]],
                        )
                        * 0.3
                    )
                    pt[horiz_axes[0]] = (
                        center[horiz_axes[0]]
                        + r0 * (i / 4 - 0.5)
                    )
                    pt[horiz_axes[1]] = (
                        center[horiz_axes[1]]
                        + r0 * (j / 4 - 0.5)
                    )
                    probe_points.append(pt)
            probe_points = np.array(probe_points)

            # Rays in the + direction
            dir_plus = np.zeros(3)
            dir_plus[axis] = 1.0
            origins_plus = probe_points.copy()
            origins_plus[:, axis] = center[axis] + 1.0
            hits_plus, _, _ = self.mesh.ray.intersects_location(
                ray_origins=origins_plus,
                ray_directions=np.tile(
                    dir_plus, (n_probes, 1)
                ),
            )

            # Rays in the - direction
            dir_minus = np.zeros(3)
            dir_minus[axis] = -1.0
            origins_minus = probe_points.copy()
            origins_minus[:, axis] = center[axis] - 1.0
            hits_minus, _, _ = self.mesh.ray.intersects_location(
                ray_origins=origins_minus,
                ray_directions=np.tile(
                    dir_minus, (n_probes, 1)
                ),
            )

            n_plus = len(hits_plus)
            n_minus = len(hits_minus)
            asymmetry = abs(n_plus - n_minus)

            axis_info[axis] = {
                "plus": n_plus,
                "minus": n_minus,
                "asymmetry": asymmetry,
            }

            if asymmetry > best_asymmetry:
                best_asymmetry = asymmetry
                best_axis = axis
                if n_plus < n_minus:
                    best_up_sign = 1.0
                elif n_minus < n_plus:
                    best_up_sign = -1.0
                else:
                    best_up_sign = 1.0

        # ═══ FALLBACK for symmetric objects ═══
        # For objects like flat_square, cube, sphere where all axes
        # have similar asymmetry, the ray-based method picks an
        # arbitrary axis. Instead, use the thinnest axis as height.
        #
        # Threshold: if best asymmetry is small relative to probe
        # count, the object is approximately symmetric.
        #
        # For hollow objects (mug, vase) asymmetry is typically
        # 10-25 (open top has far fewer hits). For symmetric objects
        # (cube, sphere, flat_square) asymmetry is 0-3.
        MIN_ASYMMETRY_THRESHOLD = 5
        if best_asymmetry < MIN_ASYMMETRY_THRESHOLD:
            sorted_extents = np.sort(extents)
            thinnest = sorted_extents[0]
            second_thinnest = sorted_extents[1]
            thickest = sorted_extents[2]

            if thinnest < second_thinnest * 0.3:
                # Flat object (100×100×3) → thinnest = height
                thin_axis = int(np.argmin(extents))
                best_axis = thin_axis
                best_up_sign = 1.0

                logger.debug(
                    f"_compute_up_direction: flat object, "
                    f"using thinnest axis {thin_axis} as height "
                    f"(extents={extents.tolist()}, "
                    f"asymmetry={best_asymmetry})"
                )
            elif thickest > second_thinnest * 2.0:
                # Elongated object (10×10×120) → longest = height
                long_axis = int(np.argmax(extents))
                best_axis = long_axis
                best_up_sign = 1.0

                logger.debug(
                    f"_compute_up_direction: elongated object, "
                    f"using longest axis {long_axis} as height "
                    f"(extents={extents.tolist()}, "
                    f"asymmetry={best_asymmetry})"
                )
            else:
                # Truly symmetric (cube, sphere)
                logger.debug(
                    f"_compute_up_direction: symmetric object, "
                    f"no thin/long axis, keeping ray-based "
                    f"axis={best_axis} "
                    f"(extents={extents.tolist()}, "
                    f"asymmetry={best_asymmetry})"
                )

        self.height_axis = best_axis
        self.up_sign = best_up_sign
        self.up_direction = np.zeros(3)
        self.up_direction[self.height_axis] = self.up_sign
        self.open_edge_height = (
            bbox_max[self.height_axis]
            if self.up_sign > 0
            else bbox_min[self.height_axis]
        )

        logger.debug(
            f"_compute_up_direction: "
            f"extents={extents.tolist()}, "
            f"centroid={center.tolist()}, "
            f"axis_info={axis_info}, "
            f"best_asymmetry={best_asymmetry}, "
            f"height_axis={self.height_axis}, "
            f"up_sign={self.up_sign}, "
            f"up_direction={self.up_direction.tolist()}, "
            f"open_edge_height={self.open_edge_height}"
        )

    def _detach_simple(self, goal_pose, detach_distance=8.0):
        """Detach from surface along normal, then look toward goal.

        Simple macro action:
        1. Move along surface normal by detach_distance
        2. Orient gaze toward goal position

        After detach, Q-store/SAC decide what to do next
        (turn, fly forward, detach again, etc.)

        Args:
            goal_pose: Target pose [x, y, z, rx, ry, rz].
            detach_distance: How far to fly from surface (mm).
        """
        self._detach_had_collision = False
        self._last_detach_sub_steps = 1

        sensor = self.get_sensor_data()
        if sensor.get("point_normal") is None:
            logger.debug("DETACH_SIMPLE_ABORT: no point_normal")
            return self.get_sensor_data()

        normal = np.array(sensor["point_normal"], dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-12)

        # Step 1: Move along normal (away from surface)
        old_pos = self.agent_pos.copy()
        self.agent_pos += normal * detach_distance

        # Check for collision during detach
        locations, _, _ = self.mesh.ray.intersects_location(
            ray_origins=[old_pos],
            ray_directions=[normal],
        )
        if len(locations) > 0:
            distances = np.linalg.norm(locations - old_pos, axis=1)
            if np.min(distances) < detach_distance:
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                logger.debug("DETACH_SIMPLE_COLLISION: hit object during detach")
                return self.get_sensor_data()

        # Step 2: Orient gaze toward goal
        # Adapt fly direction based on whether goal is behind surface
        goal_pos = goal_pose[:3]
        goal_dir = goal_pos - self.agent_pos
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 1e-8:
            goal_dir /= goal_dist
            dot_goal_normal = float(np.dot(goal_dir, normal))

            if dot_goal_normal < -0.2:
                # Goal is behind surface — fly sideways to go around
                # Project goal_dir onto tangent plane
                tangent_to_goal = goal_dir - dot_goal_normal * normal
                tangent_len = float(np.linalg.norm(tangent_to_goal))

                if tangent_len > 1e-8:
                    tangent_to_goal /= tangent_len
                    # Fly: away from surface + sideways toward goal
                    fly_direction = normal * 0.7 + tangent_to_goal * 0.7
                else:
                    # Goal exactly behind normal — fly away from surface
                    fly_direction = normal

                fly_direction /= (np.linalg.norm(fly_direction) + 1e-12)
            else:
                # Goal is in front of or beside surface — standard blend
                fly_direction = goal_dir + normal * 0.3
                fly_direction /= (np.linalg.norm(fly_direction) + 1e-12)

            self.agent_rot = self._look_at_direction(fly_direction)
            
        logger.debug(
            f"DETACH_SIMPLE_DONE: "
            f"old_pos={old_pos.tolist()}, "
            f"new_pos={self.agent_pos.tolist()}, "
            f"normal={normal.tolist()}, "
            f"goal_dir={goal_dir.tolist()}, "
            f"dist_to_goal={goal_dist:.1f}"
        )

        return self.get_sensor_data()

    def _detach_and_fly_to_edge(self, goal_pose, rotation_step=5.0, free_step=8.0, max_sub_steps=3):
        detach_fly_step = 8.0
        safe_step = 3.0
        sub_steps = 0
        detach_step = 8.0

        self._detach_had_collision = False

        logger.debug(
            f"DETACH_EDGE_START: "
            f"agent_pos={self.agent_pos.tolist()}, "
            f"agent_rot={self.agent_rot.tolist()}, "
            f"goal_pos={goal_pose[:3].tolist()}, "
            f"height_axis={self.height_axis}, "
            f"up_direction={self.up_direction.tolist()}, "
            f"open_edge_height={self.open_edge_height:.1f}, "
            f"bbox={self.mesh.bounds.tolist()}"
        )

        sensor = self.get_sensor_data()
        if sensor.get("point_normal") is None:
            logger.debug("DETACH_EDGE_ABORT: no point_normal")
            self._last_detach_sub_steps = sub_steps
            return self.get_sensor_data()

        normal = np.array(sensor["point_normal"], dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-12)

        # ═══ Phase 1: Normal Breakaway ═══
        self.agent_rot = self._look_at_direction(normal)

        logger.debug(
            f"DETACH_EDGE_PHASE1_START: "
            f"normal={normal.tolist()}, "
            f"agent_rot={self.agent_rot.tolist()}, "
            f"detach_step={detach_step}"
        )

        flown = 0.0
        while flown < detach_step:
            old_pos = self.agent_pos.copy()
            step = min(safe_step, detach_step - flown)
            sensor_check = self.get_sensor_data()
            depth = sensor_check.get("depth", 100.0)
            if depth < step + 1.0:
                step = max(depth - 1.0, 0.5)
            self._move_forward(step)
            flown += step
            sub_steps += 1
            if self._passed_through:
                logger.debug(
                    f"DETACH_EDGE_COLLISION: "
                    f"phase=1, sub_step={sub_steps}, "
                    f"pos_before={old_pos.tolist()}, "
                    f"pos_after={self.agent_pos.tolist()}, "
                    f"normal={normal.tolist()}, "
                    f"step={step:.1f}, "
                    f"depth={depth:.1f}, "
                    f"flown={flown:.1f}"
                )
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                self._last_detach_sub_steps = sub_steps
                return self.get_sensor_data()

        logger.debug(
            f"DETACH_EDGE_PHASE1_DONE: "
            f"pos={self.agent_pos.tolist()}, "
            f"flown={flown:.1f}, "
            f"sub_steps={sub_steps}"
        )

        # ═══ Phase 2: Fly to the open edge ═══
        up_dir = self.up_direction
        height_axis = self.height_axis
        open_edge_height = self.open_edge_height

        center = np.array(self.mesh.centroid, dtype=float)

        # Inside or outside?
        agent_to_center = center - self.agent_pos
        agent_to_center[height_axis] = 0.0
        dot_normal_to_center = np.dot(normal, agent_to_center)
        is_inside = dot_normal_to_center > 0

        if is_inside:
            to_center_horiz = agent_to_center.copy()
            to_center_len = np.linalg.norm(to_center_horiz)
            if to_center_len > 1e-8:
                to_center_horiz /= to_center_len
            fly_dir = up_dir * 1.0 + to_center_horiz * 0.2
            overshoot = detach_fly_step
        else:
            fly_dir = normal * 1.0 + up_dir * 0.5
            overshoot = detach_fly_step

        fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)
        self.agent_rot = self._look_at_direction(fly_dir)

        target_height = open_edge_height + overshoot * self.up_sign

        logger.debug(
            f"DETACH_EDGE_PHASE2_START: "
            f"is_inside={is_inside}, "
            f"normal={normal.tolist()}, "
            f"center={center.tolist()}, "
            f"agent_to_center_horiz={agent_to_center.tolist()}, "
            f"dot_normal_to_center={dot_normal_to_center:.3f}, "
            f"fly_dir={fly_dir.tolist()}, "
            f"up_dir={up_dir.tolist()}, "
            f"target_height={target_height:.1f}, "
            f"current_height={self.agent_pos[height_axis]:.1f}, "
            f"open_edge_height={open_edge_height:.1f}, "
            f"agent_rot={self.agent_rot.tolist()}"
        )

        max_fly_steps = 20
        for fly_step_idx in range(max_fly_steps):
            sub_steps += 1
            old_pos = self.agent_pos.copy()

            sensor_check = self.get_sensor_data()
            depth = sensor_check.get("depth", 100.0)
            step = detach_fly_step
            if depth < step + 1.0:
                step = max(depth - 1.0, 0.5)

            self._move_forward(step)

            if self._passed_through:
                logger.debug(
                    f"DETACH_EDGE_COLLISION: "
                    f"phase=2, fly_step_idx={fly_step_idx}, sub_step={sub_steps}, "
                    f"pos_before={old_pos.tolist()}, "
                    f"pos_after={self.agent_pos.tolist()}, "
                    f"fly_dir={fly_dir.tolist()}, "
                    f"step={step:.1f}, "
                    f"depth={depth:.1f}, "
                    f"current_height={old_pos[height_axis]:.1f}, "
                    f"target_height={target_height:.1f}"
                )
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

            if self.up_sign > 0:
                if self.agent_pos[height_axis] > target_height:
                    logger.debug(
                        f"DETACH_EDGE_PHASE2_REACHED: "
                        f"fly_step_idx={fly_step_idx}, "
                        f"height={self.agent_pos[height_axis]:.1f}, "
                        f"target={target_height:.1f}"
                    )
                    break
            elif self.agent_pos[height_axis] < target_height:
                logger.debug(
                    f"DETACH_EDGE_PHASE2_REACHED: "
                    f"fly_step_idx={fly_step_idx}, "
                    f"height={self.agent_pos[height_axis]:.1f}, "
                    f"target={target_height:.1f}"
                )
                break

        logger.debug(
            f"DETACH_EDGE_PHASE2_DONE: "
            f"pos={self.agent_pos.tolist()}, "
            f"height={self.agent_pos[height_axis]:.1f}, "
            f"had_collision={self._detach_had_collision}, "
            f"sub_steps={sub_steps}"
        )

        # If there was a collision in phase 2, do not continue phase 3
        if self._detach_had_collision:
            self._last_detach_sub_steps = sub_steps
            return self.get_sensor_data()

        # ═══ Phase 3: Horizontal flight towards the target ═══
        goal_pos = goal_pose[:3]

        direction_to_goal = goal_pos - self.agent_pos
        direction_horiz = direction_to_goal.copy()
        direction_horiz[height_axis] = 0.0

        horiz_dist = np.linalg.norm(direction_horiz)
        if horiz_dist > 1e-8:
            direction_horiz /= horiz_dist
        else:
            direction_horiz = -normal
            direction_horiz[height_axis] = 0.0
            direction_horiz /= (np.linalg.norm(direction_horiz) + 1e-12)

        self.agent_rot = self._look_at_direction(direction_horiz)

        total_fly = int(detach_fly_step * max_sub_steps * 1.5)

        logger.debug(
            f"DETACH_EDGE_PHASE3_START: "
            f"pos={self.agent_pos.tolist()}, "
            f"goal_pos={goal_pos.tolist()}, "
            f"direction_horiz={direction_horiz.tolist()}, "
            f"horiz_dist_to_goal={horiz_dist:.1f}, "
            f"total_fly={total_fly:.1f}, "
            f"agent_rot={self.agent_rot.tolist()}"
        )

        flown = 0.0
        phase3_step_idx = 0
        while flown < total_fly:
            old_pos = self.agent_pos.copy()
            step = min(detach_fly_step, total_fly - flown)
            self._move_forward(step)
            flown += step
            sub_steps += 1
            phase3_step_idx += 1

            if self._passed_through:
                logger.debug(
                    f"DETACH_EDGE_COLLISION: "
                    f"phase=3, phase3_step_idx={phase3_step_idx}, sub_step={sub_steps}, "
                    f"pos_before={old_pos.tolist()}, "
                    f"pos_after={self.agent_pos.tolist()}, "
                    f"direction_horiz={direction_horiz.tolist()}, "
                    f"step={step:.1f}, "
                    f"flown={flown:.1f}, "
                    f"total_fly={total_fly:.1f}"
                )
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

        logger.debug(
            f"DETACH_EDGE_DONE: "
            f"sub_steps={sub_steps}, "
            f"had_collision={self._detach_had_collision}, "
            f"final_pos={self.agent_pos.tolist()}, "
            f"final_height={self.agent_pos[height_axis]:.1f}, "
            f"goal_pos={goal_pose[:3].tolist()}, "
            f"final_dist_to_goal={np.linalg.norm(self.agent_pos - goal_pose[:3]):.1f}"
        )

        self._last_detach_sub_steps = sub_steps

        # direct agent gaze downwards
        down_direction = -self.up_direction
        self.agent_rot = self._look_at_direction(down_direction)

        return self.get_sensor_data()

    def _detach_and_fly_to_goal(self, goal_pose, rotation_step=5.0, free_step=8.0, max_sub_steps=3):
        detach_fly_step = 8.0
        sub_steps = 1
        self._detach_had_collision = False

        sensor = self.get_sensor_data()
        normal = None
        if sensor.get("point_normal") is not None:
            normal = np.array(sensor["point_normal"], dtype=float)
            normal /= (np.linalg.norm(normal) + 1e-12)
            self.agent_rot = self._look_at_direction(normal)

        total_fly = detach_fly_step * max_sub_steps
        flown = 0.0
        while flown < total_fly:
            old_pos = self.agent_pos.copy()
            sensor = self.get_sensor_data()
            depth = sensor.get("depth", 100.0)
            step = min(detach_fly_step, total_fly - flown)
            if depth < step:
                step = max(depth - 2.0, 0.5)
            self._move_forward(step)
            flown += step

            if self._passed_through:
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

        goal_pos = goal_pose[:3]
        direction = goal_pos - self.agent_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-8:
            direction /= dist

        if normal is not None:
            fly_direction = direction + normal * 0.5
            fly_direction /= (np.linalg.norm(fly_direction) + 1e-12)
        else:
            fly_direction = direction

        self.agent_rot = self._look_at_direction(fly_direction)

        for _ in range(max_sub_steps):
            sub_steps += 1
            old_pos = self.agent_pos.copy()

            sensor = self.get_sensor_data()
            depth = sensor.get("depth", 100.0)

            if depth < detach_fly_step:
                step = max(depth - 2.0, 0.5)
                self._move_forward(step)
            else:
                self._move_forward(detach_fly_step)

            if self._passed_through:
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

            sensor = self.get_sensor_data()
            if sensor.get("on_object", False):
                break

        self._last_detach_sub_steps = sub_steps
        return self.get_sensor_data()

    def render_episode_frame(
        self,
        agent_pose: np.ndarray,
        goal_pose: np.ndarray,
        filepath: str,
        trail_poses: Optional[List[np.ndarray]] = None,
        text: str = "",
        step_num: int = 0,
        distance: float = 0.0,
        result: str = "",
    ) -> None:
        """Render split-view frame: solid + x-ray using trimesh.
        
        Implements RenderableEnv protocol.
        """
        from .visualize_env import add_text_overlay
        
        resolution = (1024, 768)
        half_res = (resolution[0] // 2, resolution[1])
        fov = 50.0
        mesh_alpha_solid = 255
        mesh_alpha_xray = 60

        def build_scene(alpha):
            import trimesh as _trimesh
            scene = _trimesh.Scene()
            mesh_copy = self.mesh.copy()
            mesh_copy.visual.face_colors = [200, 200, 200, alpha]
            scene.add_geometry(mesh_copy, geom_name="mesh")
            
            # Agent (blue)
            agent_s = _trimesh.primitives.Sphere(radius=1.0, center=agent_pose[:3])
            agent_s.visual.face_colors = [0, 50, 255, 255]
            scene.add_geometry(agent_s, geom_name="agent")
            
            # Goal (green)
            goal_s = _trimesh.primitives.Sphere(radius=1.4, center=goal_pose[:3])
            goal_s.visual.face_colors = [0, 255, 0, 255]
            scene.add_geometry(goal_s, geom_name="goal")
            
            # Gaze (red line)
            rot = R.from_euler("xyz", agent_pose[3:], degrees=True)
            fwd = rot.apply([0, 0, -1])
            line_verts = np.array([agent_pose[:3], agent_pose[:3] + fwd * 3])
            arrow = _trimesh.load_path(line_verts, colors=[[255, 0, 0, 255]])
            scene.add_geometry(arrow, geom_name="gaze")
            
            # Trail (orange)
            if trail_poses:
                for i, tp in enumerate(trail_poses):
                    ts = _trimesh.primitives.Sphere(radius=0.6, center=tp[:3])
                    ts.visual.face_colors = [255, 165, 0, 255]
                    scene.add_geometry(ts, geom_name=f"trail_{i}")
            
            return scene

        def compute_camera(scene):
            midpoint = (agent_pose[:3] + goal_pose[:3]) / 2.0
            mesh_size = float(np.linalg.norm(scene.bounds[1] - scene.bounds[0]))
            cam_dist = mesh_size * 1.7
            mesh_center = scene.bounds.mean(axis=0)
            to_agent = agent_pose[:3] - mesh_center
            to_agent[2] = 0
            norm = np.linalg.norm(to_agent)
            dir_h = to_agent / norm if norm > 1e-5 else np.array([1, 0, 0])
            
            cam_pos = midpoint + dir_h * cam_dist * 0.7
            cam_pos[2] = max(agent_pose[2], goal_pose[2]) + cam_dist * 0.5
            
            forward = midpoint - cam_pos
            forward /= np.linalg.norm(forward) + 1e-6
            right = np.cross(forward, [0, 0, 1])
            if np.linalg.norm(right) < 1e-3:
                right = np.cross(forward, [0, 1, 0])
            right /= np.linalg.norm(right) + 1e-6
            up = np.cross(right, forward)
            
            T = np.eye(4)
            T[:3, 0] = right
            T[:3, 1] = up
            T[:3, 2] = -forward
            T[:3, 3] = cam_pos
            return T

        from pathlib import Path as _Path
        from PIL import Image as _Image
        import io as _io

        _Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        try:
            scene_solid = build_scene(mesh_alpha_solid)
            cam = compute_camera(scene_solid)
            scene_solid.camera_transform = cam
            scene_solid.camera.fov = (fov, fov)
            scene_solid.camera.z_near = 1.0
            scene_solid.camera.z_far = 100000.0
            png_solid = scene_solid.save_image(resolution=half_res, visible=False)

            scene_xray = build_scene(mesh_alpha_xray)
            scene_xray.camera_transform = cam
            scene_xray.camera.fov = (fov, fov)
            scene_xray.camera.z_near = 1.0
            scene_xray.camera.z_far = 100000.0
            png_xray = scene_xray.save_image(resolution=half_res, visible=False)

            if not png_solid or not png_xray or len(png_solid) < 100 or len(png_xray) < 100:
                raise RuntimeError("Empty render")

            img_left = _Image.open(_io.BytesIO(png_solid))
            img_right = _Image.open(_io.BytesIO(png_xray))
            merged = _Image.new("RGBA", resolution)
            merged.paste(img_left, (0, 0))
            merged.paste(img_right, (half_res[0], 0))

            if text:
                merged = add_text_overlay(merged, text, step_num, distance, result)
                draw = ImageDraw.Draw(merged)
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
                except OSError:
                    font = ImageFont.load_default()
                draw.text((10, resolution[1] - 20), "SOLID", fill="white", font=font)
                draw.text((half_res[0] + 10, resolution[1] - 20), "X-RAY", fill="white", font=font)

            merged.save(filepath, format="PNG")

        except Exception as exc:
            # Fallback: save text file
            txt_path = _Path(filepath).with_suffix(".txt")
            with txt_path.open("w") as f:
                f.write(f"Step {step_num}, dist={distance:.1f}mm: {text}\n")
            logger.debug("Trimesh render failed for %s", filepath, exc_info=True)

    # ═══════════════════════════════════════════════════
    # RLEnvironment protocol implementation
    # ═══════════════════════════════════════════════════

    def step_discrete(self, action_idx: int, action_space) -> dict:
        """Execute discrete action. Delegates to existing step().

        Args:
            action_idx: Discrete action index (0-23).
            action_space: ActionSpace instance.

        Returns:
            Sensor data after action.
        """
        return self.step(action_idx, action_space)

    def step_continuous(
        self, action_type: int, action_params: np.ndarray
    ) -> dict:
        """Execute continuous SAC action via direct env manipulation.

        Same semantics as ActionInterpreter.execute() but without
        requiring a separate interpreter instance. This makes
        LightweightEnv self-contained as an RLEnvironment.

        Args:
            action_type: PSAC action type (0-7).
            action_params: Continuous parameters array [3].

        Returns:
            Sensor data after action.
        """
        self._detach_had_collision = False
        self._edge_traversed = False
        self._passed_through = False

        if action_type == 0:
            # Tangential surface move: [sin_angle, cos_angle, distance]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            angle_deg = float(np.degrees(np.arctan2(sin_a, cos_a)))
            distance = float(np.clip(action_params[2], 0.5, 15.0))
            self._move_tangentially(angle_deg, distance)

        elif action_type == 1:
            # Linear move: [distance, -, -]
            distance = float(np.clip(action_params[0], -25.0, 25.0))
            self._move_forward(distance)

        elif action_type == 2:
            # Yaw (turn left/right): [sin_angle, cos_angle, -]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            rotation = float(np.degrees(np.arctan2(sin_a, cos_a)))
            rotation = float(np.clip(rotation, -45.0, 45.0))
            self.agent_rot[1] += rotation

        elif action_type == 3:
            # Pitch (look up/down): [sin_angle, cos_angle, -]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            rotation = float(np.degrees(np.arctan2(sin_a, cos_a)))
            rotation = float(np.clip(rotation, -45.0, 45.0))
            self.agent_rot[0] += rotation

        elif action_type == 4:
            # Roll (tilt): [sin_angle, cos_angle, -]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            rotation = float(np.degrees(np.arctan2(sin_a, cos_a)))
            rotation = float(np.clip(rotation, -45.0, 45.0))
            self.agent_rot[2] += rotation

        elif action_type == 5:
            # Orient horizontal: [rotation_deg, left_dist, fwd_dist]
            rotation = float(action_params[0])
            left_dist = float(action_params[1])
            fwd_dist = float(action_params[2])
            self._orient_horizontal(rotation, fwd_dist, left_dist)

        elif action_type == 6:
            # Orient vertical: [rotation_deg, down_dist, fwd_dist]
            rotation = float(action_params[0])
            down_dist = float(action_params[1])
            fwd_dist = float(action_params[2])
            self._orient_vertical(rotation, fwd_dist, down_dist)

        elif action_type == 7:
            # Detach: params ignored, uses current goal
            if (
                hasattr(self, "_current_goal")
                and self._current_goal is not None
            ):
                self._detach_simple(goal_pose=self._current_goal)

        self.agent_rot = self._normalize_euler(self.agent_rot)
        return self.get_sensor_data()

    @property
    def supports_continuous(self) -> bool:
        """LightweightEnv supports continuous actions."""
        return True

    @property
    def supports_offline_retrain(self) -> bool:
        """Trimesh is fast enough for offline retrain."""
        return True

def is_on_same_cube_side(pos_a, pos_b, cube_side=42.0, atol=1e-5):
    """Checks whether two points lie on the same side of a cube (e.g., both at +X when x = +42).

    Args:
        pos_a: position of point A [x, y, z]
        pos_b: position of point B [x, y, z]
        cube_side: cube side coordinate value (e.g., 42 mm)
        atol: numerical error tolerance

    Returns:
        bool: True if the points are on the same side of the cube
    """
    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)

    # check each axis
    for i in range(3):
        # Both points must be on the edge (coordinate ≈ ±cube_side)
        if (abs(abs(pos_a[i]) - cube_side) <= atol and
            abs(abs(pos_b[i]) - cube_side) <= atol):

            # And the sign must match (both +42 or both -42)
            if np.sign(pos_a[i]) == np.sign(pos_b[i]):
                return True

    return False


def _is_reachable_by_surface(
    env: LightweightEnv,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> bool:
    """Check if start and goal are on the same side of the object.

    Uses surface normal directions relative to object centroid.
    For vertical surfaces (walls): normal pointing outward from center
    means external side, inward means internal side.
    For horizontal surfaces (bottom/top): normal pointing against
    up_direction means external (bottom exterior), normal pointing
    with up_direction means internal (bottom interior).
    For points far from surface: considered outside (nearest.on_surface
    is unreliable when far from object).

    Args:
        env: Environment with mesh.
        start_pos: Start position [x, y, z].
        goal_pos: Goal position [x, y, z].

    Returns:
        True if both points are on the same side.
    """
    center = np.array(env.mesh.centroid, dtype=float)
    height_axis = env.height_axis
    up = env.up_direction

    # Max wall thickness — points farther than this from surface
    # are reliably outside the object
    FAR_THRESHOLD = 10.0

    # Get normals at nearest surface points
    _, start_dist, start_face = env.mesh.nearest.on_surface([start_pos])
    _, goal_dist, goal_face = env.mesh.nearest.on_surface([goal_pos])

    start_normal = env.mesh.face_normals[start_face[0]]
    goal_normal = env.mesh.face_normals[goal_face[0]]

    # Horizontal components (ignore height axis)
    start_n = start_normal.copy()
    start_n[height_axis] = 0.0

    goal_n = goal_normal.copy()
    goal_n[height_axis] = 0.0

    # Direction from center to each point (horizontal)
    start_from_center = start_pos - center
    start_from_center[height_axis] = 0.0

    goal_from_center = goal_pos - center
    goal_from_center[height_axis] = 0.0

    # Determine side for start point
    if start_dist[0] > FAR_THRESHOLD:
        # Far from surface — reliably outside
        start_outward = True
    elif np.linalg.norm(start_n) >= 0.3:
        # Wall: outward = normal points same direction as center→point
        start_outward = np.dot(start_n, start_from_center) > 0
    else:
        # Horizontal surface (bottom/top): use vertical normal component
        start_outward = np.dot(start_normal, up) < 0

    # Determine side for goal point
    if goal_dist[0] > FAR_THRESHOLD:
        goal_outward = True
    elif np.linalg.norm(goal_n) >= 0.3:
        goal_outward = np.dot(goal_n, goal_from_center) > 0
    else:
        goal_outward = np.dot(goal_normal, up) < 0

    result = start_outward == goal_outward
    logger.debug(
        f"REACHABLE_CHECK: "
        f"start={[round(x,1) for x in start_pos.tolist()]}, "
        f"goal={[round(x,1) for x in goal_pos.tolist()]}, "
        f"start_normal={[round(x,3) for x in start_normal.tolist()]}, "
        f"goal_normal={[round(x,3) for x in goal_normal.tolist()]}, "
        f"start_outward={start_outward}, "
        f"goal_outward={goal_outward}, "
        f"result={result}"
    )
    return result
