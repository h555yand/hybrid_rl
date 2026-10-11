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

"""Drone RL Controller for VelocityAviary.

Q-learning controller with obstacle avoidance.
Two Q-stores: GO (navigation) and AVOID (obstacle avoidance)
with different feature weights for HNSW distance computation.

Action = [dir_x, dir_y, dir_z, speed_fraction].

State vector (28D):
    # Base (14D)
    goal_direction_body [3D]: unit direction to goal in body frame
    distance_to_goal    [1D]: scalar distance [m]
    velocity_body       [3D]: linear velocity in body frame [m/s]
    angular_velocity    [3D]: angular velocity [rad/s]
    altitude            [1D]: height [m]
    speed               [1D]: scalar speed [m/s]
    vel_alignment       [1D]: context-dependent alignment [-1, 1]
    altitude_error      [1D]: goal_z - drone_z [m]

    # Obstacle (14D)
    flight_phase        [1D]: 0=GO_TO_GOAL, 1=AVOID_OBSTACLE
    path_blocked        [1D]: 0.0 or 1.0
    nearest_obs_dist    [1D]: distance to nearest obstacle (clipped)
    nearest_obs_dir_body[3D]: direction to nearest obstacle (body frame)
    obstacle_rays       [8D]: normalized ray distances [0..1]
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from drone.drone_action_space_vel import DroneActionSpaceVel
from drone.drone_config_vel import DEFAULT_DRONE_CONFIG_VEL

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybrid_rl.core.hnsw_state_store import HNSWStateStore

logger = logging.getLogger(__name__)


class FlightPhase:
    GO_TO_GOAL = 0
    AVOID_OBSTACLE = 1


class RunningQStats:
    """Running Q-value statistics for confidence adjustment."""

    def __init__(self, warmup: int = 200, window: int = None):
        self.warmup = warmup
        self.window = window

        if window is not None:
            self._buffer = deque(maxlen=window)
            self._cached_mean = 0.0
            self._cached_std = 1.0
            self._cache_interval = max(window // 20, 100)
            self._updates_since_cache = 0
        else:
            self._buffer = None

        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, q_values: np.ndarray):
        if self._buffer is not None:
            for q in q_values:
                self._buffer.append(float(q))
            self.count = len(self._buffer)
            self._updates_since_cache += len(q_values)
            if (
                self._updates_since_cache >= self._cache_interval
                and self.count >= self.warmup
            ):
                self._recompute_cache()
        else:
            for q in q_values:
                self.count += 1
                delta = q - self.mean
                self.mean += delta / self.count
                delta2 = q - self.mean
                self.var += (delta * delta2 - self.var) / self.count

    def _recompute_cache(self):
        arr = np.array(self._buffer)
        self._cached_mean = float(arr.mean())
        self._cached_std = float(max(arr.std(), 1e-4))
        self.mean = self._cached_mean
        self.var = self._cached_std ** 2
        self._updates_since_cache = 0

    @property
    def std(self):
        if self._buffer is not None:
            return self._cached_std
        return max(np.sqrt(max(self.var, 0.0)), 1e-4)

    @property
    def is_warmed_up(self):
        return self.count >= self.warmup

    def normalize_value(self, v):
        if not self.is_warmed_up:
            return 0.0
        if self._buffer is not None:
            return (v - self._cached_mean) / self._cached_std
        return (v - self.mean) / self.std


class DroneControllerVel:
    """Q-learning controller for VelocityAviary drone.

    Uses two Q-stores with different feature weights:
    - q_store_go: optimized for navigation (goal_dir important)
    - q_store_avoid: optimized for obstacle avoidance (rays important)
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict] = None,
    ):
        self.config = {
            **DEFAULT_DRONE_CONFIG_VEL, **(config or {})
        }
        self.agent_id = agent_id
        self.mode = self.config.get("mode", "train")

        self.action_space = DroneActionSpaceVel(
            cruise_speed=self.config["cruise_speed"],
            slow_speed=self.config["slow_speed"],
        )
        self.num_actions = self.action_space.NUM_ACTIONS

        # Two Q-stores with phase-specific feature weights
        go_weights = self.config.get("feature_weights_go", None)
        avoid_weights = self.config.get("feature_weights_avoid", None)

        self.q_store_go = HNSWStateStore(
            config={
                **self.config,
                "feature_weights": go_weights,
            },
            name="drone_vel_go",
        )
        self.q_store_avoid = HNSWStateStore(
            config={
                **self.config,
                "feature_weights": avoid_weights,
            },
            name="drone_vel_avoid",
        )

        self.gamma = self.config["gamma"]
        self.alpha = self.config["alpha"]
        self.epsilon = self.config["epsilon_start"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config.get("epsilon_decay", 0.999)

        self._avoid_enter_dist = self.config.get("avoid_enter_dist", 1.5)
        self._avoid_exit_dist = self.config.get("avoid_exit_dist", 2.0)
        self._ray_max_dist = self.config.get("ray_max_dist", 3.0)
        self._ray_count = self.config.get("ray_count", 8)

        self._prev_state: Optional[np.ndarray] = None
        self._prev_sensor: Optional[Dict] = None
        self._last_action: Optional[int] = None
        self._steps: int = 0
        self._current_goal: Optional[np.ndarray] = None
        self._episode_reward: float = 0.0
        self._episode_transitions: List[Dict] = []
        self._distance_history: List[float] = []

        self._flight_phase: int = FlightPhase.GO_TO_GOAL
        self._avoidance_direction: np.ndarray = np.zeros(3)
        self._avoidance_side: Optional[str] = None
        self._phase_cooldown: int = 0
        self._phase_min_steps: int = self.config.get("phase_min_steps", 30)

        # Stuck tracking
        self._position_history: List[np.ndarray] = []
        self._stuck_steps: int = 0

        # Running Q statistics per phase
        self._running_q_stats_go = RunningQStats(
            warmup=self.config.get("running_q_warmup", 500),
            window=self.config.get("running_q_window", 10000),
        )
        self._running_q_stats_avoid = RunningQStats(
            warmup=self.config.get("running_q_warmup", 500),
            window=self.config.get("running_q_window", 10000),
        )

        self._total_episodes: int = 0
        self._total_steps: int = 0
        self._total_goals_reached: int = 0
        self._termination_counts: Dict[str, int] = {
            "goal_reached": 0, "collision": 0,
            "out_of_bounds": 0, "timeout": 0,
        }
        self._step_logger = None

    # ══════════════════════════════════════════════════
    # STORE SELECTION
    # ══════════════════════════════════════════════════

    def _select_store(self, state: np.ndarray) -> HNSWStateStore:
        """Select Q-store based on flight phase in state."""
        flight_phase = state[14]
        if flight_phase > 0.5:
            return self.q_store_avoid
        return self.q_store_go

    def _select_running_stats(
        self, state: np.ndarray
    ) -> RunningQStats:
        """Select running stats based on flight phase."""
        flight_phase = state[14]
        if flight_phase > 0.5:
            return self._running_q_stats_avoid
        return self._running_q_stats_go

    # ══════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════

    @property
    def is_training(self) -> bool:
        return self.mode in ("train", "adaptive")

    def set_new_goal(
        self, goal_pose: np.ndarray, start_pos: np.ndarray
    ):
        self._current_goal = goal_pose.copy()
        self._prev_state = None
        self._prev_sensor = None
        self._last_action = None
        self._steps = 0
        self._episode_reward = 0.0
        self._episode_transitions = []
        self._distance_history = []
        self._flight_phase = FlightPhase.GO_TO_GOAL
        self._avoidance_direction = np.zeros(3)
        self._avoidance_side = None
        self._phase_cooldown = 0
        self._position_history = []
        self._stuck_steps = 0
        self._total_episodes += 1

        if self.is_training:
            warmup = self.config.get("warmup_episodes", 30)
            if self._total_episodes > warmup:
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * self.epsilon_decay,
                )
        if self._step_logger:
            self._step_logger.start_episode(
                self._total_episodes, goal_pose[:3], start_pos,
            )

    def set_step_logger(self, step_logger):
        self._step_logger = step_logger

    def step(
        self, sensor_data: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[Dict]]:
        if self._current_goal is None:
            return None, None

        self._steps += 1
        self._total_steps += 1

        prev_phase = self._flight_phase
        self._update_flight_phase(sensor_data)

        if self._flight_phase != prev_phase and self._step_logger:
            self._step_logger.log_phase_change(
                prev_phase, self._flight_phase,
                sensor_data.get("nearest_obstacle_dist", 3.0),
            )

        self._avoidance_direction = self._compute_avoidance_direction(
            sensor_data
        )

        state = self._compute_state(sensor_data)
        done, termination = self._check_termination(sensor_data)

        self._update_stuck_counter()

        # Choose action BEFORE Q-update (for Expected SARSA)
        action_index = None
        explanation = None
        if not done:
            action_index, explanation = self._choose_action(
                state, sensor_data
            )

        # Learn from previous step
        reward = 0.0
        if self._prev_state is not None:
            reward = self._compute_reward(
                state, self._prev_state,
                self._last_action, sensor_data,
                self._prev_sensor, termination,
            )
            self._episode_reward += reward
            self._episode_transitions.append({
                "state": self._prev_state.copy(),
                "action": int(self._last_action),
                "reward": float(reward),
            })

            if self.is_training:
                if done:
                    td_target = reward
                else:
                    # Weighted Expected SARSA
                    next_store = self._select_store(state)
                    next_q = next_store.get_q_values(state)
                    action_probs = explanation["action_probs"]

                    max_next_q = float(np.max(next_q))
                    expected_next_q = float(
                        np.sum(action_probs * next_q)
                    )

                    beta = self.config.get("td_max_weight", 0.5)
                    blended_next_q = (
                        beta * max_next_q
                        + (1 - beta) * expected_next_q
                    )
                    td_target = (
                        reward + self.gamma * blended_next_q
                    )

                prev_store = self._select_store(self._prev_state)
                prev_store.update_q_value(
                    self._prev_state, self._last_action,
                    td_target, self._get_learning_rate(),
                )

        # Handle termination
        if done:
            if self._prev_state is None:
                reward = self._compute_terminal_reward(termination)
                self._episode_reward += reward

            if termination == "goal_reached" and self.is_training:
                self._apply_success_backup()

            if self._step_logger:
                self._step_logger.log_terminal(
                    step=self._steps,
                    termination=termination,
                    sensor_data=sensor_data,
                    state=state,
                    flight_phase=self._flight_phase,
                    reward=reward,
                    cum_reward=self._episode_reward,
                )

            self._on_episode_done(termination)
            return None, None

        # Log and return
        reason = ""
        if self._step_logger and self._stuck_steps > 0:
            reason = f"STUCK({self._stuck_steps}steps)"

        if self._step_logger:
            self._step_logger.log_step(
                sensor_data=sensor_data,
                state=state,
                action_idx=action_index,
                explanation=explanation,
                avoidance_dir=self._avoidance_direction,
                flight_phase=self._flight_phase,
                reward=reward,
                cum_reward=self._episode_reward,
                reason=reason,
            )

        self._prev_state = state
        self._prev_sensor = sensor_data
        self._last_action = action_index

        return action_index, explanation

    def _compute_terminal_reward(self, termination: str) -> float:
        cfg = self.config
        if termination == "goal_reached":
            return cfg["reward_goal_reached"]
        elif termination == "collision":
            return cfg["reward_collision"]
        elif termination == "out_of_bounds":
            return cfg["reward_out_of_bounds"]
        elif termination == "timeout":
            return cfg["reward_timeout"]
        return 0.0

    # ══════════════════════════════════════════════════
    # ACTION SELECTION
    # ══════════════════════════════════════════════════

    def _choose_action(
        self, state: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Tuple[int, Optional[Dict]]:
        store = self._select_store(state)
        running_stats = self._select_running_stats(state)

        q_values, confidence = (
            store.get_q_values_with_confidence(state)
        )
        heuristic = self._compute_heuristic(state, sensor_data)

        eps = self._get_current_epsilon()

        # Update running stats
        if np.max(np.abs(q_values)) > 1e-8:
            running_stats.update(q_values)

        # Confidence adjustment
        base_confidence = confidence["overall"]
        V = float(np.mean(q_values))
        v_boost_strength = self.config.get("v_boost_strength", 0.15)
        v_penalty_strength = self.config.get("v_penalty_strength", 0.3)
        v_penalty_max = self.config.get("v_penalty_max", 0.5)
        v_boost_max = self.config.get("v_boost_max", 0.3)

        if running_stats.is_warmed_up:
            V_norm = running_stats.normalize_value(V)
            if V_norm >= 0:
                boost = float(np.clip(
                    V_norm * v_boost_strength, 0.0, v_boost_max
                ))
                q_confidence = min(
                    base_confidence * (1.0 + boost), 0.95
                )
            else:
                penalty = float(np.clip(
                    -V_norm * v_penalty_strength, 0.0, v_penalty_max
                ))
                q_confidence = base_confidence * (1.0 - penalty)
        else:
            q_confidence = base_confidence

        q_scaled = q_values * q_confidence

        has_q = (
            store.next_id > 0
            and np.max(np.abs(q_scaled)) > 1e-6
        )

        q_norm = self._normalize_values(q_scaled)
        h_norm = self._normalize_values(heuristic)

        if has_q:
            combined = (1 - eps) * q_norm + eps * h_norm
            temperature = max(
                np.clip(0.5 * eps, 0.01, 0.5), 0.05
            )
        else:
            combined = h_norm.copy()
            temperature = 0.02

        if self._total_episodes <= self.config.get(
            "warmup_episodes", 30
        ):
            combined = h_norm.copy()
            temperature = 0.01

        v = combined / temperature
        v = v - np.max(v)
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()

        # Random exploration
        stuck_random = 0.0
        if self._is_stuck():
            cfg = self.config
            stuck_random = min(
                cfg.get("stuck_random_base", 0.15)
                + self._stuck_steps
                * cfg.get("stuck_random_growth", 0.001),
                cfg.get("stuck_random_max", 0.5),
            )

        base_random = 0.05 * eps
        total_random = max(base_random, stuck_random)

        if np.random.random() < total_random:
            action_index = np.random.randint(self.num_actions)
        else:
            action_index = int(
                np.random.choice(len(probs), p=probs)
            )

        # Diagnostics
        q_best_idx = int(np.argmax(q_values))
        h_best_idx = int(np.argmax(heuristic))

        nn_dist = 0.0
        if store.next_id > 0:
            norm_state = store._normalize(state)
            k = min(1, store.next_id)
            _, distances = store._index.knn_query(
                norm_state.reshape(1, -1), k=k
            )
            nn_dist = float(np.sqrt(max(distances[0][0], 0)))

        phase_name = (
            "AVOID"
            if self._flight_phase == FlightPhase.AVOID_OBSTACLE
            else "GO"
        )
        store_name = (
            "avoid" if state[14] > 0.5 else "go"
        )
        explanation = {
            "action": self.action_space.get_info(action_index).name,
            "q_best": self.action_space.get_info(q_best_idx).name,
            "h_best": self.action_space.get_info(h_best_idx).name,
            "epsilon": eps,
            "has_q_data": has_q,
            "distance": float(state[3]),
            "flight_phase": phase_name,
            "path_blocked": bool(state[15] > 0.5),
            "nearest_obs": float(state[16] * self._ray_max_dist),
            "stuck": self._is_stuck(),
            "stuck_random": stuck_random,
            "q_agree_h": q_best_idx == h_best_idx,
            "nn_dist": round(nn_dist, 4),
            "q_spread": round(
                float(np.max(q_values) - np.min(q_values)), 2
            ),
            "q_best_val": round(float(np.max(q_values)), 2),
            "h_best_val": round(float(np.max(heuristic)), 2),
            "q_conf": round(q_confidence, 3),
            "q_prox": round(confidence["proximity"], 3),
            "q_exp": round(confidence["experience"], 3),
            "q_cons": round(confidence["consistency"], 3),
            "v_norm": round(
                running_stats.normalize_value(V), 3
            ) if running_stats.is_warmed_up else 0.0,
            "store": store_name,
            "action_probs": probs,
        }

        return action_index, explanation

    # ══════════════════════════════════════════════════
    # BACKUP (success/collision) — use _select_store
    # ══════════════════════════════════════════════════

    def _apply_success_backup(self):
        if not self._episode_transitions:
            return
        cfg = self.config
        k = min(
            cfg.get("success_backup_steps", 50),
            len(self._episode_transitions),
        )
        tail = self._episode_transitions[-k:]
        base_alpha = (
            self._get_learning_rate()
            * cfg.get("success_backup_alpha_multiplier", 2.0)
        )
        g_return = 0.0
        for depth, tr in enumerate(reversed(tail)):
            g_return = tr["reward"] + self.gamma * g_return
            lr = base_alpha * (
                cfg.get("success_backup_lambda", 0.95) ** depth
            )
            store = self._select_store(tr["state"])
            store.update_q_value(
                tr["state"], tr["action"],
                g_return, lr, count_visit=False,
            )

    def _apply_collision_backup(self):
        if not self._episode_transitions:
            return
        cfg = self.config
        k = min(
            cfg.get("collision_backup_steps", 50),
            len(self._episode_transitions),
        )
        tail = self._episode_transitions[-k:]
        base_alpha = (
            self._get_learning_rate()
            * cfg.get("collision_backup_alpha_multiplier", 1.5)
        )
        g_return = 0.0
        for depth, tr in enumerate(reversed(tail)):
            g_return = tr["reward"] + self.gamma * g_return
            lr = base_alpha * (
                cfg.get("collision_backup_lambda", 0.95) ** depth
            )
            store = self._select_store(tr["state"])
            store.update_q_value(
                tr["state"], tr["action"],
                g_return, lr, count_visit=False,
            )

    # ══════════════════════════════════════════════════
    # HER — use _select_store
    # ══════════════════════════════════════════════════

    def _apply_hindsight_replay(self):
        goal_indices = self._find_virtual_goals()
        if not goal_indices:
            return

        cfg = self.config
        her_alpha = cfg.get("her_alpha", 0.3)
        her_backup = cfg.get("her_backup_steps", 15)
        her_goal_reward = cfg.get("her_goal_reward", 0.5)
        base_reward = cfg["reward_goal_reached"] * her_goal_reward

        start_dist = self._distance_history[0]

        for goal_idx in goal_indices:
            goal_progress = (
                start_dist - self._distance_history[goal_idx]
            )
            max_possible = max(start_dist, 0.1)
            quality = np.clip(
                goal_progress / max_possible, 0.0, 1.0
            )

            if quality < 0.05:
                continue

            scaled_reward = base_reward * quality

            g_return = scaled_reward
            k = min(her_backup, goal_idx)

            for depth in range(k):
                tr_idx = goal_idx - depth - 1
                if tr_idx < 0:
                    break
                tr = self._episode_transitions[tr_idx]
                g_return = (
                    tr["reward"] * 0.3 + self.gamma * g_return
                )
                lr = her_alpha * (0.9 ** depth)
                store = self._select_store(tr["state"])
                store.update_q_value(
                    tr["state"], tr["action"],
                    g_return, lr, count_visit=False,
                )

        logger.debug(
            "HER: replayed %d virtual goals from %d transitions",
            len(goal_indices), len(self._episode_transitions),
        )

    # ══════════════════════════════════════════════════
    # DEMONSTRATIONS — load into correct store
    # ══════════════════════════════════════════════════

    def load_demonstrations(self, demo_path: str):
        if not Path(demo_path).exists():
            logger.warning(
                "Demo file not found: %s", demo_path
            )
            return

        data = np.load(demo_path)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]

        cfg = self.config
        demo_alpha = cfg.get("demo_learning_rate", 0.8)
        demo_bonus = cfg.get("demo_reward_bonus", 0.5)

        n = len(actions)
        returns = np.zeros(n)
        g = 0.0
        for i in reversed(range(n)):
            g = rewards[i] + demo_bonus + self.gamma * g
            returns[i] = g

        go_count = 0
        avoid_count = 0
        for i in range(n):
            store = self._select_store(states[i])
            store.update_q_value(
                states[i], int(actions[i]),
                returns[i], demo_alpha,
                count_visit=True,
            )
            if states[i][14] > 0.5:
                avoid_count += 1
            else:
                go_count += 1

        logger.info(
            "Loaded %d demonstrations (%d GO, %d AVOID) from %s, "
            "Q-store GO=%d, AVOID=%d points",
            n, go_count, avoid_count, demo_path,
            self.q_store_go.next_id,
            self.q_store_avoid.next_id,
        )

    # ══════════════════════════════════════════════════
    # EPISODE LIFECYCLE
    # ══════════════════════════════════════════════════

    def _on_episode_done(self, termination: str):
        self._termination_counts[termination] = (
            self._termination_counts.get(termination, 0) + 1
        )
        if termination == "goal_reached":
            self._total_goals_reached += 1

        # Collision backup
        if termination == "collision" and self.is_training:
            self._apply_collision_backup()

        # HER
        if (
            self.config.get("do_her", False)
            and termination in ("timeout", "collision")
            and self.is_training
            and len(self._episode_transitions) > 20
        ):
            self._apply_hindsight_replay()

        logger.info(
            "Episode %d: %s in %d steps, "
            "reward=%.1f, eps=%.3f, rate=%d/%d, "
            "GO=%d pts, AVOID=%d pts",
            self._total_episodes,
            termination.upper(),
            self._steps,
            self._episode_reward,
            self._get_current_epsilon(),
            self._total_goals_reached,
            self._total_episodes,
            self.q_store_go.next_id,
            self.q_store_avoid.next_id,
        )

        self._current_goal = None
        self._prev_state = None
        self._prev_sensor = None
        self._last_action = None

        if self._step_logger:
            self._step_logger.end_episode(
                result=termination,
                total_steps=self._steps,
                final_dist=(
                    self._distance_history[-1]
                    if self._distance_history else 0
                ),
                cum_reward=self._episode_reward,
            )

    # ══════════════════════════════════════════════════
    # STATS & PERSISTENCE
    # ══════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        total = max(self._total_episodes, 1)
        return {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "total_goals_reached": self._total_goals_reached,
            "success_rate": self._total_goals_reached / total,
            "epsilon": self.epsilon,
            "q_store_go": self.q_store_go.get_stats(),
            "q_store_avoid": self.q_store_avoid.get_stats(),
            "termination_counts": dict(self._termination_counts),
        }

    def save(self, dirpath: str):
        pathlib.Path(dirpath).mkdir(exist_ok=True, parents=True)
        self.q_store_go.save_with_index(
            os.path.join(dirpath, "q_store_go")
        )
        self.q_store_avoid.save_with_index(
            os.path.join(dirpath, "q_store_avoid")
        )
        with open(os.path.join(dirpath, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        np.savez(
            os.path.join(dirpath, "controller_state.npz"),
            epsilon=np.array([self.epsilon]),
            total_episodes=np.array([self._total_episodes]),
            total_steps=np.array([self._total_steps]),
            total_goals_reached=np.array(
                [self._total_goals_reached]
            ),
        )

        # Save running Q stats
        np.savez(
            os.path.join(dirpath, "running_q_stats.npz"),
            go_mean=np.array([self._running_q_stats_go.mean]),
            go_var=np.array([self._running_q_stats_go.var]),
            go_count=np.array([self._running_q_stats_go.count]),
            avoid_mean=np.array([self._running_q_stats_avoid.mean]),
            avoid_var=np.array([self._running_q_stats_avoid.var]),
            avoid_count=np.array([self._running_q_stats_avoid.count]),
        )

        logger.info(
            "DroneControllerVel saved to %s "
            "(GO=%d pts, AVOID=%d pts)",
            dirpath,
            self.q_store_go.next_id,
            self.q_store_avoid.next_id,
        )

    @classmethod
    def load(
        cls, dirpath: str, agent_id: str, config=None
    ) -> "DroneControllerVel":
        with open(os.path.join(dirpath, "config.json")) as f:
            saved_config = json.load(f)
        cfg = {**saved_config, **(config or {})}
        controller = cls(agent_id=agent_id, config=cfg)

        # Load GO store
        go_path = os.path.join(dirpath, "q_store_go")
        if Path(go_path + ".npz").exists():
            go_cfg = {
                **cfg,
                "feature_weights": cfg.get(
                    "feature_weights_go", None
                ),
            }
            controller.q_store_go = HNSWStateStore.load_with_index(
                go_path, extra_cfg=go_cfg, name="drone_vel_go",
            )

        # Load AVOID store
        avoid_path = os.path.join(dirpath, "q_store_avoid")
        if Path(avoid_path + ".npz").exists():
            avoid_cfg = {
                **cfg,
                "feature_weights": cfg.get(
                    "feature_weights_avoid", None
                ),
            }
            controller.q_store_avoid = HNSWStateStore.load_with_index(
                avoid_path, extra_cfg=avoid_cfg,
                name="drone_vel_avoid",
            )

        # Backward compatibility: load old single store
        old_path = os.path.join(dirpath, "q_store_drone_vel")
        if (
            not Path(go_path + ".npz").exists()
            and Path(old_path + ".npz").exists()
        ):
            logger.info(
                "Loading legacy single Q-store into GO store"
            )
            controller.q_store_go = HNSWStateStore.load_with_index(
                old_path, extra_cfg=cfg, name="drone_vel_go",
            )

        state = np.load(
            os.path.join(dirpath, "controller_state.npz")
        )
        controller.epsilon = float(state["epsilon"][0])
        controller._total_episodes = int(
            state["total_episodes"][0]
        )
        controller._total_steps = int(state["total_steps"][0])
        controller._total_goals_reached = int(
            state["total_goals_reached"][0]
        )

        # Load running Q stats
        rq_path = os.path.join(dirpath, "running_q_stats.npz")
        if Path(rq_path).exists():
            try:
                rq = np.load(rq_path, allow_pickle=False)
                controller._running_q_stats_go.mean = float(
                    rq["go_mean"][0]
                )
                controller._running_q_stats_go.var = float(
                    rq["go_var"][0]
                )
                controller._running_q_stats_go.count = int(
                    rq["go_count"][0]
                )
                controller._running_q_stats_avoid.mean = float(
                    rq["avoid_mean"][0]
                )
                controller._running_q_stats_avoid.var = float(
                    rq["avoid_var"][0]
                )
                controller._running_q_stats_avoid.count = int(
                    rq["avoid_count"][0]
                )
            except Exception as exc:
                logger.warning(
                    "Could not load running Q stats: %s", exc
                )

        # Warmup running stats from loaded stores
        controller._warmup_running_q_stats()

        return controller

    def _warmup_running_q_stats(self):
        """Warm up RunningQStats from loaded Q-store points."""
        max_warmup = 10000

        for store, stats, name in [
            (self.q_store_go, self._running_q_stats_go, "GO"),
            (self.q_store_avoid, self._running_q_stats_avoid, "AVOID"),
        ]:
            if not store.points:
                continue

            pids = list(store.points.keys())
            if len(pids) > max_warmup:
                pids = list(np.random.choice(
                    pids, size=max_warmup, replace=False
                ))

            for pid in pids:
                stats.update(store.points[pid].q_values)

            logger.info(
                "RunningQStats warmup %s: mean=%.3f, std=%.3f, n=%d",
                name, stats.mean, stats.std, stats.count,
            )

    # ══════════════════════════════════════════════════
    # STUCK DETECTION
    # ══════════════════════════════════════════════════
    def _update_stuck_counter(self):
        """Stuck = moved a lot but got nowhere.

        Ratio of net displacement to path length.
        Straight line: ratio ~ 1.0
        Oscillation:   ratio ~ 0.0
        Hovering:      not stuck (min_path filter)
        """
        cfg = self.config
        window = cfg.get("stuck_window", 200)
        efficiency_threshold = cfg.get(
            "stuck_efficiency_threshold", 0.15
        )
        min_path_length = cfg.get("stuck_min_path", 0.5)

        n = len(self._position_history)
        if n < window:
            self._stuck_steps = 0
            return

        recent = self._position_history[-window:]
        pos_array = np.array(recent)

        net_displacement = np.linalg.norm(
            pos_array[-1] - pos_array[0]
        )

        step_deltas = np.diff(pos_array, axis=0)
        step_distances = np.linalg.norm(step_deltas, axis=1)
        path_length = np.sum(step_distances)

        if path_length < min_path_length:
            self._stuck_steps = 0
            return

        efficiency = net_displacement / path_length

        if efficiency < efficiency_threshold:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

    def _is_stuck(self) -> bool:
        return self._stuck_steps > 0

    # ══════════════════════════════════════════════════
    # FLIGHT PHASE
    # ══════════════════════════════════════════════════

    def _update_flight_phase(self, sensor_data: Dict) -> None:
        path_blocked = sensor_data.get("path_blocked", False)
        nearest_dist = sensor_data.get(
            "nearest_obstacle_dist", self._ray_max_dist
        )
        obstacle_rays = sensor_data.get("obstacle_rays", [])
        min_ray = min(obstacle_rays) if obstacle_rays else 1.0
        min_ray_dist = min_ray * self._ray_max_dist

        if self._phase_cooldown > 0:
            self._phase_cooldown -= 1

        if self._flight_phase == FlightPhase.GO_TO_GOAL:
            enter_blocked = (
                path_blocked
                and nearest_dist < self._avoid_enter_dist
            )
            enter_proximity = (
                nearest_dist
                < self._avoid_enter_dist
                * self.config.get("avoid_proximity_ratio", 0.5)
            )
            if enter_blocked or enter_proximity:
                self._flight_phase = FlightPhase.AVOID_OBSTACLE
                self._phase_cooldown = self._phase_min_steps
                self._avoidance_side = self._choose_avoidance_side(
                    sensor_data
                )

        elif self._flight_phase == FlightPhase.AVOID_OBSTACLE:
            if self._phase_cooldown <= 0:
                if not path_blocked and min_ray_dist > self._avoid_exit_dist:
                    self._flight_phase = FlightPhase.GO_TO_GOAL

    def _compute_avoidance_direction(
        self, sensor_data: Dict
    ) -> np.ndarray:
        cfg = self.config
        goal_dir = np.array(
            sensor_data.get("goal_direction_body", [1, 0, 0]),
            dtype=float,
        )
        nearest_obs_dir = np.array(
            sensor_data.get("nearest_obstacle_dir_body", [0, 0, 0]),
            dtype=float,
        )
        obstacle_rays = np.array(
            sensor_data.get(
                "obstacle_rays", [1.0] * self._ray_count
            ),
            dtype=float,
        )

        attract_strength = cfg.get("potential_field_attract", 0.3)
        repulse_strength = cfg.get("potential_field_repulse", 1.0)
        tangent_strength = cfg.get("potential_field_tangent", 0.8)

        attract = goal_dir * attract_strength

        repulse = np.zeros(3)
        ray_angles = np.linspace(
            0, 2 * np.pi, self._ray_count, endpoint=False
        )
        for i, angle in enumerate(ray_angles):
            ray_dir_body = np.array([
                np.cos(angle), np.sin(angle), 0.0
            ])
            ray_dist_norm = obstacle_rays[i]
            if ray_dist_norm < 1.0:
                strength = (1.0 - ray_dist_norm) * repulse_strength
                repulse -= ray_dir_body * strength

        obs_norm = np.linalg.norm(nearest_obs_dir[:2])
        if obs_norm > 1e-6:
            obs_2d = nearest_obs_dir.copy()
            obs_2d[2] = 0.0
            obs_2d /= np.linalg.norm(obs_2d)

            tangent_cw = np.array([obs_2d[1], -obs_2d[0], 0.0])
            tangent_ccw = np.array([-obs_2d[1], obs_2d[0], 0.0])

            if self._avoidance_side is not None:
                tangent = (
                    tangent_cw if self._avoidance_side == "cw"
                    else tangent_ccw
                )
            else:
                dot_cw = np.dot(tangent_cw, goal_dir[:3])
                dot_ccw = np.dot(tangent_ccw, goal_dir[:3])
                tangent = (
                    tangent_cw if dot_cw > dot_ccw else tangent_ccw
                )

            tangent *= tangent_strength
        else:
            tangent = np.zeros(3)

        if np.mean(obstacle_rays) < 0.3:
            repulse[2] += repulse_strength * 0.5

        combined = attract + repulse + tangent
        norm = np.linalg.norm(combined)
        if norm > 1e-6:
            return combined / norm
        return np.array([0.0, 0.0, 1.0])
    
    def _choose_avoidance_side(self, sensor_data: Dict) -> str:
        goal_dir = np.array(
            sensor_data.get("goal_direction_body", [1, 0, 0]),
            dtype=float,
        )
        nearest_obs_dir = np.array(
            sensor_data.get("nearest_obstacle_dir_body", [0, 0, 0]),
            dtype=float,
        )
        obs_2d = nearest_obs_dir.copy()
        obs_2d[2] = 0.0
        n = np.linalg.norm(obs_2d)
        if n > 1e-6:
            obs_2d /= n
        tangent_cw = np.array([obs_2d[1], -obs_2d[0], 0.0])
        tangent_ccw = np.array([-obs_2d[1], obs_2d[0], 0.0])
        if np.dot(tangent_cw, goal_dir) >= np.dot(tangent_ccw, goal_dir):
            return "cw"
        return "ccw"

    # ══════════════════════════════════════════════════
    # STATE (28D)
    # ══════════════════════════════════════════════════

    def _compute_state(
        self, sensor_data: Dict[str, Any]
    ) -> np.ndarray:
        goal_dir = np.array(
            sensor_data["goal_direction_body"], dtype=float
        )
        distance = float(sensor_data["distance_to_goal"])
        vel_body = np.array(
            sensor_data["velocity_body"], dtype=float
        )
        ang_vel = np.array(
            sensor_data["angular_velocity"], dtype=float
        )
        altitude = float(sensor_data["altitude"])
        speed = float(sensor_data["speed"])

        delta_world = np.array(
            sensor_data["goal_delta_world"], dtype=float
        )
        altitude_error = float(delta_world[2])

        if self._flight_phase == FlightPhase.AVOID_OBSTACLE:
            if speed > 1e-4:
                vel_dir_body = vel_body / speed
                vel_alignment = float(
                    np.dot(vel_dir_body, self._avoidance_direction)
                )
            else:
                vel_alignment = 0.0
        else:
            vel_alignment = float(sensor_data["vel_alignment"])

        path_blocked = (
            1.0 if sensor_data.get("path_blocked", False) else 0.0
        )
        nearest_obs_dist = float(sensor_data.get(
            "nearest_obstacle_dist", self._ray_max_dist
        ))
        nearest_obs_dist_norm = min(
            nearest_obs_dist / self._ray_max_dist, 1.0
        )
        nearest_obs_dir = np.array(
            sensor_data.get("nearest_obstacle_dir_body", [0, 0, 0]),
            dtype=float,
        )
        obstacle_rays = np.array(
            sensor_data.get(
                "obstacle_rays", [1.0] * self._ray_count
            ),
            dtype=float,
        )

        self._distance_history.append(distance)
        self._position_history.append(
            np.array(sensor_data["position"], dtype=float)
        )

        return np.array([
            goal_dir[0], goal_dir[1], goal_dir[2],
            distance,
            vel_body[0], vel_body[1], vel_body[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
            altitude,
            speed,
            vel_alignment,
            altitude_error,
            float(self._flight_phase),
            path_blocked,
            nearest_obs_dist_norm,
            nearest_obs_dir[0], nearest_obs_dir[1],
            nearest_obs_dir[2],
            obstacle_rays[0], obstacle_rays[1],
            obstacle_rays[2], obstacle_rays[3],
            obstacle_rays[4], obstacle_rays[5],
            obstacle_rays[6], obstacle_rays[7],
        ], dtype=float)

    # ══════════════════════════════════════════════════
    # TERMINATION
    # ══════════════════════════════════════════════════

    def _check_termination(
        self, sensor_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        if sensor_data["collision"]:
            return True, "collision"
        if sensor_data["out_of_bounds"]:
            return True, "out_of_bounds"
        if sensor_data["distance_to_goal"] < self.config[
            "goal_reach_distance"
        ]:
            return True, "goal_reached"
        if self._steps >= self.config["max_steps_per_goal"]:
            return True, "timeout"
        return False, None

    # ══════════════════════════════════════════════════
    # REWARD
    # ══════════════════════════════════════════════════

    def _compute_reward(
        self, state, prev_state, action,
        sensor_data, prev_sensor, termination,
    ) -> float:
        cfg = self.config
        reward = 0.0

        distance = state[3]
        prev_distance = prev_state[3]
        vel_alignment = state[12]
        speed = state[11]
        flight_phase = int(state[14])
        nearest_obs_dist_norm = state[16]
        nearest_obs_dist = nearest_obs_dist_norm * self._ray_max_dist

        if flight_phase == FlightPhase.GO_TO_GOAL:
            progress = prev_distance - distance
            reward += progress * cfg["reward_progress"]

            if speed > 0.01:
                reward += vel_alignment * cfg["reward_alignment"]

            reward += cfg["reward_step_penalty"]

            if distance < 0.5 and speed < 0.05:
                reward += cfg["reward_hover_bonus"]

            if nearest_obs_dist < self._avoid_enter_dist:
                proximity_penalty = (
                    (self._avoid_enter_dist - nearest_obs_dist)
                    / self._avoid_enter_dist
                )
                reward += (
                    proximity_penalty
                    * cfg["reward_obstacle_proximity"] * 0.3
                )

        elif flight_phase == FlightPhase.AVOID_OBSTACLE:
            # Proximity penalty
            if nearest_obs_dist < self._avoid_enter_dist:
                proximity_penalty = (
                    (self._avoid_enter_dist - nearest_obs_dist)
                    / self._avoid_enter_dist
                )
                reward += (
                    proximity_penalty
                    * cfg["reward_obstacle_proximity"]
                )

            # Avoidance alignment
            if speed > 0.01:
                reward += vel_alignment * cfg["reward_avoid_alignment"]

            # Weak goal progress (0.0 = don't penalize detours)
            progress = prev_distance - distance
            reward += (
                progress * cfg["reward_progress"]
                * cfg.get("reward_progress_in_avoid", 0.0)
            )

            # Step penalty
            reward += cfg["reward_step_penalty"]

            # Reward for moving away from obstacle
            prev_nearest = prev_state[16] * self._ray_max_dist
            curr_nearest = nearest_obs_dist
            obstacle_progress = curr_nearest - prev_nearest
            if obstacle_progress > 0:
                reward += obstacle_progress * cfg.get(
                    "reward_obstacle_progress", 3.0
                )

            # Bonus for clearing obstacle
            prev_phase = int(prev_state[14])
            if prev_phase == FlightPhase.AVOID_OBSTACLE:
                if not sensor_data.get("path_blocked", False):
                    reward += cfg.get(
                        "reward_obstacle_cleared", 10.0
                    )

        # Oscillation penalty
        if len(self._distance_history) > 10:
            recent = self._distance_history[-10:]
            diffs = np.diff(recent)
            if len(diffs) > 1:
                signs = np.sign(diffs)
                sign_changes = np.sum(
                    np.abs(np.diff(signs)) > 0
                )
                if sign_changes > cfg.get(
                    "oscillation_threshold", 5
                ):
                    reward += cfg.get(
                        "reward_oscillation_penalty", -0.3
                    )

        # Progressive stuck penalty
        if self._is_stuck():
            stuck_severity = min(
                self._stuck_steps
                / cfg.get("stuck_max_steps", 200),
                1.0,
            )
            reward += (
                cfg.get("reward_stuck_penalty", -0.5)
                * stuck_severity
            )

        # Terminal rewards
        if termination == "goal_reached":
            reward += cfg["reward_goal_reached"]
        elif termination == "collision":
            reward += cfg["reward_collision"]
        elif termination == "out_of_bounds":
            reward += cfg["reward_out_of_bounds"]
        elif termination == "timeout":
            reward += cfg["reward_timeout"]

        # Altitude safety
        altitude = state[10]
        if altitude < 0.15:
            reward += cfg["reward_altitude_penalty"]

        return reward

    def _compute_heuristic(
        self, state: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> np.ndarray:
        bias = np.zeros(self.num_actions, dtype=float)
        cfg = self.config

        distance = state[3]
        altitude = state[10]
        flight_phase = int(state[14])

        S = cfg["heuristic_approach_strength"]
        S_alt = cfg["heuristic_altitude_strength"]
        S_stab = cfg["heuristic_stabilize_strength"]
        S_avoid = cfg.get("heuristic_avoid_strength", 10.0)

        AS = DroneActionSpaceVel

        if flight_phase == FlightPhase.GO_TO_GOAL:
            delta_world = np.array(
                sensor_data.get("goal_delta_world", [0, 0, 0]),
                dtype=float,
            )
            dist = max(np.linalg.norm(delta_world), 0.01)
            direction = delta_world / dist
            dx, dy, dz = direction

            if abs(dx) > abs(dy):
                if dx > 0.1:
                    bias[AS.IDX_FORWARD] += S * abs(dx)
                elif dx < -0.1:
                    bias[AS.IDX_BACKWARD] += S * abs(dx)
            else:
                if dy > 0.1:
                    bias[AS.IDX_RIGHT] += S * abs(dy)
                elif dy < -0.1:
                    bias[AS.IDX_LEFT] += S * abs(dy)

            if dx > 0.3 and dy > 0.3:
                bias[AS.IDX_FORWARD_RIGHT] += S * 0.8
            elif dx > 0.3 and dy < -0.3:
                bias[AS.IDX_FORWARD_LEFT] += S * 0.8
            elif dx > 0.3 and dz > 0.3:
                bias[AS.IDX_FORWARD_UP] += S * 0.8
            elif dx > 0.3 and dz < -0.3:
                bias[AS.IDX_FORWARD_DOWN] += S * 0.8

            if dz > 0.3:
                bias[AS.IDX_UP] += S_alt * abs(dz)
            elif dz < -0.3:
                bias[AS.IDX_DOWN] += S_alt * abs(dz)

            if dist < 0.5:
                bias[AS.IDX_HOVER] += S_stab * 3.0

        elif flight_phase == FlightPhase.AVOID_OBSTACLE:
            avoid_dir = self._avoidance_direction
            ax, ay, az = avoid_dir

            if abs(ax) > abs(ay):
                if ax > 0.1:
                    bias[AS.IDX_FORWARD] += S_avoid * abs(ax)
                elif ax < -0.1:
                    bias[AS.IDX_BACKWARD] += S_avoid * abs(ax)
            else:
                if ay > 0.1:
                    bias[AS.IDX_RIGHT] += S_avoid * abs(ay)
                elif ay < -0.1:
                    bias[AS.IDX_LEFT] += S_avoid * abs(ay)

            if ax > 0.3 and ay > 0.3:
                bias[AS.IDX_FORWARD_RIGHT] += S_avoid * 0.8
            elif ax > 0.3 and ay < -0.3:
                bias[AS.IDX_FORWARD_LEFT] += S_avoid * 0.8
            elif ax < -0.3 and ay > 0.3:
                bias[AS.IDX_RIGHT] += S_avoid * 0.6
            elif ax < -0.3 and ay < -0.3:
                bias[AS.IDX_LEFT] += S_avoid * 0.6

            if az > 0.3:
                bias[AS.IDX_UP] += S_avoid * abs(az)
                if ax > 0.2:
                    bias[AS.IDX_FORWARD_UP] += S_avoid * 0.7
            elif az < -0.3:
                bias[AS.IDX_DOWN] += S_avoid * abs(az)
                if ax > 0.2:
                    bias[AS.IDX_FORWARD_DOWN] += S_avoid * 0.7

            nearest_dist_norm = state[16]
            if nearest_dist_norm < 0.3:
                bias *= 0.5
                bias[AS.IDX_HOVER] += S_stab

            if np.max(bias) > 0:
                bias[AS.IDX_HOVER] = min(
                    bias[AS.IDX_HOVER],
                    np.max(bias) * 0.3,
                )

        if altitude < 0.1:
            bias[AS.IDX_UP] += S_alt * 5.0
            bias[AS.IDX_DOWN] = 0.0
            bias[AS.IDX_FORWARD_DOWN] = 0.0

        return bias

    # ══════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════

    def _get_current_epsilon(self) -> float:
        if not self.is_training:
            return self.config["eval_epsilon"]

        base_eps = self.epsilon

        if self._is_stuck():
            return max(base_eps * 0.5, self.epsilon_min)

        return base_eps

    def _get_learning_rate(self) -> float:
        if self.is_training:
            return self.alpha
        return self.alpha * self.config["eval_alpha_multiplier"]

    @staticmethod
    def _normalize_values(values: np.ndarray) -> np.ndarray:
        v_min, v_max = np.min(values), np.max(values)
        v_range = v_max - v_min
        if v_range < 1e-8:
            return np.zeros_like(values)
        return 2.0 * (values - v_min) / v_range - 1.0

    # ══════════════════════════════════════════════════
    # HER (Hindsight Experience Replay)
    # ══════════════════════════════════════════════════

    def _find_virtual_goals(self) -> List[int]:
        """Find trajectory points where drone made real progress.

        Scores each point by:
          - goal_progress (primary): did distance to goal decrease?
          - displacement (secondary): did drone move from start?

        Returns indices of best points per segment.
        Only points with positive score (real progress) qualify.
        """
        transitions = self._episode_transitions
        positions = self._position_history
        distances = self._distance_history

        n = min(len(transitions), len(positions), len(distances))
        if n < 20:
            return []

        cfg = self.config
        n_virtual = cfg.get("her_virtual_goals", 3)
        her_backup = cfg.get("her_backup_steps", 15)

        start_dist = distances[0]
        start_pos = positions[0]

        segment_size = n // (n_virtual + 1)
        goal_indices = []

        for seg in range(n_virtual):
            seg_start = (
                (seg + 1) * segment_size - segment_size // 2
            )
            seg_end = min(seg_start + segment_size, n)
            if seg_start >= n:
                break

            best_idx = -1
            best_score = -999.0

            for i in range(seg_start, seg_end):
                # Primary: progress toward goal
                goal_progress = start_dist - distances[i]

                # Secondary: displacement from start
                displacement = np.linalg.norm(
                    positions[i] - start_pos
                )

                # Combined score
                score = goal_progress * 2.0 + displacement * 0.5

                # Strong penalty for moving away from goal
                if goal_progress < 0:
                    score += goal_progress * 3.0

                if score > best_score:
                    best_score = score
                    best_idx = i

            # Only take points with positive score and enough
            # history for backup
            if best_idx > her_backup and best_score > 0:
                goal_indices.append(best_idx)

        return goal_indices
