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

"""Per-step debug logger for drone diagnostics.

Usage:
    python run_experiment.py configs/drone_vel_experiment.yaml --override logging.debug=true

Writes detailed CSV + console output for every step.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class StepLogger:
    """Logs every step for debugging obstacle avoidance and Q-learning."""

    def __init__(
        self,
        output_dir: str = "logs/debug",
        console_every: int = 100,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.console_every = console_every
        self._step = 0
        self._episode = 0
        self._csv_file = None
        self._csv_writer = None

        if enabled:
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def start_episode(self, episode_idx: int, goal_pos, start_pos):
        if not self.enabled:
            return
        self._episode = episode_idx
        self._step = 0

        csv_path = self._output_dir / f"episode_{episode_idx:04d}.csv"
        if self._csv_file:
            self._csv_file.close()
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "step", "phase",
            "pos_x", "pos_y", "pos_z",
            "goal_x", "goal_y", "goal_z",
            "distance", "speed", "vel_alignment",
            "path_blocked", "nearest_obs_dist",
            "nearest_obs_dx", "nearest_obs_dy", "nearest_obs_dz",
            "ray_0", "ray_1", "ray_2", "ray_3",
            "ray_4", "ray_5", "ray_6", "ray_7",
            "avoid_dx", "avoid_dy", "avoid_dz",
            "action_name", "action_idx",
            "h_best", "q_best",
            "q_agree_h", "nn_dist", "q_spread",
            "q_best_val", "h_best_val",
            "q_conf", "q_prox", "q_exp", "q_cons",
            "epsilon", "has_q_data",
            "stuck", "stuck_random",
            "reward", "cum_reward",
            "reason",
        ])

        print(f"\n{'='*80}")
        print(f"DEBUG Episode {episode_idx}")
        print(
            f"  Start: [{start_pos[0]:.2f}, "
            f"{start_pos[1]:.2f}, {start_pos[2]:.2f}]"
        )
        print(
            f"  Goal:  [{goal_pos[0]:.2f}, "
            f"{goal_pos[1]:.2f}, {goal_pos[2]:.2f}]"
        )
        print(f"{'='*80}")
        print(
            f"{'step':>5} {'phase':>6} "
            f"{'dist':>5} {'spd':>5} {'valign':>6} "
            f"{'blk':>4} {'obs_d':>5} {'rmin':>5} "
            f"{'action':>15} {'h_best':>15} {'q_best':>15} "
            f"{'ag':>2} {'nn_d':>5} {'qspr':>5} "
            f"{'conf':>5} {'prox':>5} {'exp':>5} {'cons':>5} "
            f"{'reward':>7} {'cum_r':>7} "
            f"{'reason'}"
        )
        print("-" * 170)

    def log_step(
        self,
        sensor_data: Dict[str, Any],
        state: np.ndarray,
        action_idx: int,
        explanation: Dict[str, Any],
        avoidance_dir: np.ndarray,
        flight_phase: int,
        reward: float,
        cum_reward: float,
        reason: str = "",
    ):
        if not self.enabled:
            return

        self._step += 1

        pos = sensor_data["position"]
        goal_delta = sensor_data["goal_delta_world"]
        goal_pos = [pos[i] + goal_delta[i] for i in range(3)]
        distance = sensor_data["distance_to_goal"]
        speed = sensor_data["speed"]
        vel_alignment = state[12]
        path_blocked = sensor_data.get("path_blocked", False)
        nearest_dist = sensor_data.get("nearest_obstacle_dist", 3.0)
        nearest_dir = sensor_data.get(
            "nearest_obstacle_dir_body", [0, 0, 0]
        )
        rays = sensor_data.get("obstacle_rays", [1.0] * 8)

        phase_name = "AVOID" if flight_phase == 1 else "GO"
        action_name = explanation.get("action", "?")
        h_best = explanation.get("h_best", "?")
        q_best = explanation.get("q_best", "?")
        q_agree_h = explanation.get("q_agree_h", False)
        nn_dist = explanation.get("nn_dist", 0.0)
        q_spread = explanation.get("q_spread", 0.0)
        q_best_val = explanation.get("q_best_val", 0.0)
        h_best_val = explanation.get("h_best_val", 0.0)
        q_conf = explanation.get("q_conf", 0.0)
        q_prox = explanation.get("q_prox", 0.0)
        q_exp = explanation.get("q_exp", 0.0)
        q_cons = explanation.get("q_cons", 0.0)
        epsilon = explanation.get("epsilon", 0.0)
        has_q_data = explanation.get("has_q_data", False)
        stuck = explanation.get("stuck", False)
        stuck_random = explanation.get("stuck_random", 0.0)

        show = (
            self._step % self.console_every == 0
            or self._step <= 3
            # or reason
        )

        # CSV — every step
        if self._csv_writer and show:
            self._csv_writer.writerow([
                self._step, phase_name,
                f"{pos[0]:.3f}", f"{pos[1]:.3f}", f"{pos[2]:.3f}",
                f"{goal_pos[0]:.3f}", f"{goal_pos[1]:.3f}",
                f"{goal_pos[2]:.3f}",
                f"{distance:.3f}", f"{speed:.4f}",
                f"{vel_alignment:.3f}",
                int(path_blocked), f"{nearest_dist:.3f}",
                f"{nearest_dir[0]:.3f}", f"{nearest_dir[1]:.3f}",
                f"{nearest_dir[2]:.3f}",
                *[f"{r:.3f}" for r in rays[:8]],
                f"{avoidance_dir[0]:.3f}", f"{avoidance_dir[1]:.3f}",
                f"{avoidance_dir[2]:.3f}",
                action_name, action_idx,
                h_best, q_best,
                int(q_agree_h), f"{nn_dist:.4f}", f"{q_spread:.2f}",
                f"{q_best_val:.2f}", f"{h_best_val:.2f}",
                f"{q_conf:.3f}", f"{q_prox:.3f}",
                f"{q_exp:.3f}", f"{q_cons:.3f}",
                f"{epsilon:.4f}", int(has_q_data),
                int(stuck), f"{stuck_random:.3f}",
                f"{reward:.3f}", f"{cum_reward:.2f}",
                reason,
            ])
            if self._step % 100 == 0:
                self._csv_file.flush()

        # Console
        if show:
            self._print_step_line(
                self._step, phase_name, flight_phase,
                distance, speed, vel_alignment,
                path_blocked, nearest_dist, rays,
                action_name, h_best, q_best,
                q_agree_h, nn_dist, q_spread,
                q_conf, q_prox, q_exp, q_cons,
                reward, cum_reward, reason,
            )

    def log_terminal(
        self,
        step: int,
        termination: str,
        sensor_data: Dict[str, Any],
        state: np.ndarray,
        flight_phase: int,
        reward: float,
        cum_reward: float,
    ):
        if not self.enabled:
            return

        distance = sensor_data["distance_to_goal"]
        speed = sensor_data["speed"]
        vel_alignment = state[12]
        path_blocked = sensor_data.get("path_blocked", False)
        nearest_dist = sensor_data.get("nearest_obstacle_dist", 3.0)
        rays = sensor_data.get("obstacle_rays", [1.0] * 8)
        phase_name = "AVOID" if flight_phase == 1 else "GO"

        label = termination.upper()
        self._print_step_line(
            step, phase_name, flight_phase,
            distance, speed, vel_alignment,
            path_blocked, nearest_dist, rays,
            f"[{label}]", "", "",
            False, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            reward, cum_reward, label,
        )

        if self._csv_writer:
            pos = sensor_data["position"]
            goal_delta = sensor_data["goal_delta_world"]
            goal_pos = [pos[i] + goal_delta[i] for i in range(3)]
            nearest_dir = sensor_data.get(
                "nearest_obstacle_dir_body", [0, 0, 0]
            )
            self._csv_writer.writerow([
                step, phase_name,
                f"{pos[0]:.3f}", f"{pos[1]:.3f}", f"{pos[2]:.3f}",
                f"{goal_pos[0]:.3f}", f"{goal_pos[1]:.3f}",
                f"{goal_pos[2]:.3f}",
                f"{distance:.3f}", f"{speed:.4f}",
                f"{vel_alignment:.3f}",
                int(path_blocked), f"{nearest_dist:.3f}",
                f"{nearest_dir[0]:.3f}", f"{nearest_dir[1]:.3f}",
                f"{nearest_dir[2]:.3f}",
                *[f"{r:.3f}" for r in rays[:8]],
                "0.000", "0.000", "0.000",
                f"[{label}]", -1,
                "", "",
                0, "0.0000", "0.00",
                "0.00", "0.00",
                "0.000", "0.000", "0.000", "0.000",
                "0.0000", 0,
                0, "0.000",
                f"{reward:.3f}", f"{cum_reward:.2f}",
                label,
            ])
            self._csv_file.flush()

    def log_phase_change(
        self, old_phase: int, new_phase: int, nearest_dist: float
    ):
        if not self.enabled:
            return
        old_name = "AVOID" if old_phase == 1 else "GO"
        new_name = "AVOID" if new_phase == 1 else "GO"
        print(
            f"  >>> PHASE CHANGE: {old_name} → {new_name} "
            f"(nearest_obs={nearest_dist:.2f}m)"
        )

    def log_stuck(self, steps_stuck: int, distance: float):
        if not self.enabled:
            return
        print(
            f"  >>> STUCK DETECTED: {steps_stuck} steps "
            f"at dist={distance:.2f}m"
        )

    def end_episode(
        self,
        result: str,
        total_steps: int,
        final_dist: float,
        cum_reward: float,
    ):
        if not self.enabled:
            return
        status = "✅" if result == "goal_reached" else "❌"
        print(f"\n{status} Episode {self._episode} ended: {result}")
        print(
            f"  Steps: {total_steps}, "
            f"Final dist: {final_dist:.3f}m, "
            f"Reward: {cum_reward:.1f}"
        )
        print(f"{'='*80}\n")

        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    @staticmethod
    def _print_step_line(
        step, phase_name, flight_phase,
        distance, speed, vel_alignment,
        path_blocked, nearest_dist, rays,
        action_name, h_best, q_best,
        q_agree_h, nn_dist, q_spread,
        q_conf, q_prox, q_exp, q_cons,
        reward, cum_reward, reason,
    ):
        rays_min = min(rays) if rays else 1.0
        blocked_str = "BLK" if path_blocked else " ok"
        phase_icon = "⚠️" if flight_phase == 1 else "  "
        agree_str = "✓" if q_agree_h else "✗"

        flag = ""
        if reason:
            flag = f" ← {reason}"

        print(
            f"{step:5d} {phase_icon}{phase_name:>5} "
            f"{distance:5.2f} {speed:5.3f} {vel_alignment:+6.3f} "
            f"{blocked_str:>4} {nearest_dist:5.2f} {rays_min:5.3f} "
            f"{action_name:>15} {h_best:>15} {q_best:>15} "
            f"{agree_str:>2} {nn_dist:5.2f} {q_spread:5.0f} "
            f"{q_conf:5.2f} {q_prox:5.2f} {q_exp:5.2f} {q_cons:5.2f} "
            f"{reward:+7.2f} {cum_reward:+7.1f}"
            f"{flag}"
        )
