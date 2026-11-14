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

"""AdaptiveTrainingManager: monitors performance and decides
when/how to train for new objects.

Modes:
  - mastered: success_rate > 80%, light tuning (Q alpha*0.1, eps=0.02)
  - online: 50-80%, full Q learning + periodic SAC updates
    (critic CQL + actor every 10th step with strong BC),
    adaptive epsilon based on success rate
  - offline: < 50% sustained, Q retrain (500 ep, eps 1→0.3)
    + SAC retrain via PSACTrainer.train(300 ep)
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .ablation_runner import run_episodes
from .arbitrator import Arbitrator
from .experience_extractor import ExperienceExtractor
from .lightweight_env import LightweightEnv
from .sac_trainer import PSACTrainer
from .rl_goal_approach_controller import (
    RLGoalApproachController,
    RunningQStats,
)

logger = logging.getLogger(__name__)


class AdaptiveTrainingManager:

    def __init__(
        self,
        controller: RLGoalApproachController,
        env: LightweightEnv,
        config: Dict[str, Any],
        runs_dir: str,
        mesh_path: str,
        mastered_threshold: float = 0.95,
        offline_threshold: float = 0.40,
        monitor_window: int = 100,
        online_sac_update_every: int = 100,
        online_sac_update_steps: int = 40,
        offline_q_episodes: int = 500,
        offline_sac_episodes: int = 300,
        post_offline_cooldown: int = 200,
        min_online_before_offline: int = 300,
        max_offline_iterations: int = 2,
        max_bc_transitions: int = 5000,
        epsilon_warmup_episodes: int = 20,
        epsilon_default: float = 0.15,
        q_save_dir: str = None,
        sac_save_dir: str = None,
        offline_check_window: int = 50,
        promote_threshold: float = 0.70,
        promote_window: int = 100,
        enable_online_sac_updates: bool = True,
        enable_offline_sac_retrain: bool = True,
    ):
        self.controller = controller
        self.env = env
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.mesh_path = mesh_path
        self.q_save_dir = q_save_dir or str(Path(runs_dir) / "adaptive_q")
        self.sac_save_dir = sac_save_dir or str(Path(runs_dir) / "adaptive_sac")

        # Mode thresholds
        self.mastered_threshold = mastered_threshold
        self.offline_threshold = offline_threshold
        self.monitor_window = monitor_window
        self.offline_check_window = offline_check_window
        self.promote_threshold = promote_threshold
        self.promote_window = promote_window

        # Online SAC updates
        self.online_sac_update_every = online_sac_update_every
        self.online_sac_update_steps = online_sac_update_steps

        # Offline retrain params
        self.offline_q_episodes = offline_q_episodes
        self.offline_q_epsilon_start = 1.0
        self.offline_q_epsilon_min = 0.3
        self.offline_sac_episodes = offline_sac_episodes
        self.post_offline_cooldown = post_offline_cooldown
        self.min_online_before_offline = min_online_before_offline
        self.max_offline_iterations = max_offline_iterations

        # BC data management
        self.max_bc_transitions = max_bc_transitions
        self.bc_transitions: List = []

        # Adaptive epsilon
        self.epsilon_warmup_episodes = epsilon_warmup_episodes
        self.epsilon_default = epsilon_default

        # Performance tracking
        self.success_history: deque = deque(maxlen=monitor_window)
        self.online_transitions_raw: List = []
        self.online_episodes_since_sac_update = 0
        self.total_episodes = 0
        self.total_sac_updates = 0
        self.total_offline_iterations = 0
        self._episodes_since_offline = 0
        self.mode = "online"

        # Components
        self.sac_trainer: PSACTrainer | None = None
        self.action_space = controller.action_space if controller is not None else None
        mesh_name = Path(mesh_path).stem if mesh_path else "unknown"
        self.extractor = ExperienceExtractor(
            config=config, mesh_name=mesh_name
        )
        if controller is not None:
            self.arbitrator = Arbitrator(
                controller=controller,
                temperature_override=config.get("temperature_override", None)
            )
        else:
            self.arbitrator = None
        self.mode_changes: List[Dict[str, Any]] = []
        self._level_success_history: Dict[int, deque] = {}
        self._offline_just_completed = False
        self._online_critic_warmup_updates = 0
        self._critic_warmup_threshold = 3
        self.enable_online_sac_updates = enable_online_sac_updates
        self.enable_offline_sac_retrain = enable_offline_sac_retrain

    @property
    def success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(
            self.success_history
        )

    def sleep_sac_consolidation(
        self,
        object_name: str,
        success_trails: List[List[Dict[str, Any]]],
        failure_trails: List[List[Dict[str, Any]]],
        level: int = 0,
        sac_update_steps: int = 3000,
        critic_warmup_steps: int = 200,
        actor_update_every: int = 5,
        actor_lr_scale: float = 0.3,
        bc_lambda_init: float = 3.0,
        bc_lambda_decay: float = 0.999,
        bc_lambda_min: float = 0.3,
        use_cql: bool = True,
        cql_alpha: float = 0.5,
        current_mesh_ratio: float = 0.7,
        bc_ratio: float = 0.2,
        elite_ratio: float = 0.1,
    ) -> Dict[str, Any]:
        """Offline SAC consolidation using accumulated trails.

        Like _maybe_online_sac_update() but:
        - More SGD steps (3000 vs 40)
        - More aggressive actor updates (every 5th vs 10th)
        - Higher actor lr (0.3x vs 0.1x)
        - Faster critic warmup (200 steps vs 300 episodes)
        - Uses all accumulated trails, not just recent

        Actor learns from success trails (BC).
        Critic learns from both success and failure (replay buffer).
        """
        if self.sac_trainer is None:
            logger.warning("Sleep SAC skipped: no SAC trainer loaded")
            return {"status": "skipped", "reason": "no_sac_trainer"}

        if not success_trails:
            logger.warning("Sleep SAC skipped: no success trails")
            return {"status": "skipped", "reason": "no_success_trails"}

        mesh_name = object_name

        # ═══ 1. Convert trails to PSACTransition ═══
        success_psac = self.extractor.convert_all_trajectories(
            success_trails, level=level
        )
        failure_psac = self.extractor.convert_all_trajectories(
            failure_trails, level=level
        ) if failure_trails else []

        logger.info(
            "Sleep SAC: %d success transitions from %d trails, "
            "%d failure transitions from %d trails",
            len(success_psac), len(success_trails),
            len(failure_psac), len(failure_trails),
        )

        # ═══ 2. Normalize and load into buffer ═══
        all_psac = success_psac + failure_psac
        all_normalized = self.sac_trainer._normalize_bc_transitions(all_psac)

        self.sac_trainer.buffer.set_current_mesh(mesh_name)
        added = 0
        for tr in all_normalized:
            if tr.next_state is None:
                continue
            self.sac_trainer.buffer.add(
                state=tr.state,
                action_type=tr.action_type,
                action_params=tr.action_params,
                reward=tr.reward,
                next_state=tr.next_state,
                done=tr.done,
                mesh_name=mesh_name,
            )
            added += 1

        # ═══ 3. Update BC data (success only → actor BC loss) ═══
        success_normalized = self.sac_trainer._normalize_bc_transitions(
            success_psac
        )
        if self.sac_trainer.bc_data is not None:
            combined_bc = self.sac_trainer.bc_data + list(success_normalized)
            if len(combined_bc) > self.max_bc_transitions * 2:
                # Keep all new + sample from old
                old_count = len(self.sac_trainer.bc_data)
                keep_old = min(old_count, self.max_bc_transitions)
                old_indices = np.random.choice(old_count, keep_old, replace=False)
                combined_bc = (
                    [self.sac_trainer.bc_data[i] for i in old_indices]
                    + list(success_normalized)
                )
            self.sac_trainer.bc_data = combined_bc
        else:
            self.sac_trainer.bc_data = list(success_normalized)

        if len(self.sac_trainer.buffer) < self.sac_trainer.batch_size:
            logger.warning(
                "Sleep SAC skipped: buffer too small (%d < %d)",
                len(self.sac_trainer.buffer), self.sac_trainer.batch_size,
            )
            return {"status": "skipped", "reason": "buffer_too_small", "added": added}

        # ═══ 4. Save original settings ═══
        old_use_cql = self.sac_trainer.use_cql
        old_cql_alpha = self.sac_trainer.cql_alpha
        old_bc_lambda = self.sac_trainer.bc_lambda
        old_actor_lr = self.sac_trainer.actor_optimizer.param_groups[0]["lr"]

        # ═══ 5. Apply sleep settings ═══
        self.sac_trainer.use_cql = use_cql
        self.sac_trainer.cql_alpha = cql_alpha
        self.sac_trainer.bc_lambda = bc_lambda_init
        sleep_actor_lr = old_actor_lr * actor_lr_scale
        self.sac_trainer.actor_optimizer.param_groups[0]["lr"] = sleep_actor_lr

        # ═══ 6. SGD loop ═══
        critic_losses = []
        actor_losses = []

        for step_i in range(sac_update_steps):
            batch = self.sac_trainer.buffer.sample_balanced(
                self.sac_trainer.batch_size,
                current_mesh=mesh_name,
                current_ratio=current_mesh_ratio,
                bc_ratio=bc_ratio,
                elite_ratio=elite_ratio,
            )

            # Critic every step
            self.sac_trainer.update_critic_cql(batch)

            # Actor: after warmup, every N-th step
            if step_i >= critic_warmup_steps and step_i % actor_update_every == 0:
                self.sac_trainer.update_actor(batch)
                self.sac_trainer.update_alpha(batch)

            self.sac_trainer.soft_update_target()

            # BC lambda decay
            self.sac_trainer.bc_lambda = max(
                self.sac_trainer.bc_lambda * bc_lambda_decay,
                bc_lambda_min,
            )

            # Logging
            if (step_i + 1) % 500 == 0:
                logger.info(
                    "Sleep SAC step %d/%d: bc_lambda=%.3f, actor=%s",
                    step_i + 1, sac_update_steps,
                    self.sac_trainer.bc_lambda,
                    "active" if step_i >= critic_warmup_steps else "warmup",
                )

        # ═══ 7. Restore settings ═══
        self.sac_trainer.use_cql = old_use_cql
        self.sac_trainer.cql_alpha = old_cql_alpha
        # Keep decayed bc_lambda (don't restore — it's the new state)
        self.sac_trainer.actor_optimizer.param_groups[0]["lr"] = old_actor_lr

        # ═══ 8. Save ═══
        self.sac_trainer.save(self.sac_save_dir)

        result = {
            "status": "completed",
            "object": object_name,
            "level": level,
            "sac_update_steps": sac_update_steps,
            "transitions_added": added,
            "success_trails": len(success_trails),
            "failure_trails": len(failure_trails),
            "success_transitions": len(success_psac),
            "failure_transitions": len(failure_psac),
            "bc_data_size": len(self.sac_trainer.bc_data) if self.sac_trainer.bc_data else 0,
            "final_bc_lambda": round(self.sac_trainer.bc_lambda, 4),
            "buffer_size": len(self.sac_trainer.buffer),
        }

        logger.info(
            "Sleep SAC complete: object=%s, level=%d, steps=%d, "
            "added=%d, bc=%d, final_bc_lambda=%.3f",
            object_name, level, sac_update_steps,
            added, result["bc_data_size"], self.sac_trainer.bc_lambda,
        )

        return result

    def _get_level_success_rate(self, level: int) -> float:
        history = self._level_success_history.get(level)
        if history is None or len(history) == 0:
            return 0.0
        return sum(history) / len(history)

    def _get_adaptive_epsilon(self) -> float:
        """Compute epsilon based on rolling success rate.

        Returns moderate default during warmup, then adapts:
        high success → low epsilon (exploit),
        low success → high epsilon (explore).
        """
        return 0.02
        if len(self.success_history) < self.epsilon_warmup_episodes:
            return self.epsilon_default
        rate = self.success_rate
        return 0.05 + 0.3 * (1.0 - rate)

    def get_action(self, state, current_pose, sensor_data):
        """Get action from arbitrator.

        Returns:
            Tuple of (action_type, action_params, source_name).
        """
        if self.sac_trainer is not None:
            self.arbitrator.sac_actor = self.sac_trainer.actor
            self.arbitrator.state_mean = self.sac_trainer.state_mean
            self.arbitrator.state_std = self.sac_trainer.state_std
            self.arbitrator.param_mean = self.sac_trainer.param_mean
            self.arbitrator.param_std = self.sac_trainer.param_std

        return self.arbitrator.decide(
            state, current_pose, sensor_data
        )

    def decide_mode(self) -> str:
        level_history_len = 50
        current_level = self.arbitrator._current_level
        level_history = self._level_success_history.get(current_level)

        if level_history is None or len(level_history) < level_history_len:
            return "online"

        # --- Get per-level tracks and eval counts ---
        q_results = self.arbitrator._level_q_results[current_level]
        h_results = self.arbitrator._level_heuristic_results[current_level]
        sac_results = self.arbitrator._level_sac_results[current_level]
        blend_results = self.arbitrator._level_blend_results[current_level]

        q_track = self.arbitrator._get_track(q_results)
        h_track = self.arbitrator._get_track(h_results)
        sac_track = self.arbitrator._get_track(sac_results)
        blend_track = self.arbitrator._get_track(blend_results)
        best_ml_track = max(q_track, sac_track, blend_track)
        h_track = max(h_track, 0.1)

        # --- Mastered hysteresis based on Q only ---
        enter_thr = float(self.mastered_threshold)            # e.g. 0.95
        exit_thr = float(getattr(self, "mastered_exit_threshold", enter_thr - 0.05))  # e.g. 0.90

        min_q_evals = int(getattr(self, "mastered_min_q_evals", level_history_len))  # recommend 30-50
        q_evals = len(q_results)

        if self.mode == "mastered":
            # stay mastered while Q remains good
            if q_evals >= min_q_evals and q_track >= exit_thr:
                return "mastered"
            # else fall through
        else:
            if q_evals >= min_q_evals and q_track >= enter_thr:
                return "mastered"

        return "online"

    def decide_mode_old(self) -> str:
        current_level = self.arbitrator._current_level
        level_history = self._level_success_history.get(current_level)
        if level_history is None or len(level_history) < self.monitor_window:
            return "online"
        if self._episodes_since_offline < self.post_offline_cooldown:
            return "online"

        rate = sum(level_history) / len(level_history)

        if rate >= self.mastered_threshold:
            return "mastered"

        if self._episodes_since_offline < self.offline_check_window:
            return "online"

        h_track = self.arbitrator._get_track(
            self.arbitrator._level_heuristic_results[current_level]
        )
        h_track = max(h_track, 0.1)

        q_track = self.arbitrator._get_track(
            self.arbitrator._level_q_results[current_level]
        )
        sac_track = self.arbitrator._get_track(
            self.arbitrator._level_sac_results[current_level]
        )
        blend_track = self.arbitrator._get_track(
            self.arbitrator._level_blend_results[current_level]
        )
        best_ml_track = max(q_track, sac_track, blend_track)

        if best_ml_track < h_track * 0.5:
            if (
                self.total_offline_iterations == 0
                and self.total_episodes < self.min_online_before_offline
            ):
                return "online"
            if self.total_offline_iterations >= self.max_offline_iterations:
                return "online"
            return "offline"

        return "online"
    
    def on_episode_complete(
        self,
        success: bool,
        transitions: List[Dict[str, Any]],
    ) -> None:
        self.success_history.append(success)
        current_level = self.arbitrator._current_level
        if current_level not in self._level_success_history:
            self._level_success_history[current_level] = deque(maxlen=self.monitor_window)
        self._level_success_history[current_level].append(success)
        self.total_episodes += 1
        self._episodes_since_offline += 1
        self.online_episodes_since_sac_update += 1

        old_mode = self.mode
        self.mode = self.decide_mode()

        if self.mode != old_mode:
            self.mode_changes.append({
                "episode": self.total_episodes,
                "from_mode": old_mode,
                "to_mode": self.mode,
                "success_rate": round(self.success_rate, 3),
                "offline_iterations": self.total_offline_iterations,
                "sac_updates": self.total_sac_updates,
            })
            logger.info(
                "AdaptiveTraining: mode %s → %s "
                "(rate=%.3f, episodes=%d)",
                old_mode,
                self.mode,
                self.success_rate,
                self.total_episodes,
            )

        if self.mode == "mastered":
            self.controller.mode = "adaptive"
            self.controller.epsilon = 0.02
            self.controller.strategic_epsilon = 0.02
            # NEW: freeze learning
            self.controller.learning_enabled = False
            return

        if self.mode == "online":
            self.controller.learning_enabled = True
            self.controller.mode = "adaptive"
            eps = self._get_adaptive_epsilon()
            self.controller.epsilon = eps
            self.controller.strategic_epsilon = eps
            self._collect_transitions(success, transitions)
            if self.enable_online_sac_updates:
                self._maybe_online_sac_update()
            return

        if self.mode == "offline":
            self.controller.learning_enabled = True
            if not self.enable_offline_sac_retrain:
                logger.info("OFFLINE SAC disabled (policy-only mode)")
            else:
                self._trigger_offline()
                self._offline_just_completed = True
                self.controller.mode = "adaptive"
                eps = self._get_adaptive_epsilon()
                self.controller.epsilon = eps
                self.controller.strategic_epsilon = eps
                self.mode = "online"
                self.mode_changes.append({
                    "episode": self.total_episodes,
                    "from_mode": "offline",
                    "to_mode": "online",
                    "success_rate": 0.0,
                    "offline_iterations": self.total_offline_iterations,
                    "sac_updates": self.total_sac_updates,
                    "note": "post_offline_reset",
                })

    def _collect_transitions(
        self,
        success: bool,
        transitions: List[Dict[str, Any]],
    ) -> None:
        """Collect transitions for SAC training.

        Success transitions always collected (buffer + BC).
        Failure transitions collected with 30% probability
        (critic needs some negatives, but not too many).
        """
        if not transitions:
            return

        psac_transitions = self.extractor.convert_trajectory(
            transitions
        )

        if success:
            self.online_transitions_raw.extend(psac_transitions)
            self.bc_transitions.extend(psac_transitions)
            if len(self.bc_transitions) > self.max_bc_transitions:
                self.bc_transitions = self.bc_transitions[
                    -self.max_bc_transitions :
                ]
        else:
            # Adaptive failure collection: less when success rate is low
            failure_prob = min(0.3, self.success_rate * 0.5)
            if np.random.random() < failure_prob:
                self.online_transitions_raw.extend(psac_transitions)

    def _maybe_online_sac_update(self) -> None:
        """Periodic online SAC update.

        Mirrors PSACTrainer.train() logic:
        - Critic warmup: first N updates critic-only
        - BC lambda decay: gradually free actor from BC
        - Reduced actor lr: prevent catastrophic forgetting
        - CQL critic: prevent overestimation on new object
        """
        if self.sac_trainer is None:
            return

        if (
            self.online_episodes_since_sac_update
            < self.online_sac_update_every
        ):
            return

        if len(self.online_transitions_raw) < 100:
            self.online_episodes_since_sac_update = 0
            return

        # Add transitions to buffer
        all_normalized = (
            self.sac_trainer._normalize_bc_transitions(
                self.online_transitions_raw
            )
        )
        added = 0
        mesh_name = Path(self.mesh_path).stem if self.mesh_path else "unknown"
        for tr in all_normalized:
            if tr.next_state is not None:
                self.sac_trainer.buffer.add(
                    state=tr.state,
                    action_type=tr.action_type,
                    action_params=tr.action_params,
                    reward=tr.reward,
                    next_state=tr.next_state,
                    done=tr.done,
                    mesh_name=mesh_name,
                )
                added += 1

        # Update BC data (successful only)
        if self.bc_transitions:
            self.sac_trainer.bc_data = self.bc_transitions.copy()

        self.online_transitions_raw = []

        if len(self.sac_trainer.buffer) < self.sac_trainer.batch_size:
            self.online_episodes_since_sac_update = 0
            return

        # ═══ BC lambda decay (like training) ═══
        self.sac_trainer.bc_lambda = max(
            self.sac_trainer.bc_lambda * 0.95,
            self.sac_trainer.bc_lambda_min,
        )

        # ═══ Determine if actor should update ═══
        actor_ready = (
            self._online_critic_warmup_updates
            >= self._critic_warmup_threshold
        )

        mesh_name = Path(self.mesh_path).stem if self.mesh_path else "unknown"
        for step_i in range(self.online_sac_update_steps):
            batch = self.sac_trainer.buffer.sample_balanced(
                self.sac_trainer.batch_size,
                current_mesh=mesh_name,
                current_ratio=0.5,
                bc_ratio=0.1,
                elite_ratio=0.1,
            )

            # CQL critic every step
            old_cql_alpha = self.sac_trainer.cql_alpha
            self.sac_trainer.cql_alpha = 1.0
            self.sac_trainer.update_critic_cql(batch)
            self.sac_trainer.cql_alpha = old_cql_alpha

            # Actor: after warmup, every 10th step, reduced lr
            if (
                actor_ready
                and step_i % 10 == 0
                and self.bc_transitions
            ):
                old_lr = (
                    self.sac_trainer.actor_optimizer
                    .param_groups[0]['lr']
                )
                self.sac_trainer.actor_optimizer \
                    .param_groups[0]['lr'] = old_lr * 0.1
                self.sac_trainer.update_actor(batch)
                self.sac_trainer.actor_optimizer \
                    .param_groups[0]['lr'] = old_lr

            self.sac_trainer.soft_update_target()

        self._online_critic_warmup_updates += 1
        self.total_sac_updates += 1
        self.online_episodes_since_sac_update = 0

        logger.info(
            "Online SAC update #%d: added=%d, bc=%d, "
            "steps=%d, bc_lambda=%.3f, "
            "actor=%s, warmup=%d/%d",
            self.total_sac_updates,
            added,
            len(self.bc_transitions),
            self.online_sac_update_steps,
            self.sac_trainer.bc_lambda,
            "active" if actor_ready else "warmup",
            self._online_critic_warmup_updates,
            self._critic_warmup_threshold,
        )

    def _trigger_offline(self):
        self.total_offline_iterations += 1
        self._episodes_since_offline = 0
        # ═══ Guard: offline retrain requires CAD mesh ═══
        if self.env is None or self.mesh_path is None:
            logger.warning(
                "OFFLINE retrain skipped: no mesh/env available "
                "(no-CAD inference-only mode)"
            )
            return

        logger.info(
            "AdaptiveTraining: OFFLINE #%d (rate=%.3f)",
            self.total_offline_iterations,
            self.success_rate,
        )

        logger.info(
            "AdaptiveTraining: OFFLINE #%d (rate=%.3f)",
            self.total_offline_iterations,
            self.success_rate,
        )

        # Log pre-offline track records
        current_level = self.arbitrator._current_level
        for source_name, results_dict in [
            ("Q", self.arbitrator._level_q_results),
            ("SAC", self.arbitrator._level_sac_results),
            ("Heuristic", self.arbitrator._level_heuristic_results),
        ]:
            for level, results in results_dict.items():
                if results:
                    logger.info(
                        "Pre-offline %s track L%d: %.3f "
                        "(%d episodes)",
                        source_name,
                        level,
                        sum(results) / len(results),
                        len(results),
                    )

        current_level = self.arbitrator._current_level
        all_levels = list(
            self.config.get("curriculum_levels", [[10, 120]])
        )
        all_filters = list(
            self.config.get("curriculum_filters", [{}])
        )
        if current_level >= len(all_levels):
            remaining_levels = [all_levels[-1]] if all_levels else [[10, 120]]
            remaining_filters = [all_filters[-1]] if all_filters else [{}]
        else:
            remaining_levels = all_levels[current_level:]
            remaining_filters = all_filters[current_level:]

        if not remaining_levels:
            remaining_levels = [[10, 120]]
        if not remaining_filters:
            remaining_filters = [{}]

        q_save_dir = self.q_save_dir
        self.controller.save(q_save_dir)

        q_warmup = self.offline_q_episodes // 2

        logger.info(
            "OFFLINE Q: %d episodes (warmup=%d), eps %.1f→%.1f, "
            "from level %d (%d levels)",
            self.offline_q_episodes,
            q_warmup,
            self.offline_q_epsilon_start,
            self.offline_q_epsilon_min,
            current_level,
            len(remaining_levels),
        )

        train_result = run_episodes(
            mesh_dir=str(self.runs_dir.parent),
            save_dir=q_save_dir,
            load_dir=q_save_dir,
            num_episodes=self.offline_q_episodes,
            config={
                **self.config,
                "mode": "train_adapt_epsilon",
                "epsilon_start": self.offline_q_epsilon_start,
                "epsilon_min": self.offline_q_epsilon_min,
                "num_episodes": self.offline_q_episodes,
                "warmup_episodes": q_warmup,
                "curriculum_levels": remaining_levels,
                "curriculum_filters": remaining_filters,
                "unfreeze_normalization": True,
            },
            mesh_path=self.mesh_path,
            seed=42,
            return_metrics=True,
            curriculum_config={
                "levels": remaining_levels,
                "promote_threshold": self.promote_threshold,
                "promote_window": self.promote_window,
            },
        )

        self.controller = RLGoalApproachController.load(
            q_save_dir,
            agent_id=self.controller.agent_id,
            config={**self.config, "mode": "adaptive"},
        )
        self.arbitrator.controller = self.controller
        # Re-warmup RunningQStats after offline retrain
        self.arbitrator._running_q_stats_free = RunningQStats(warmup=200)
        self.arbitrator._running_q_stats_surface = RunningQStats(warmup=200)
        self.arbitrator._warmup_running_stats()

        self.arbitrator._level_q_results.clear()
        self.arbitrator._level_sac_results.clear()
        self.arbitrator._level_heuristic_results.clear()
        self.arbitrator._level_blend_results.clear()
        self.arbitrator._calibration_counter = 0
        self.arbitrator._is_calibrating = True
        self.arbitrator._episodes_on_level = 0

        q_rate = train_result.get("success_rate", 0.0)
        q_retrain_result = {
            "episodes": self.offline_q_episodes,
            "success_rate": q_rate,
            "curriculum_stats": train_result.get(
                "curriculum_stats", {}
            ),
            "stats": {
                k: v
                for k, v in train_result.get(
                    "stats", {}
                ).items()
                if k in (
                    "total_episodes",
                    "total_steps",
                    "total_goals_reached",
                    "success_rate",
                    "termination_counts",
                    "termination_rates",
                    "q_store_free",
                    "q_store_surface",
                    "collision_stats",
                    "steps_per_success",
                )
            },
            "phase_metrics": train_result.get(
                "phase_metrics", {}
            ),
        }

        q_retrain_path = (
            Path(self.q_save_dir)
            / "offline_retrain_result.json"
        )
        q_retrain_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        try:
            import json
            with q_retrain_path.open("w") as f:
                json.dump(
                    q_retrain_result, f, indent=2
                )
        except Exception as exc:
            logger.warning(
                "Could not save Q retrain result: %s",
                exc,
            )

        logger.info(
            "OFFLINE Q complete: rate=%.3f, "
            "saved to %s",
            q_rate,
            q_retrain_path,
        )

        success_trails = train_result.get("success_trails", [])

        if (
            self.sac_trainer is not None
            and success_trails
            and len(success_trails) >= 5
        ):
            all_bc = self.extractor.convert_all_trajectories(
                success_trails
            )

            logger.info(
                "OFFLINE SAC: %d ep, %d bc from %d trails, "
                "from level %d, CQL=True",
                self.offline_sac_episodes,
                len(all_bc),
                len(success_trails),
                current_level,
            )

            # Combine original BC with new success trails
            # (don't lose knowledge about other objects)
            if self.sac_trainer.bc_data is not None:
                combined_bc = self.sac_trainer.bc_data + all_bc
                # Keep manageable size
                if len(combined_bc) > self.max_bc_transitions * 2:
                    # Keep all new + sample from old
                    old_count = len(self.sac_trainer.bc_data)
                    keep_old = min(old_count, self.max_bc_transitions)
                    old_indices = np.random.choice(
                        old_count, keep_old, replace=False
                    )
                    combined_bc = (
                        [self.sac_trainer.bc_data[i] for i in old_indices]
                        + all_bc
                    )
                self.sac_trainer.bc_data = combined_bc
            else:
                self.sac_trainer.bc_data = all_bc

            self.sac_trainer.bc_lambda = self.sac_trainer.bc_lambda_init

            old_use_cql = self.sac_trainer.use_cql
            old_cql_alpha = self.sac_trainer.cql_alpha
            self.sac_trainer.use_cql = True
            self.sac_trainer.cql_alpha = 1.0
            old_eval_interval = self.sac_trainer.eval_interval
            self.sac_trainer.eval_interval = 100
            self.sac_trainer.buffer.set_current_mesh(
                Path(self.mesh_path).stem
            )
            self.sac_trainer.train(
                env=self.env,
                controller=self.controller,
                num_episodes=self.offline_sac_episodes,
                warmup_steps=500,
                update_every=1,
                updates_per_step=1,
                log_interval=50,
                save_dir=self.sac_save_dir,
                curriculum_levels=remaining_levels,
                curriculum_filters=remaining_filters,
                mesh_name=Path(self.mesh_path).stem,
            )

            self.sac_trainer.eval_interval = old_eval_interval
            self.sac_trainer.use_cql = old_use_cql
            self.sac_trainer.cql_alpha = old_cql_alpha

            self.online_transitions_raw = []

            # Log SAC retrain results
            sac_stats = self.sac_trainer.get_stats() if hasattr(self.sac_trainer, 'get_stats') else {}
            sac_retrain_result = {
                "episodes": self.offline_sac_episodes,
                "sac_stats": sac_stats,
                "bc_data_size": len(all_bc),
                "success_trails_used": len(success_trails),
            }
            
            sac_retrain_path = (
                Path(self.sac_save_dir)
                / "offline_retrain_result.json"
            )
            sac_retrain_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            try:
                import json
                with sac_retrain_path.open("w") as f:
                    json.dump(
                        sac_retrain_result, f, indent=2
                    )
            except Exception as exc:
                logger.warning(
                    "Could not save SAC retrain result: %s",
                    exc,
                )

            logger.info(
                "OFFLINE SAC complete: %d episodes, "
                "bc=%d, trails=%d, saved to %s",
                self.offline_sac_episodes,
                len(all_bc),
                len(success_trails),
                sac_retrain_path,
            )
        else:
            logger.info(
                "OFFLINE SAC skipped: insufficient trails (%d)",
                len(success_trails) if success_trails else 0,
            )

        self.success_history.clear()
        self._level_success_history.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "success_rate": round(self.success_rate, 3),
            "total_episodes": self.total_episodes,
            "history_size": len(self.success_history),
            "online_transitions_pending": len(
                self.online_transitions_raw
            ),
            "bc_transitions": len(self.bc_transitions),
            "sac_loaded": self.sac_trainer is not None,
            "total_sac_updates": self.total_sac_updates,
            "total_offline_iterations": self.total_offline_iterations,
            "episodes_since_sac_update": (
                self.online_episodes_since_sac_update
            ),
            "episodes_since_offline": self._episodes_since_offline,
            "current_epsilon": round(
                self._get_adaptive_epsilon(), 3
            ),
        }
