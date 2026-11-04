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

"""Arbitrator: decides which action source to use per step.

Logic:
  1. Calibration phase (10 ep per source × 3 sources): forced Q/SAC/heuristic
  2. Normal phase: scoring = confidence × track_record[level]
     - ML track < heuristic track × 0.75: heuristic fallback
     - Q >> SAC: Q type + SAC params
     - Q > SAC, types differ: Q type + Q params
     - Q > SAC, types agree: blend params (50/50)
     - SAC > Q: SAC type + SAC params
"""

import logging
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .experience_extractor import ExperienceExtractor
from .sac_actor import SACActorNetwork
from .rl_goal_approach_controller import (
    RLGoalApproachController,
    RunningQStats,
)

logger = logging.getLogger(__name__)


def sac_to_discrete(action_type: int, action_params: np.ndarray) -> int:
    if action_type == 0:
        sin_a = float(action_params[0])
        cos_a = float(action_params[1])
        angle = np.degrees(np.arctan2(sin_a, cos_a)) % 360.0
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        best_idx = 0
        best_diff = 360.0
        for i, d in enumerate(directions):
            diff = abs(angle - d)
            diff = min(diff, 360 - diff)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx
    elif action_type == 1:
        step = float(action_params[0])
        if step < 0:
            return 9
        elif abs(step) <= 3.0:
            return 19
        else:
            return 8
    elif action_type == 2:
        sin_a = float(action_params[0])
        cos_a = float(action_params[1])
        angle = np.degrees(np.arctan2(sin_a, cos_a))
        if abs(angle) > 10.0:
            return 22 if angle >= 0 else 23
        return 10 if angle >= 0 else 11
    elif action_type == 3:
        sin_a = float(action_params[0])
        cos_a = float(action_params[1])
        angle = np.degrees(np.arctan2(sin_a, cos_a))
        if abs(angle) > 10.0:
            return 20 if angle >= 0 else 21
        return 12 if angle >= 0 else 13
    elif action_type == 4:
        sin_a = float(action_params[0])
        cos_a = float(action_params[1])
        angle = np.degrees(np.arctan2(sin_a, cos_a))
        return 14 if angle >= 0 else 15
    elif action_type == 5:
        return 16
    elif action_type == 6:
        return 17
    elif action_type == 7:
        return 18
    return 0

def sac_to_discrete_old(action_type: int, action_params: np.ndarray) -> int:
    if action_type == 0:
        angle = float(action_params[0]) % 360.0
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        best_idx = 0
        best_diff = 360.0
        for i, d in enumerate(directions):
            diff = abs(angle - d)
            diff = min(diff, 360 - diff)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx
    elif action_type == 1:
        step = float(action_params[0])
        if step < 0:
            return 9
        elif abs(step) <= 3.0:
            return 19
        else:
            return 8
    elif action_type == 2:
        angle = float(action_params[0])
        if abs(angle) > 10.0:
            return 22 if angle >= 0 else 23
        return 10 if angle >= 0 else 11
    elif action_type == 3:
        angle = float(action_params[0])
        if abs(angle) > 10.0:
            return 20 if angle >= 0 else 21
        return 12 if angle >= 0 else 13
    elif action_type == 4:
        return 14 if float(action_params[0]) >= 0 else 15
    elif action_type == 5:
        return 16
    elif action_type == 6:
        return 17
    elif action_type == 7:
        return 18
    return 0


class Arbitrator:
    """Decides which action source to use per step.

    Logic:
      1. Q-confident override (high confidence + spread > 3.0):
         - Q type == SAC type → SAC params (Q confirms SAC)
         - Q type != SAC type → heuristic (conflict resolution)
      2. Track record scoring:
         - ML below heuristic and not improving → heuristic
         - Heuristic budget = gap between heuristic and ML track
         - Otherwise → SAC (default)
    """

    def __init__(
        self,
        controller: RLGoalApproachController,
        sac_actor: Optional[SACActorNetwork] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        param_mean: Optional[np.ndarray] = None,
        param_std: Optional[np.ndarray] = None,
        min_eval_per_source: int = 5,
        force_heuristic: bool = False,
        temperature_override = None,
        sac_proven_threshold: float = 0.6,
    ):
        self.controller = controller
        self.sac_actor = sac_actor
        self.state_mean = state_mean
        self.state_std = state_std
        self.param_mean = param_mean
        self.param_std = param_std

        self.stats = {
            "q_store_chosen": 0,
            "sac_chosen": 0,
            "blend_chosen": 0,
            "heuristic_chosen": 0,
            "total_decisions": 0,
        }

        self.q_proposed_actions = defaultdict(int)
        self.sac_proposed_actions = defaultdict(int)
        self.q_chosen_actions = defaultdict(int)
        self.sac_chosen_actions = defaultdict(int)
        self.blend_chosen_actions = defaultdict(int)
        self.heuristic_chosen_actions = defaultdict(int)

        self.agreement_count = 0
        self.both_proposed_count = 0
        # Q confidence calibrated thresholds
        self._q_reliable_threshold_surface = 0.25
        self._q_reliable_threshold_free = 0.25

        # Running best_Q stats for best_q_norm
        self._running_best_q_free = RunningQStats(warmup=200)
        self._running_best_q_surface = RunningQStats(warmup=200)

        self.q_confidence_history = deque(maxlen=100)
        self.q_spread_history = deque(maxlen=100)

        self._current_episode_sources = []

        # Per-level track record
        self._min_eval_per_source = min_eval_per_source
        self._level_q_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._level_sac_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._level_heuristic_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._level_blend_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )

        # Level state (no calibration)
        self._current_level = 0
        self._is_calibrating = False
        self._calibration_counter = 0
        self._calibration_source = None
        self._episodes_on_level = 0

        self._param_dims = ExperienceExtractor.get_param_dims()
        self._type_names = ExperienceExtractor.get_type_names()

        # Running Q statistics
        self._running_q_stats_free = RunningQStats(warmup=200)
        self._running_q_stats_surface = RunningQStats(warmup=200)
        self._running_best_q_free = RunningQStats(warmup=200)
        self._running_best_q_surface = RunningQStats(warmup=200)
        self._warmup_running_stats()

        # Strategic SAC references
        self._sac_strategic_detach: Optional[Any] = None
        self._sac_strategic_direction: Optional[Any] = None

        # Arbitrage-only stats
        self._arbitrage_stats = {
            "q_store_chosen": 0,
            "sac_chosen": 0,
            "blend_chosen": 0,
            "heuristic_chosen": 0,
            "total_decisions": 0,
        }

        # Heuristic budget tracking per level
        self._level_total_decisions: Dict[int, int] = defaultdict(int)
        self._level_heuristic_decisions: Dict[int, int] = defaultdict(int)
        # Per-step per-level source counts
        self._level_step_counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"q_store": 0, "sac": 0, "blend": 0, "heuristic": 0, "total": 0}
        )
        self._heuristic_eps_min = 0.05
        self._force_heuristic = force_heuristic
        self._agreement_categories: Dict[str, int] = defaultdict(int)
        self._disagreement_log: deque = deque(maxlen=1000)
        # ═══ Q diagnostics counters ═══
        self._q_diag_confidence_sum = 0.0
        self._q_diag_confidence_count = 0
        self._q_diag_reliable_count = 0
        self._q_diag_total_count = 0
        self._q_diag_nn_distance_sum = 0.0
        self._q_diag_nn_distance_count = 0
        # Per-level Q diagnostics
        self._q_diag_per_level: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {
                "confidence_sum": 0.0,
                "confidence_count": 0,
                "reliable_count": 0,
                "total_count": 0,
                "nn_distance_sum": 0.0,
                "nn_distance_count": 0,
                # Confidence breakdown components
                "distance_comp_sum": 0.0,
                "consistency_comp_sum": 0.0,
                "recency_comp_sum": 0.0,
                "count_comp_sum": 0.0,
                "top2_spread_comp_sum": 0.0,
                "comp_count": 0,
            }
        )
        self.temperature_override = temperature_override
        self.sac_proven_threshold = sac_proven_threshold

        # Calibration: ensure SAC/heuristic have track records per level
        self._calibration_episodes_per_source = 10
        self._level_calibration_done: Dict[int, bool] = {}

    def _needs_calibration(self, level: int) -> Optional[str]:
        """Check if SAC needs calibration on this level.
        
        Returns 'sac' if SAC needs more evals, None otherwise.
        Heuristic doesn't need calibration — it's geometry-based fallback.
        """
        if not self.sac_actor:
            return None
            
        if self._level_calibration_done.get(level, False):
            return None

        sac_evals = len(self._level_sac_results[level])
        target = self._calibration_episodes_per_source

        if sac_evals >= target:
            self._level_calibration_done[level] = True
            return None

        return "sac"

    def _warmup_running_stats(self):
        MAX_WARMUP_POINTS = 10000

        for store, stats, best_stats in [
            (self.controller.q_store_free,
             self._running_q_stats_free,
             self._running_best_q_free),
            (self.controller.q_store_surface,
             self._running_q_stats_surface,
             self._running_best_q_surface),
        ]:
            if not store.points:
                continue

            pids = list(store.points.keys())
            if len(pids) > MAX_WARMUP_POINTS:
                pids = np.random.choice(
                    pids, size=MAX_WARMUP_POINTS, replace=False
                )

            for pid in pids:
                q = store.points[pid].q_values
                stats.update(q)
                best_stats.update(np.array([float(np.max(q))]))

            logger.info(
                f"Arbitrator warmup {store.name}: "
                f"mean={stats.mean:.3f}, std={stats.std:.3f}, "
                f"n={stats.count}, "
                f"best_q_mean={best_stats.mean:.3f}, "
                f"best_q_std={best_stats.std:.3f}, "
                f"points_sampled={len(pids)}"
            )

    def _get_heuristic_eps(self, ml_track: float, h_track: float) -> float:
        """Dynamic heuristic epsilon = gap between heuristic and ML.

        h_track=0.8, best_ml=0.6 → eps=0.2 (20% heuristic)
        h_track=0.8, best_ml=0.85 → eps=0.05 (ML better, minimum)
        """
        gap = h_track - ml_track
        eps_max = max(gap, self._heuristic_eps_min)
        eps_max_min = min(eps_max, 0.2)
        return eps_max_min

    def _get_level_tracks(self, level: int):
        """Compute ML and heuristic tracks for a level."""
        q_track = self._get_track(self._level_q_results[level])
        sac_track = self._get_track(self._level_sac_results[level])
        b_track = self._get_track(self._level_blend_results[level])
        h_track = self._get_track(self._level_heuristic_results[level])

        ml_tracks = []
        if len(self._level_q_results[level]) >= self._min_eval_per_source:
            ml_tracks.append(q_track)
        if len(self._level_sac_results[level]) >= self._min_eval_per_source:
            ml_tracks.append(sac_track)
        if len(self._level_blend_results[level]) >= self._min_eval_per_source:
            ml_tracks.append(b_track)

        if len(self._level_heuristic_results[level]) < self._min_eval_per_source:
            h_track = 0.5
        best_ml_track = max(ml_tracks) if ml_tracks else 0.5
        worst_ml_track = min(ml_tracks) if ml_tracks else 0.5

        return q_track, sac_track, b_track, h_track, best_ml_track, worst_ml_track, len(ml_tracks) > 0
    
    def _is_ml_trend_increasing(self, level: int) -> bool:
        """Detect if any ML source is trending upward."""
        for results in [
            self._level_q_results[level],
            self._level_sac_results[level],
            self._level_blend_results[level],
        ]:
            if len(results) < 10:
                continue
            mid = len(results) // 2
            results_list = list(results)
            first_half = sum(results_list[:mid]) / max(mid, 1)
            second_half = sum(results_list[mid:]) / max(len(results_list) - mid, 1)
            if second_half > first_half + 0.05:
                return True
        return False

    def start_episode(self, level: int):
        if level != self._current_level:
            self._current_level = level
            self._episodes_on_level = 0
        self._episodes_on_level += 1
        self._current_episode_sources = []

    def _decide_crawl(
        self,
        q_action: int,
        q_params: np.ndarray,
        q_confidence: float,
        sac_type: int,
        sac_params: np.ndarray,
        has_sac: bool,
    ) -> tuple:
        """Decide crawl direction.

        Q knows the right direction (learned from heuristics on this object).
        SAC has better continuous params (precise angle, step size).

        If SAC also proposes MoveTangentially and direction within 45°:
            → blend: SAC continuous params, Q-confirmed direction
        If SAC proposes different direction or different type:
            → Q: Q knows this object's geometry better

        Returns:
            (action_type, action_params, source_name)
        """
        q_name = self._type_names.get(0, "MoveTangentially")

        if not has_sac or sac_type != 0:
            # SAC proposes non-crawl → Q direction
            self._record_decision("q_store")
            self.q_chosen_actions[q_name] += 1
            self._current_episode_sources.append("q_store")
            return 0, q_params, (
                f"q_crawl_sac_other("
                f"sac_type={sac_type},"
                f"conf={q_confidence:.2f})"
            )

        # Both propose MoveTangentially — compare directions
        # Q: discrete angle from action index
        q_angle = q_action * 45.0  # actions 0-7 → 0°, 45°, ..., 315°

        # SAC: continuous angle from sin/cos params
        sac_sin = float(sac_params[0])
        sac_cos = float(sac_params[1])
        sac_angle = float(np.degrees(np.arctan2(sac_sin, sac_cos))) % 360.0

        # Angular difference (0-180°)
        angle_diff = abs(q_angle - sac_angle)
        angle_diff = min(angle_diff, 360.0 - angle_diff)

        if angle_diff <= 45.0:
            # Close enough → blend: SAC continuous params
            sac_n = self._type_names.get(0, "MoveTangentially")
            self._record_decision("blend")
            self.blend_chosen_actions[sac_n] += 1
            self._current_episode_sources.append("blend")
            return 0, sac_params, (
                f"blend_crawl("
                f"q_angle={q_angle:.0f},"
                f"sac_angle={sac_angle:.0f},"
                f"diff={angle_diff:.0f},"
                f"conf={q_confidence:.2f})"
            )

        # Direction disagree > 45° → Q knows better
        self._record_decision("q_store")
        self.q_chosen_actions[q_name] += 1
        self._current_episode_sources.append("q_store")
        return 0, q_params, (
            f"q_crawl_disagree("
            f"q_angle={q_angle:.0f},"
            f"sac_angle={sac_angle:.0f},"
            f"diff={angle_diff:.0f},"
            f"conf={q_confidence:.2f})"
        )
    
    def _compute_agreement(
        self,
        q_discrete: int,
        sac_type: int,
        sac_params: np.ndarray,
    ) -> tuple[float, str, int]:
        """Compute agreement between Q (discrete) and SAC (continuous).

        Converts SAC to discrete via sac_to_discrete, then compares
        structural distance in the 24-action space.

        Returns:
            (score, category, sac_discrete) where:
            score: 1.0=identical, 0.75=neighbor, 0.5=moderate,
                0.25=opposite, 0.0=type_disagree
            category: "same"|"neighbor"|"moderate"|"opposite"|"type_disagree"
            sac_discrete: SAC action converted to discrete index
        """
        sac_discrete = sac_to_discrete(sac_type, sac_params)

        if q_discrete == sac_discrete:
            return 1.0, "same", sac_discrete

        q_type_id = ExperienceExtractor.DISCRETE_TO_PSAC[q_discrete][0]
        sac_type_id = ExperienceExtractor.DISCRETE_TO_PSAC[sac_discrete][0]

        if q_type_id != sac_type_id:
            return 0.0, "type_disagree", sac_discrete

        # Same type, different discrete action
        if q_type_id == 0:
            # MoveTangentially: circular distance on 8-direction ring
            # Actions 0-7 map to 0°,45°,...,315°
            q_idx = q_discrete      # 0..7
            sac_idx = sac_discrete   # 0..7
            dist = abs(q_idx - sac_idx)
            dist = min(dist, 8 - dist)

            if dist == 1:
                return 0.75, "neighbor", sac_discrete
            elif dist == 2:
                return 0.5, "moderate", sac_discrete
            else:
                # dist 3 or 4 (135°-180°)
                return 0.25, "opposite", sac_discrete

        elif q_type_id == 1:
            # MoveLinear: {8,19}=forward group, {9}=backward
            fwd = {8, 19}
            if (q_discrete in fwd and sac_discrete in fwd):
                return 0.75, "neighbor", sac_discrete
            else:
                return 0.25, "opposite", sac_discrete

        elif q_type_id == 2:
            # Turn: {10,22}=right, {11,23}=left
            right = {10, 22}
            left = {11, 23}
            same_group = (
                (q_discrete in right and sac_discrete in right)
                or (q_discrete in left and sac_discrete in left)
            )
            if same_group:
                return 0.75, "neighbor", sac_discrete
            else:
                return 0.25, "opposite", sac_discrete

        elif q_type_id == 3:
            # Look: {12,20}=up, {13,21}=down
            up = {12, 20}
            down = {13, 21}
            same_group = (
                (q_discrete in up and sac_discrete in up)
                or (q_discrete in down and sac_discrete in down)
            )
            if same_group:
                return 0.75, "neighbor", sac_discrete
            else:
                return 0.25, "opposite", sac_discrete

        elif q_type_id == 4:
            # SensorRotate: {14}=right, {15}=left — only opposite
            return 0.25, "opposite", sac_discrete

        # Types 5,6,7: single action each — same discrete caught above
        return 1.0, "same", sac_discrete

    def decide(self, state, current_pose, sensor_data):
        self.stats["total_decisions"] += 1
        level = self._current_level
        has_sac = self.sac_actor is not None
        self._level_total_decisions[level] += 1

        # ═══ Force heuristic when stuck ═══
        if self._force_heuristic:
            return self._decide_heuristic(
                state, current_pose, sensor_data,
                "forced_heuristic(mode=force_all)",
            )

        phase = getattr(self.controller, '_current_phase', 'CRAWL_TO_GOAL')
        if phase == "DETACH_NEEDED":
            return self._decide_heuristic(
                state, current_pose, sensor_data,
                "forced_heuristic(phase=DETACH_NEEDED)",
            )

        # ═══ Get proposals ═══
        q_action, q_confidence, q_spread, q_recency, q_best_norm = (
            self._get_q_action(state)
        )
        q_type = ExperienceExtractor.DISCRETE_TO_PSAC[q_action][0]
        q_params = self._discrete_to_params(q_action)

        if has_sac:
            sac_type, sac_params = self._get_sac_action_continuous(state)
        else:
            sac_type, sac_params = q_type, q_params.copy()

        # ═══ Agreement ═══
        agreement_score, agreement_cat, sac_discrete = (
            self._compute_agreement(q_action, sac_type, sac_params)
        )

        # ═══ Track proposals ═══
        q_name = self._type_names.get(q_type, f"type_{q_type}")
        self.q_proposed_actions[q_name] += 1
        self.q_confidence_history.append(q_confidence)
        self.q_spread_history.append(q_spread)

        if has_sac:
            sac_name = self._type_names.get(sac_type, f"type_{sac_type}")
            self.sac_proposed_actions[sac_name] += 1
            self.both_proposed_count += 1
            if agreement_cat == "same":
                self.agreement_count += 1

        self._agreement_categories[agreement_cat] = (
            self._agreement_categories.get(agreement_cat, 0) + 1
        )

        # ═══ Track records ═══
        #q_track, sac_track, b_track, h_track, best_ml_track, worst_ml_track, has_enough_data = (
        #    self._get_level_tracks(level)
        #)

        # ═══ Q reliability: only q_confidence ═══
        on_object = state[11] > 0.5
        q_reliable_threshold = (
            self._q_reliable_threshold_surface if on_object
            else self._q_reliable_threshold_free
        )
        q_is_reliable = q_confidence >= q_reliable_threshold

        # ═══ sac_proven: combined SAC + blend track ═══
        sac_results = self._level_sac_results[level]
        blend_results = self._level_blend_results[level]
        combined_count = len(sac_results) + len(blend_results)
        
        if combined_count >= self._min_eval_per_source and has_sac:
            combined_successes = sum(sac_results) + sum(blend_results)
            combined_track = combined_successes / combined_count
            sac_proven = combined_track >= self.sac_proven_threshold
        else:
            sac_proven = False

        # ═══════════════════════════════════════════════════════
        # CALIBRATION: force SAC/heuristic to get track records
        # ═══════════════════════════════════════════════════════
        cal_source = self._needs_calibration(level)
        if cal_source is not None:
            if cal_source == "sac" and has_sac:
                sac_n = self._type_names.get(sac_type, f"type_{sac_type}")
                self._record_decision("sac")
                self.sac_chosen_actions[sac_n] += 1
                self._current_episode_sources.append("sac")
                return sac_type, sac_params, (
                    f"calibration_sac(level={level},"
                    f"sac_evals={len(self._level_sac_results[level])})"
                )

        # ═══════════════════════════════════════════════════════
        # DECISION TREE
        # ═══════════════════════════════════════════════════════

        # --- 1. Q NOT RELIABLE → SAC if proven, else heuristic ---
        if not q_is_reliable:
            if sac_proven:
                sac_n = self._type_names.get(sac_type, f"type_{sac_type}")
                self._record_decision("sac")
                self.sac_chosen_actions[sac_n] += 1
                self._current_episode_sources.append("sac")
                return sac_type, sac_params, (
                    f"q_NOT_reliable_sac_proven("
                    f"q_conf={q_confidence:.2f},"
                    f"sac_track={combined_track:.2f})"
                )

            return self._decide_heuristic(
                state, current_pose, sensor_data,
                f"q_NOT_reliable_heuristic_default("
                f"q_conf={q_confidence:.2f},"
                f"level={level})",
            )
        
        # --- 2. Q RELIABLE ---
        # MoveTangentially (type 0): Q decides direction
        if q_type == 0:
            return self._decide_crawl(
                q_action, q_params, q_confidence,
                sac_type, sac_params, has_sac,
            )

        # All other actions: SAC (continuous params better)
        if has_sac:
            sac_n = self._type_names.get(sac_type, f"type_{sac_type}")
            self._record_decision("sac")
            self.sac_chosen_actions[sac_n] += 1
            self._current_episode_sources.append("sac")
            return sac_type, sac_params, (
                f"sac_non_crawl("
                f"q_type={q_name},"
                f"sac_type={sac_n},"
                f"conf={q_confidence:.2f})"
            )

        # No SAC → Q
        self._record_decision("q_store")
        self.q_chosen_actions[q_name] += 1
        self._current_episode_sources.append("q_store")
        return q_type, q_params, (
            f"q_non_crawl(conf={q_confidence:.2f})"
        )
    
    def _decide_heuristic(self, state, current_pose, sensor_data, reason):
        """Execute heuristic decision with standard bookkeeping."""
        level = self._current_level
        h_action = self._get_heuristic_action(state, current_pose, sensor_data)
        h_type = ExperienceExtractor.DISCRETE_TO_PSAC[h_action][0]
        h_params = self._discrete_to_params(h_action)
        self._record_decision("heuristic")
        self._level_heuristic_decisions[level] += 1
        h_name = self._type_names.get(h_type, f"type_{h_type}")
        self.heuristic_chosen_actions[h_name] += 1
        self._current_episode_sources.append("heuristic")
        return h_type, h_params, reason

    def _compute_heuristic_budget(
        self,
        q_confidence: float,
        q_best_norm: float,
        h_track: float,
        best_ml_track: float,
        has_data: bool,
    ) -> float:
        """Compute heuristic budget based on confidence and track records.

        Two factors:
        1. Q reliability: low confidence or low best_q_norm → more heuristic
        2. Track record gap: heuristic better than ML → more heuristic

        As ML learns (track improves, confidence grows),
        budget naturally shrinks.
        """
        on_object = True  # conservative default
        q_reliable_threshold = (
            self._q_reliable_threshold_surface if on_object
            else self._q_reliable_threshold_free
        )

        # Factor 1: Q reliability
        conf_ratio = q_confidence / max(q_reliable_threshold, 0.1)
        conf_component = np.clip(1.0 - conf_ratio, 0.0, 1.0)

        # best_q_norm < 0 → Q doesn't see good path → more heuristic
        bqn_component = np.clip(-q_best_norm * 0.3, 0.0, 0.3)

        # Factor 2: track record gap
        if has_data and h_track > 0.1:
            gap = h_track - best_ml_track
            track_component = np.clip(gap, 0.0, 0.5)
        else:
            track_component = 0.15

        budget = 0.05 + 0.4 * conf_component + 0.3 * bqn_component + 0.8 * track_component
        return float(np.clip(budget, 0.05, 0.70))

    def _log_disagreement(
        self,
        q_discrete: int,
        sac_discrete: int,
        agreement_cat: str,
        q_confidence: float,
        q_recency: float,
        q_track: float,
        sac_track: float,
        chosen: str,
        state: np.ndarray,
    ):
        """Log disagreement for post-hoc analysis."""
        if not hasattr(self, '_disagreement_log'):
            self._disagreement_log = deque(maxlen=1000)

        q_type_name = self._type_names.get(
            ExperienceExtractor.DISCRETE_TO_PSAC[q_discrete][0],
            f"idx_{q_discrete}",
        )
        sac_type_name = self._type_names.get(
            ExperienceExtractor.DISCRETE_TO_PSAC[sac_discrete][0],
            f"idx_{sac_discrete}",
        )

        self._disagreement_log.append({
            "step": self.stats["total_decisions"],
            "q_discrete": q_discrete,
            "sac_discrete": sac_discrete,
            "q_action": q_type_name,
            "sac_action": sac_type_name,
            "category": agreement_cat,
            "q_confidence": round(q_confidence, 3),
            "q_recency": round(q_recency, 3),
            "q_track": round(q_track, 3),
            "sac_track": round(sac_track, 3),
            "chosen": chosen,
            "dist_to_goal": round(float(state[0]), 1),
            "on_object": bool(state[11] > 0.5),
            "movement_eff": round(float(state[6]), 3),
        })

    def _record_decision(self, source: str):
        self.stats[f"{source}_chosen"] += 1
        self._arbitrage_stats[f"{source}_chosen"] += 1
        self._arbitrage_stats["total_decisions"] += 1
        # Per-step per-level
        lvl = self._current_level
        self._level_step_counts[lvl][source] += 1
        self._level_step_counts[lvl]["total"] += 1

    def _get_track(self, results: deque) -> float:
        if len(results) < self._min_eval_per_source:
            return 0.5
        return sum(results) / len(results)

    def _discrete_to_params(self, discrete_action: int) -> np.ndarray:
        _, params_fn = ExperienceExtractor.DISCRETE_TO_PSAC[discrete_action]
        raw_params = params_fn(self.controller.config)
        padded = np.zeros(3, dtype=np.float32)
        padded[: len(raw_params)] = raw_params
        return padded

    def _get_sac_action_continuous(
        self, state: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        state_norm = state
        if self.state_mean is not None:
            state_norm = (
                (state - self.state_mean) / (self.state_std + 1e-8)
            )

        on_object = state[11] > 0.5

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(
                state_norm.astype(np.float32)
            ).unsqueeze(0)

            type_logits, _, _ = self.sac_actor(state_t)

            if not on_object:
                type_logits[0, 0] = -1e9
                type_logits[0, 7] = -1e9
            if on_object:
                type_logits[0, 1] = -1e9
            if self.controller._consecutive_detach_count >= 3:
                type_logits[0, 7] = -1e9

            temperature = 0.3
            type_probs = torch.softmax(
                type_logits / temperature, dim=-1
            )
            type_probs = type_probs.clamp(min=1e-8)
            type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)
            type_dist = torch.distributions.Categorical(type_probs)
            action_type_t = type_dist.sample()
            action_type = action_type_t[0].item()

            _, ap, _, _ = self.sac_actor.sample_eval(
                state_t, temperature=temperature
            )
            dim = self._param_dims.get(action_type, 0)
            if dim > 0:
                _, param_mus, param_log_stds = self.sac_actor(state_t)
                if action_type in param_mus:
                    mu = param_mus[action_type][0]
                    log_std = param_log_stds[action_type][0]
                    std = (log_std.exp() * temperature).clamp(min=1e-6)
                    normal = torch.distributions.Normal(mu, std)
                    raw_sample = normal.rsample()
                    squashed = torch.tanh(raw_sample)

                    scale, center = self.sac_actor._get_scale_center(action_type)
                    if scale is not None:
                        scaled = squashed * scale[:dim] + center[:dim]
                    else:
                        scaled = squashed

                    padded = np.zeros(3, dtype=np.float32)
                    padded[:dim] = scaled.numpy()
                else:
                    padded = np.zeros(3, dtype=np.float32)
            else:
                padded = np.zeros(3, dtype=np.float32)

        return action_type, padded

    def _get_sac_params_for_type(
        self, state: np.ndarray, forced_type: int
    ) -> np.ndarray:
        if forced_type == 7:
            return np.zeros(3, dtype=np.float32)

        state_norm = state
        if self.state_mean is not None:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(
                state_norm.astype(np.float32)
            ).unsqueeze(0)
            _, param_mus, _ = self.sac_actor(state_t)

            dim = self._param_dims.get(forced_type, 0)
            if (
                dim > 0
                and forced_type in param_mus
                and self.param_mean is not None
            ):
                raw_mu = param_mus[forced_type][0]
                squashed = torch.tanh(raw_mu)
                from .sac_actor import ACTION_PARAM_BOUNDS
                bounds = ACTION_PARAM_BOUNDS.get(forced_type, [])
                if bounds:
                    for i, (lo, hi) in enumerate(bounds[:dim]):
                        center = (hi + lo) / 2.0
                        scale = (hi - lo) / 2.0
                        squashed[i] = squashed[i] * scale + center
                params = squashed.numpy()[:dim]
                padded = np.zeros(3, dtype=np.float32)
                padded[:dim] = params
                return padded

        return np.zeros(3, dtype=np.float32)

    def _get_q_action(self, state):
        store = self.controller._select_store(state)
        running_stats = (
            self._running_q_stats_surface
            if state[11] > 0.5
            else self._running_q_stats_free
        )
        running_best = (
            self._running_best_q_surface
            if state[11] > 0.5
            else self._running_best_q_free
        )

        if store.next_id == 0:
            return 0, 0.0, 0.0, 0.5, 0.0  # added best_q_norm=0.0

        q_values, confidence_info = (
            store.get_q_values_with_confidence(state)
        )

        # ============= q_confidence =================
        recency = confidence_info.get("recency", 0.5)

        if np.max(np.abs(q_values)) > 1e-8:
            running_stats.update(q_values)

        # ═══ Stable confidence from components ═══
        raw_t2s = confidence_info.get("top2_spread", 0.0)
        prox = confidence_info.get("proximity", 0.0)
        exp = confidence_info.get("experience", 0.0)
        cons = confidence_info.get("consistency", 0.0)

        t2s_norm = float(np.clip(raw_t2s / 1.0, 0.0, 1.0))
        t2s_gate = 0.3 + 0.7 * t2s_norm
        base_conf = 0.45 * prox + 0.35 * cons + 0.20 * exp
        confidence = float(np.clip(t2s_gate * base_conf, 0.0, 0.95))

        # ═══ V-norm boost/penalty ═══
        V = float(np.mean(q_values))
        if running_stats.is_warmed_up:
            V_norm = running_stats.normalize_value(V)
            if V_norm >= 0:
                boost = float(np.clip(V_norm * 0.15, 0.0, 0.3))
                q_confidence = min(confidence * (1.0 + boost), 0.95)
            else:
                # penalty = float(np.clip(-V_norm * 0.3, 0.0, 0.5))
                penalty = float(np.clip(-V_norm * 0.15, 0.0, 0.3))
                q_confidence = confidence * (1.0 - penalty)
        else:
            q_confidence = confidence

        # ═══ best_q_norm ═══
        best_q = float(np.max(q_values))
        if np.max(np.abs(q_values)) > 1e-8:
            running_best.update(np.array([best_q]))
        if running_best.is_warmed_up:
            best_q_norm = running_best.normalize_value(best_q)
        else:
            best_q_norm = 0.0

        # ═══ Q diagnostics collection ═══
        self._q_diag_confidence_sum += q_confidence
        self._q_diag_confidence_count += 1

        on_object_diag = state[11] > 0.5
        q_rel_thresh = (
            self._q_reliable_threshold_surface if on_object_diag
            else self._q_reliable_threshold_free
        )
        if q_confidence >= q_rel_thresh and best_q_norm >= -1.0:
            self._q_diag_reliable_count += 1
        self._q_diag_total_count += 1

        # Per-level
        lvl = self._current_level
        d = self._q_diag_per_level[lvl]
        d["confidence_sum"] += q_confidence
        d["confidence_count"] += 1
        if q_confidence >= q_rel_thresh and best_q_norm >= -1.0:
            d["reliable_count"] += 1
        d["total_count"] += 1

        # Confidence breakdown components (real keys from get_q_values_with_confidence)
        prox = confidence_info.get("proximity", 0.0)
        exp = confidence_info.get("experience", 0.0)
        cons = confidence_info.get("consistency", 0.0)
        rec = confidence_info.get("recency", 0.0)
        t2s = confidence_info.get("top2_spread", 0.0)

        d["distance_comp_sum"] += prox       # proximity = distance component
        d["consistency_comp_sum"] += cons
        d["recency_comp_sum"] += rec
        d["count_comp_sum"] += exp           # experience = count/visit component
        d["top2_spread_comp_sum"] += t2s
        d["comp_count"] += 1

        q_values = self.controller.apply_action_mask(q_values, state)

        valid = q_values > -1e8
        if valid.sum() > 1:
            q_spread = float(
                np.max(q_values[valid]) - np.min(q_values[valid])
            )
        else:
            q_spread = 0.0

        eps = self.controller._get_current_epsilon()
        temperature = max(np.clip(0.5 * eps, 0.01, 0.5), 0.05)
        if self.temperature_override is not None:
            temperature = self.temperature_override

        v = q_values.copy()
        v[~valid] = -1e9
        v = v / temperature
        v = v - np.max(v)
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()
        q_action = int(np.random.choice(len(probs), p=probs))

        return q_action, q_confidence, q_spread, recency, best_q_norm

    def _get_sensor_proxy(self) -> Optional[dict]:
        if self.controller._prev_sensor_data is not None:
            return self.controller._prev_sensor_data
        return None

    def _get_heuristic_action(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> int:
        return self.controller._choose_action_heuristic(
            state=state,
            current_pose=current_pose,
            sensor_data=sensor_data,
        )

    def on_episode_end(self, success: bool):
        level = self._current_level
        counts = Counter(self._current_episode_sources)
        if counts:
            dominant = counts.most_common(1)[0][0]
            if dominant == "q_store":
                self._level_q_results[level].append(success)
            elif dominant == "blend":
                self._level_blend_results[level].append(success)
            elif dominant == "sac":
                self._level_sac_results[level].append(success)
            elif dominant == "heuristic":
                self._level_heuristic_results[level].append(success)
        self._current_episode_sources = []

    @property
    def q_success_rate(self) -> float:
        all_results = []
        for results in self._level_q_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    @property
    def sac_success_rate(self) -> float:
        all_results = []
        for results in self._level_sac_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    @property
    def heuristic_success_rate(self) -> float:
        all_results = []
        for results in self._level_heuristic_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    @property
    def blend_success_rate(self) -> float:
        all_results = []
        for results in self._level_blend_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    def get_stats(self) -> Dict[str, Any]:
        total = max(self.stats["total_decisions"], 1)

        agreement_rate = (
            self.agreement_count / max(self.both_proposed_count, 1)
        )
        q_conf_mean = (
            float(np.mean(self.q_confidence_history))
            if self.q_confidence_history
            else 0.0
        )
        q_spread_mean = (
            float(np.mean(self.q_spread_history))
            if self.q_spread_history
            else 0.0
        )

        def _top_actions(action_dict, n=15):
            sorted_actions = sorted(
                action_dict.items(), key=lambda x: -x[1]
            )
            total_actions = max(sum(action_dict.values()), 1)
            return {
                name: {
                    "count": count,
                    "rate": round(count / total_actions, 3),
                }
                for name, count in sorted_actions[:n]
            }

        level_stats = {}
        for level in sorted(
            set(
                list(self._level_q_results.keys())
                + list(self._level_sac_results.keys())
                + list(self._level_blend_results.keys())
                + list(self._level_heuristic_results.keys())
            )
        ):
            total_lvl = max(self._level_total_decisions.get(level, 0), 1)
            h_lvl = self._level_heuristic_decisions.get(level, 0)
            q_r, s_r, b_r, h_r, best_ml, worst_ml, _ = self._get_level_tracks(level)
            ml_track = (best_ml + worst_ml) * 0.5
            
            # Per-step counts for this level
            step_counts = self._level_step_counts[level]
            step_total = max(step_counts["total"], 1)
            
            level_stats[f"level_{level}"] = {
                "q_rate": round(q_r, 3),
                "sac_rate": round(s_r, 3),
                "blend_rate": round(b_r, 3),
                "heuristic_rate": round(h_r, 3),
                "q_evals": len(self._level_q_results[level]),
                "sac_evals": len(self._level_sac_results[level]),
                "blend_evals": len(self._level_blend_results[level]),
                "heuristic_evals": len(self._level_heuristic_results[level]),
                "heuristic_budget_used": round(h_lvl / total_lvl, 3),
                "heuristic_eps": round(self._get_heuristic_eps(ml_track, h_r), 4),
                "ml_trend_increasing": self._is_ml_trend_increasing(level),
                # NEW: per-step distribution
                "per_step": {
                    "q_store": round(step_counts["q_store"] / step_total, 3),
                    "sac": round(step_counts["sac"] / step_total, 3),
                    "blend": round(step_counts["blend"] / step_total, 3),
                    "heuristic": round(step_counts["heuristic"] / step_total, 3),
                    "total_steps": step_counts["total"],
                },
            }

        total_agreements = max(sum(self._agreement_categories.values()), 1)
        agreement_analysis = {
            cat: {
                "count": count,
                "rate": round(count / total_agreements, 4),
            }
            for cat, count in sorted(
                self._agreement_categories.items(),
                key=lambda x: -x[1],
            )
        }

        # Disagreement summary
        disagree_log = list(self._disagreement_log)
        q_chosen_disagrees = [d for d in disagree_log if d["chosen"] == "q_chosen"]
        sac_chosen_disagrees = [d for d in disagree_log if d["chosen"] == "sac_chosen"]

        disagreement_summary = {
            "total_logged": len(disagree_log),
            "q_chosen_count": len(q_chosen_disagrees),
            "sac_chosen_count": len(sac_chosen_disagrees),
            "q_chosen_mean_confidence": round(
                float(np.mean([d["q_confidence"] for d in q_chosen_disagrees]))
                if q_chosen_disagrees else 0.0, 3,
            ),
            "q_chosen_mean_recency": round(
                float(np.mean([d["q_recency"] for d in q_chosen_disagrees]))
                if q_chosen_disagrees else 0.0, 3,
            ),
            "by_category": dict(Counter(d["category"] for d in disagree_log)),
        }

        # ═══ Q diagnostics ═══
        q_diag_global = {
            "confidence_mean": round(
                self._q_diag_confidence_sum / max(self._q_diag_confidence_count, 1), 4
            ),
            "reliable_pct": round(
                self._q_diag_reliable_count / max(self._q_diag_total_count, 1), 4
            ),
            "total_queries": self._q_diag_total_count,
        }

        q_diag_per_level = {}
        for lvl in sorted(self._q_diag_per_level.keys()):
            d = self._q_diag_per_level[lvl]
            cc = max(d["comp_count"], 1)
            q_diag_per_level[f"level_{lvl}"] = {
                "confidence_mean": round(
                    d["confidence_sum"] / max(d["confidence_count"], 1), 4
                ),
                "reliable_pct": round(
                    d["reliable_count"] / max(d["total_count"], 1), 4
                ),
                "total_queries": d["total_count"],
                "confidence_breakdown": {
                    "proximity_mean": round(d["distance_comp_sum"] / cc, 4),
                    "experience_mean": round(d["count_comp_sum"] / cc, 4),
                    "consistency_mean": round(d["consistency_comp_sum"] / cc, 4),
                    "recency_mean": round(d["recency_comp_sum"] / cc, 4),
                    "top2_spread_mean": round(d["top2_spread_comp_sum"] / cc, 4),
                },
            }

        return {
            "total_decisions": self.stats["total_decisions"],
            "q_store_rate": round(self.stats["q_store_chosen"] / total, 3),
            "sac_rate": round(self.stats["sac_chosen"] / total, 3),
            "blend_rate": round(self.stats.get("blend_chosen", 0) / total, 3),
            "heuristic_rate": round(
                self.stats["heuristic_chosen"] / total, 3
            ),
            "q_success_rate": round(self.q_success_rate, 3),
            "sac_success_rate": round(self.sac_success_rate, 3),
            "heuristic_success_rate": round(self.heuristic_success_rate, 3),
            "blend_success_rate": round(self.blend_success_rate, 3),
            "agreement_rate": round(agreement_rate, 3),
            "q_confidence_mean": round(q_conf_mean, 3),
            "q_spread_mean": round(q_spread_mean, 3),
            "is_calibrating": False,
            "current_level": self._current_level,
            "per_level_track_record": level_stats,
            "arbitrage_only": {
                "total_decisions": self._arbitrage_stats["total_decisions"],
                "q_store_rate": round(
                    self._arbitrage_stats["q_store_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
                "sac_rate": round(
                    self._arbitrage_stats["sac_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
                "blend_rate": round(
                    self._arbitrage_stats["blend_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
                "heuristic_rate": round(
                    self._arbitrage_stats["heuristic_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
            },
            "q_proposed_top": _top_actions(self.q_proposed_actions),
            "sac_proposed_top": _top_actions(self.sac_proposed_actions),
            "q_chosen_top": _top_actions(self.q_chosen_actions),
            "sac_chosen_top": _top_actions(self.sac_chosen_actions),
            "blend_chosen_top": _top_actions(self.blend_chosen_actions),
            "agreement_analysis": agreement_analysis,
            "disagreement_summary": disagreement_summary,
            "recent_disagreements": disagree_log[-20:],  # last 20 for inspection
            "q_diagnostics": q_diag_global,
            "q_diagnostics_per_level": q_diag_per_level,
        }
    