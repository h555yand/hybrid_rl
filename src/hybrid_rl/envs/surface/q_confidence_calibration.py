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

"""QConfidenceCalibration: calibrates Q-store confidence against
episode outcomes.

Collects median Q confidence per episode (separate for surface
and free-space steps) and maps it to episode success rate.
Produces reliable thresholds used by the Arbitrator to decide
when Q-store predictions can be trusted.

Usage:
    # During eval:
    cal = QConfidenceCalibration()
    for episode in episodes:
        for step in steps:
            cal.record_step(confidence_info, on_object)
        cal.record_episode_end(success)
    cal.save("q_calibration.json")

    # In Arbitrator:
    cal = QConfidenceCalibration.load("q_calibration.json")
    thresholds = cal.get_reliable_thresholds()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class QConfidenceCalibration:
    """Calibrate Q-store confidence components against episode outcomes."""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

        # Episode-level overall bins (surface vs free)
        self.surface_bins: List[List[bool]] = [
            [] for _ in range(num_bins)
        ]
        self.free_bins: List[List[bool]] = [
            [] for _ in range(num_bins)
        ]

        # Per-component bins
        self.proximity_bins = [[] for _ in range(num_bins)]
        self.experience_bins = [[] for _ in range(num_bins)]
        self.consistency_bins = [[] for _ in range(num_bins)]
        self.recency_bins = [[] for _ in range(num_bins)]
        self.v_mean_bins = [[] for _ in range(num_bins)]
        self.v_max_bins = [[] for _ in range(num_bins)]
        self.v_spread_bins = [[] for _ in range(num_bins)]
        self.best_q_norm_bins = [[] for _ in range(num_bins)]

        # Current episode accumulators
        self._current_surface_confs: List[float] = []
        self._current_free_confs: List[float] = []
        self._current_proximities: List[float] = []
        self._current_experiences: List[float] = []
        self._current_consistencies: List[float] = []
        self._current_recencies: List[float] = []
        self._current_v_means: List[float] = []
        self._current_v_maxs: List[float] = []
        self._current_v_spreads: List[float] = []
        self._current_best_q_norms: List[float] = []

        # V-value scales (from Q-store stats)
        self._current_v_max_scale: float = 1.0
        self._current_v_spread_scale: float = 1.0
        self._current_v_mean_scale: float = 1.0

        # Counters
        self._total_episodes: int = 0
        self._total_steps: int = 0

    def record_step(
        self, confidence_info: dict, on_object: bool
    ) -> None:
        """Record all confidence components for one step."""
        self._total_steps += 1
        overall = confidence_info.get("overall", 0.0)

        if on_object:
            self._current_surface_confs.append(overall)
        else:
            self._current_free_confs.append(overall)

        self._current_proximities.append(
            confidence_info.get("proximity", 0.0)
        )
        self._current_experiences.append(
            confidence_info.get("experience", 0.0)
        )
        self._current_consistencies.append(
            confidence_info.get("consistency", 0.0)
        )
        self._current_recencies.append(
            confidence_info.get("recency", 0.0)
        )
        self._current_v_means.append(
            confidence_info.get("v_mean", 0.0)
        )
        self._current_v_maxs.append(
            confidence_info.get("v_max", 0.0)
        )
        self._current_v_spreads.append(
            confidence_info.get("v_spread", 0.0)
        )
        self._current_best_q_norms.append(
            confidence_info.get("best_q_norm", 0.0)
        )

        # Save scales (overwrite — same for all steps)
        self._current_v_max_scale = confidence_info.get(
            "v_max_scale", self._current_v_max_scale
        )
        self._current_v_spread_scale = confidence_info.get(
            "v_spread_scale", self._current_v_spread_scale
        )
        self._current_v_mean_scale = confidence_info.get(
            "v_mean_scale", self._current_v_mean_scale
        )

    def record_episode_end(self, success: bool) -> None:
        """Bin median of each component against episode outcome."""
        self._total_episodes += 1

        # Overall (surface / free)
        if self._current_surface_confs:
            median = float(np.median(self._current_surface_confs))
            bin_idx = min(
                int(median * self.num_bins), self.num_bins - 1
            )
            self.surface_bins[bin_idx].append(success)

        if self._current_free_confs:
            median = float(np.median(self._current_free_confs))
            bin_idx = min(
                int(median * self.num_bins), self.num_bins - 1
            )
            self.free_bins[bin_idx].append(success)

        # Per-component (0-1 scale)
        for values, bins in [
            (self._current_proximities, self.proximity_bins),
            (self._current_experiences, self.experience_bins),
            (self._current_consistencies, self.consistency_bins),
            (self._current_recencies, self.recency_bins),
        ]:
            if values:
                median = float(np.median(values))
                bin_idx = min(
                    int(median * self.num_bins), self.num_bins - 1
                )
                bins[bin_idx].append(success)

        # V_mean: scale from Q-store stats
        if self._current_v_means:
            median_v = float(np.median(self._current_v_means))
            scale = max(self._current_v_mean_scale, 0.1)
            v_scaled = np.clip(median_v / scale, 0.0, 1.0)
            bin_idx = min(
                int(v_scaled * self.num_bins), self.num_bins - 1
            )
            self.v_mean_bins[bin_idx].append(success)

        # V_max: scale from Q-store stats
        if self._current_v_maxs:
            median_v_max = float(np.median(self._current_v_maxs))
            scale = max(self._current_v_max_scale, 0.1)
            v_max_scaled = np.clip(median_v_max / scale, 0.0, 1.0)
            bin_idx = min(
                int(v_max_scaled * self.num_bins), self.num_bins - 1
            )
            self.v_max_bins[bin_idx].append(success)

        # V_spread: scale from Q-store stats
        if self._current_v_spreads:
            median_v_spread = float(np.median(self._current_v_spreads))
            scale = max(self._current_v_spread_scale, 0.1)
            v_spread_scaled = np.clip(
                median_v_spread / scale, 0.0, 1.0
            )
            bin_idx = min(
                int(v_spread_scaled * self.num_bins), self.num_bins - 1
            )
            self.v_spread_bins[bin_idx].append(success)

        # best_Q_norm: z-score, range [-3, +3] → [0, 1]
        if self._current_best_q_norms:
            median_bqn = float(np.median(self._current_best_q_norms))
            bqn_scaled = np.clip((median_bqn + 3.0) / 6.0, 0.0, 1.0)
            bin_idx = min(
                int(bqn_scaled * self.num_bins), self.num_bins - 1
            )
            self.best_q_norm_bins[bin_idx].append(success)

        # Reset accumulators
        self._current_surface_confs = []
        self._current_free_confs = []
        self._current_proximities = []
        self._current_experiences = []
        self._current_consistencies = []
        self._current_recencies = []
        self._current_v_means = []
        self._current_v_maxs = []
        self._current_v_spreads = []
        self._current_best_q_norms = []

    def get_calibration(self) -> Dict[str, Any]:
        """Get full calibration data."""

        def _bins_to_dict(bins):
            result = {}
            for i, outcomes in enumerate(bins):
                if outcomes:
                    lo = i / self.num_bins
                    hi = (i + 1) / self.num_bins
                    result[f"{lo:.1f}-{hi:.1f}"] = {
                        "count": len(outcomes),
                        "success_rate": round(
                            sum(outcomes) / len(outcomes), 3
                        ),
                    }
            return result

        def _v_bins_to_dict(bins, scale):
            """V bins with real-value labels."""
            result = {}
            for i, outcomes in enumerate(bins):
                if outcomes:
                    lo = i / self.num_bins * scale
                    hi = (i + 1) / self.num_bins * scale
                    result[f"{lo:.2f}-{hi:.2f}"] = {
                        "count": len(outcomes),
                        "success_rate": round(
                            sum(outcomes) / len(outcomes), 3
                        ),
                    }
            return result

        def _bqn_bins_to_dict(bins):
            """best_Q_norm bins: scaled [0,1] → z-score [-3, +3]."""
            result = {}
            for i, outcomes in enumerate(bins):
                if outcomes:
                    lo = i / self.num_bins * 6.0 - 3.0
                    hi = (i + 1) / self.num_bins * 6.0 - 3.0
                    result[f"{lo:.1f}-{hi:.1f}"] = {
                        "count": len(outcomes),
                        "success_rate": round(
                            sum(outcomes) / len(outcomes), 3
                        ),
                    }
            return result

        return {
            "surface": _bins_to_dict(self.surface_bins),
            "free": _bins_to_dict(self.free_bins),
            "by_component": {
                "proximity": _bins_to_dict(self.proximity_bins),
                "experience": _bins_to_dict(self.experience_bins),
                "consistency": _bins_to_dict(self.consistency_bins),
                "recency": _bins_to_dict(self.recency_bins),
                "v_mean": _v_bins_to_dict(
                    self.v_mean_bins, self._current_v_mean_scale
                ),
                "v_max": _v_bins_to_dict(
                    self.v_max_bins, self._current_v_max_scale
                ),
                "v_spread": _v_bins_to_dict(
                    self.v_spread_bins, self._current_v_spread_scale
                ),
                "best_q_norm": _bqn_bins_to_dict(
                    self.best_q_norm_bins
                ),
            },
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
        }

    def get_reliable_thresholds(
        self,
        min_success_rate: float = 0.6,
        min_samples: int = 10,
    ) -> Dict[str, float]:
        """Find lowest confidence where success rate >= threshold."""

        def _find(bins: List[List[bool]]) -> float:
            for i, outcomes in enumerate(bins):
                if len(outcomes) >= min_samples:
                    sr = sum(outcomes) / len(outcomes)
                    if sr >= min_success_rate:
                        return round(i / self.num_bins, 2)
            return 0.9

        return {
            "surface": _find(self.surface_bins),
            "free": _find(self.free_bins),
        }

    def merge(self, other: "QConfidenceCalibration") -> None:
        """Merge another calibration into this one."""
        for i in range(self.num_bins):
            self.surface_bins[i].extend(other.surface_bins[i])
            self.free_bins[i].extend(other.free_bins[i])
            self.proximity_bins[i].extend(other.proximity_bins[i])
            self.experience_bins[i].extend(other.experience_bins[i])
            self.consistency_bins[i].extend(other.consistency_bins[i])
            self.recency_bins[i].extend(other.recency_bins[i])
            self.v_mean_bins[i].extend(other.v_mean_bins[i])
            self.v_max_bins[i].extend(other.v_max_bins[i])
            self.v_spread_bins[i].extend(other.v_spread_bins[i])
            self.best_q_norm_bins[i].extend(
                other.best_q_norm_bins[i]
            )
        self._total_episodes += other._total_episodes
        self._total_steps += other._total_steps

    def save(self, path: str) -> None:
        """Save calibration to JSON file."""
        data = {
            "calibration": self.get_calibration(),
            "reliable_thresholds": self.get_reliable_thresholds(),
            "num_bins": self.num_bins,
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "QConfidenceCalibration saved to %s "
            "(%d episodes, %d steps, thresholds=%s)",
            path,
            self._total_episodes,
            self._total_steps,
            data["reliable_thresholds"],
        )

    @classmethod
    def load(cls, path: str) -> "QConfidenceCalibration":
        """Load calibration from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        num_bins = data.get("num_bins", 10)
        cal = cls(num_bins=num_bins)
        cal._thresholds = data.get(
            "reliable_thresholds",
            {"surface": 0.7, "free": 0.7},
        )
        cal._calibration_data = data.get("calibration", {})
        cal._total_episodes = cal._calibration_data.get(
            "total_episodes", 0
        )
        cal._total_steps = cal._calibration_data.get(
            "total_steps", 0
        )

        for store_name, bins in [
            ("surface", cal.surface_bins),
            ("free", cal.free_bins),
        ]:
            store_data = cal._calibration_data.get(store_name, {})
            for bin_label, bin_info in store_data.items():
                try:
                    lo = float(bin_label.split("-")[0])
                    bin_idx = min(int(lo * num_bins), num_bins - 1)
                    count = bin_info.get("count", 0)
                    sr = bin_info.get("success_rate", 0.0)
                    successes = int(round(count * sr))
                    failures = count - successes
                    bins[bin_idx] = (
                        [True] * successes + [False] * failures
                    )
                except (ValueError, IndexError):
                    continue

        logger.info(
            "QConfidenceCalibration loaded from %s "
            "(%d episodes, thresholds=%s)",
            path,
            cal._total_episodes,
            cal._thresholds,
        )
        return cal

    @property
    def surface_threshold(self) -> float:
        if hasattr(self, "_thresholds"):
            return self._thresholds.get("surface", 0.7)
        return self.get_reliable_thresholds()["surface"]

    @property
    def free_threshold(self) -> float:
        if hasattr(self, "_thresholds"):
            return self._thresholds.get("free", 0.7)
        return self.get_reliable_thresholds()["free"]

    def get_threshold_for_state(self, on_object: bool) -> float:
        return (
            self.surface_threshold if on_object
            else self.free_threshold
        )

    def __repr__(self) -> str:
        return (
            f"QConfidenceCalibration("
            f"episodes={self._total_episodes}, "
            f"steps={self._total_steps}, "
            f"surface_thresh={self.surface_threshold:.2f}, "
            f"free_thresh={self.free_threshold:.2f})"
        )
