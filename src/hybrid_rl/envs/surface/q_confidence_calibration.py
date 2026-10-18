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
            cal.record_step(confidence, on_object)
        cal.record_episode_end(success)
    cal.save("q_calibration.json")

    # In Arbitrator:
    cal = QConfidenceCalibration.load("q_calibration.json")
    thresholds = cal.get_reliable_thresholds()
    # thresholds = {"surface": 0.6, "free": 0.4}
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QConfidenceCalibration:
    """Calibrate Q-store confidence against episode outcomes.

    For each eval episode, records per-step Q confidence
    (separated by surface vs free-space), then at episode end
    bins the median confidence and records success/failure.

    Produces a calibration curve and reliable thresholds:
    the minimum confidence at which episode success rate
    exceeds a target (default 60%).

    Attributes:
        num_bins: Number of confidence bins (default 10,
            giving 0.0-0.1, 0.1-0.2, ..., 0.9-1.0).
        surface_bins: Per-bin list of episode outcomes
            for surface-dominant episodes.
        free_bins: Per-bin list of episode outcomes
            for free-space-dominant episodes.
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

        # Episode-level bins: median confidence → success/fail
        self.surface_bins: List[List[bool]] = [
            [] for _ in range(num_bins)
        ]
        self.free_bins: List[List[bool]] = [
            [] for _ in range(num_bins)
        ]

        # Current episode accumulators
        self._current_surface_confs: List[float] = []
        self._current_free_confs: List[float] = []

        # Counters
        self._total_episodes: int = 0
        self._total_steps: int = 0

    def record_step(
        self, confidence: float, on_object: bool
    ) -> None:
        """Record Q confidence for one step.

        Call this every step during evaluation, before
        the controller executes the action.

        Args:
            confidence: Q-store confidence for current state
                (from store.get_q_values_with_confidence).
            on_object: Whether agent is on object surface.
        """
        self._total_steps += 1
        if on_object:
            self._current_surface_confs.append(confidence)
        else:
            self._current_free_confs.append(confidence)

    def record_episode_end(self, success: bool) -> None:
        """Record episode outcome and bin median confidences.

        Call this at the end of each episode, after determining
        whether the episode was successful.

        Args:
            success: Whether the episode reached the goal.
        """
        self._total_episodes += 1

        if self._current_surface_confs:
            median = float(
                np.median(self._current_surface_confs)
            )
            bin_idx = min(
                int(median * self.num_bins),
                self.num_bins - 1,
            )
            self.surface_bins[bin_idx].append(success)

        if self._current_free_confs:
            median = float(
                np.median(self._current_free_confs)
            )
            bin_idx = min(
                int(median * self.num_bins),
                self.num_bins - 1,
            )
            self.free_bins[bin_idx].append(success)

        self._current_surface_confs = []
        self._current_free_confs = []

    def get_calibration(self) -> Dict[str, Any]:
        """Get full calibration data.

        Returns:
            Dict with surface and free calibration curves,
            plus total episode/step counts.
        """
        result: Dict[str, Any] = {}

        for name, bins in [
            ("surface", self.surface_bins),
            ("free", self.free_bins),
        ]:
            bin_data: Dict[str, Any] = {}
            for i, outcomes in enumerate(bins):
                if outcomes:
                    lo = i / self.num_bins
                    hi = (i + 1) / self.num_bins
                    bin_data[f"{lo:.1f}-{hi:.1f}"] = {
                        "count": len(outcomes),
                        "success_rate": round(
                            sum(outcomes) / len(outcomes), 3
                        ),
                    }
            result[name] = bin_data

        result["total_episodes"] = self._total_episodes
        result["total_steps"] = self._total_steps

        return result

    def get_reliable_thresholds(
        self, min_success_rate: float = 0.6,
        min_samples: int = 10,
    ) -> Dict[str, float]:
        """Find lowest confidence where success rate >= threshold.

        Scans bins from low to high confidence. Returns the lower
        bound of the first bin where success rate meets the target
        and has enough samples.

        Args:
            min_success_rate: Minimum success rate to consider
                a confidence level "reliable".
            min_samples: Minimum number of episodes in a bin
                to trust the success rate estimate.

        Returns:
            Dict with "surface" and "free" threshold values.
        """
        def _find(bins: List[List[bool]]) -> float:
            for i, outcomes in enumerate(bins):
                if len(outcomes) >= min_samples:
                    sr = sum(outcomes) / len(outcomes)
                    if sr >= min_success_rate:
                        return round(i / self.num_bins, 2)
            return 0.9  # fallback: very conservative

        return {
            "surface": _find(self.surface_bins),
            "free": _find(self.free_bins),
        }

    def save(self, path: str) -> None:
        """Save calibration to JSON file.

        Saves calibration curves, thresholds, and metadata.

        Args:
            path: File path for the JSON output.
        """
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
        """Load calibration from JSON file.

        Args:
            path: Path to the calibration JSON file.

        Returns:
            QConfidenceCalibration instance with loaded thresholds.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        num_bins = data.get("num_bins", 10)
        cal = cls(num_bins=num_bins)

        # Restore thresholds
        cal._thresholds = data.get(
            "reliable_thresholds",
            {"surface": 0.7, "free": 0.7},
        )

        # Restore calibration data for inspection
        cal._calibration_data = data.get("calibration", {})
        cal._total_episodes = cal._calibration_data.get(
            "total_episodes", 0
        )
        cal._total_steps = cal._calibration_data.get(
            "total_steps", 0
        )

        # Rebuild bins from saved data for further analysis
        for store_name, bins in [
            ("surface", cal.surface_bins),
            ("free", cal.free_bins),
        ]:
            store_data = cal._calibration_data.get(
                store_name, {}
            )
            for bin_label, bin_info in store_data.items():
                # Parse "0.6-0.7" → bin index 6
                try:
                    lo = float(bin_label.split("-")[0])
                    bin_idx = min(
                        int(lo * num_bins), num_bins - 1
                    )
                    count = bin_info.get("count", 0)
                    sr = bin_info.get("success_rate", 0.0)
                    successes = int(round(count * sr))
                    failures = count - successes
                    bins[bin_idx] = (
                        [True] * successes
                        + [False] * failures
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
        """Reliable confidence threshold for surface Q-store."""
        if hasattr(self, "_thresholds"):
            return self._thresholds.get("surface", 0.7)
        return self.get_reliable_thresholds()["surface"]

    @property
    def free_threshold(self) -> float:
        """Reliable confidence threshold for free-space Q-store."""
        if hasattr(self, "_thresholds"):
            return self._thresholds.get("free", 0.7)
        return self.get_reliable_thresholds()["free"]

    def get_threshold_for_state(
        self, on_object: bool
    ) -> float:
        """Get appropriate threshold based on agent state.

        Args:
            on_object: Whether agent is on object surface.

        Returns:
            Confidence threshold for the current Q-store.
        """
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
