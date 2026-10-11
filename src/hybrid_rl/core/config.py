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

"""Minimal config stub for HNSW state store.

HNSW store uses DEFAULT_CONFIG for default values when
domain-specific config doesn't provide them.
"""

DEFAULT_CONFIG = {
    "state_dim": 14,
    "num_actions": 9,
    "max_points": 100000,
    "k_neighbors": 7,
    "sigma": 1.0,
    "insert_threshold": 0.5,
    "adaptive_sigma": True,
    "evict_fraction": 0.1,
    "auto_calibrate": False,
    "calibration_percentile": 10.0,
    "min_calibration_samples": 100,
    "min_weight_threshold": 0.01,
    "norm_warmup_steps": 3000,
    "norm_min_std": 1e-4,
    "rebuild_on_freeze": True,
    "feature_weights": None,
}