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

"""RL Goal Approach Experiment — Hydra-compatible experiment class.

Orchestrates the full RL navigation pipeline: Q-learning training,
evaluation, Behavioral Cloning, SAC training, and adaptive mode.
Configured via Hydra YAML.
"""

from __future__ import annotations

import datetime
import json
import logging
import pickle
import time
import types
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
from typing_extensions import Self
import argparse
import yaml
from omegaconf import DictConfig, OmegaConf

from .ablation_runner import (
    run_episodes,
    run_eval_per_seed,
)
from .action_interpreter import ActionInterpreter
from .adaptive_manager import AdaptiveTrainingManager
from .behavioral_cloning import BCTrainer
from .config import DEFAULT_CONFIG
from .episode_pools import get_or_generate_pools
from .experience_extractor import ExperienceExtractor
from .lightweight_env import LightweightEnv
from .mesh_factory import prepare_demo_meshes
from .rl_goal_approach_controller import (
    RLGoalApproachController,
    RunningQStats
)
from .sac_trainer import PSACTrainer
from .ablation_runner import _maybe_save_visualization, visualize_agent_goal
from .strategic_sac import StrategicSAC, StrategicBCTrainer
from .episode_pools import _is_reachable_by_surface
from .arbitrator import sac_to_discrete
from .action_interpreter import ActionInterpreter
from .ycb_utils import convert_ycb_objects
from .visualize_env import EpisodeVisualizer
from .episode_pools import (
    _is_on_wall, _get_handle_info, _check_handle_between
)


logger = logging.getLogger(__name__)

_ADAPTIVE_MIN_DIST = 10.0
_ADAPTIVE_MAX_DIST = 120.0
_ADAPTIVE_LOG_INTERVAL = 100
_SAC_WARMUP_STEPS = 5000
_BC_NUM_EPOCHS = 200
# _EVAL_EPSILON = 0.02
_COLLISION_DEPTH_THRESHOLD = 0.5


class RLGoalApproachExperiment:
    """Hydra-compatible experiment for RL goal approach training.

    Supports stages: Q-learning train, eval, BC train, SAC train,
    SAC eval, and adaptive mode. All parameters configured via YAML.

    Model storage convention::

        runs/
        ├── q_store_seed_11/     # Q-learning model
        ├── sac_seed_11/         # SAC model
        ├── bc_model/            # BC model (no seed)
        ├── adaptive_q_seed_11/  # Adaptive Q-store
        ├── adaptive_sac_seed_11/# Adaptive SAC
        └── meta/                # Training metadata

    Usage::

        python run.py experiment=rl_goal_approach
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize experiment from Hydra config.

        Args:
            config: Hydra DictConfig with experiment parameters.
        """
        self.config: dict[str, Any] = (
            OmegaConf.to_object(config)
            if isinstance(config, DictConfig)
            else config
        )

        self.seed = self.config.get("seed", 42)
        self.output_dir = Path(
            self.config["logging"]["output_dir"]
        )
        self.run_name = self.config["logging"]["run_name"]

        raw_vis = self.config.get("visualise", False)
        if raw_vis is False or raw_vis is None:
            self.visualise = None
        else:
            self.visualise = str(raw_vis)

        # Directories
        self.data_dir = self.output_dir / "data"
        self.runs_dir = self.data_dir / "runs"
        self.scripts_dir = self.data_dir / "episode_scripts"

        # Pipeline stages
        self.do_train = self.config.get("do_train", False)
        self.do_eval = self.config.get("do_eval", False)
        self.do_heur_eval = self.config.get("do_heur_eval", False)
        self.do_bc_train = self.config.get("do_bc_train", False)
        self.do_sac_train = self.config.get(
            "do_sac_train", False
        )
        self.do_sac_eval = self.config.get(
            "do_sac_eval", False
        )
        self.do_adaptive = self.config.get(
            "do_adaptive", False
        )

        # Training config
        self.training_stages: list[dict[str, Any]] = (
            self.config.get("training_stages", [])
        )
        self.curriculum_levels: list[tuple[float, float]] = [
            tuple(level)
            for level in self.config.get(
                "curriculum_levels",
                [[10.0, 40.0], [20.0, 80.0], [40.0, 120.0]],
            )
        ]
        self.curriculum_filters: list[dict[str, Any]] = (
            self.config.get(
                "curriculum_filters", []
            )
        )
        self.train_seeds: list[int] = self.config.get(
            "train_seeds", [11]
        )
        self.eval_seeds: list[int] = self.config.get(
            "eval_seeds", [44]
        )
        self.sac_eval_seeds: list[int] = self.config.get(
            "sac_eval_seeds", [77, 88, 99]
        )
        self.sac_seed: int = self.config.get(
            "sac_seed", self.train_seeds[0]
        )
        self.adapt_seed: int = self.config.get(
            "adapt_seed", 42
        )
        self.eval_episodes_per_level: int = self.config.get(
            "eval_episodes_per_level", 500
        )
        self.sac_eval_episodes_per_level: int = self.config.get(
            "sac_eval_episodes_per_level", 500
        )
        self.regenerate_scripts: bool = self.config.get(
            "regenerate_scripts", True
        )

        # RL config
        self.rl_config: dict[str, Any] = {
            **DEFAULT_CONFIG,
            **self.config.get("rl_config", {}),
        }

        self.promote_threshold: float = self.config.get(
            "promote_threshold", 0.55
        )
        self.promote_window: int = self.config.get(
            "promote_window", 100
        )

        # SAC config
        self.sac_config: dict[str, Any] = self.config.get(
            "sac_config", {}
        )
        self.sac_meshes: list[str] = self.config.get(
            "sac_meshes",
            ["cube", "cylinder", "mug", "cup"],
        )
        self.sac_episodes_per_mesh: dict[str, int] = (
            self.config.get(
                "sac_episodes_per_mesh",
                {
                    "cube": 1000,
                    "cylinder": 1500,
                    "mug": 2000,
                    "cup": 2000,
                },
            )
        )

        # Adaptive config
        self.adaptive_mesh: str = self.config.get(
            "adaptive_mesh", "cup"
        )
        self.adaptive_episodes: int = self.config.get(
            "adaptive_episodes", 3000
        )

        # Eval meshes
        self.eval_meshes: list[str] = self.config.get(
            "eval_meshes",
            ["cube", "cylinder", "mug", "cup"],
        )
        # BC mesh filter: which meshes to include in BC training data
        # If not specified, uses all eval_meshes (backward compatible)
        self.bc_mesh_names: set[str] = set(
            self.config.get(
                "bc_mesh_names",
                [],  # empty = use all eval_meshes
            )
        )
        # YCB eval config
        self.do_ycb_eval = self.config.get("do_ycb_eval", False)
        self.ycb_config = self.config.get("ycb_config", {})
        self.do_ycb_mujoco_eval = self.config.get("do_ycb_mujoco_eval", False)
        self.ycb_mujoco_config = self.config.get("ycb_config", {})
        self.do_ycb_mujoco_adaptive = self.config.get("do_ycb_mujoco_adaptive", False)
        self.ycb_mujoco_adaptive_config = self.config.get("ycb_mujoco_adaptive_config", {})

    # ══════════════════════════════════════════════════════
    # Model path helpers
    # ══════════════════════════════════════════════════════

    def _q_model_dir(self, seed: int) -> str:
        """Get Q-store directory path for a seed.

        Args:
            seed: Training seed.

        Returns:
            Path string to Q-store directory.
        """
        return str(self.runs_dir / f"q_store_seed_{seed}")

    def _sac_model_dir(self, seed: int) -> str:
        """Get SAC model directory path for a seed.

        Args:
            seed: Training seed.

        Returns:
            Path string to SAC model directory.
        """
        return str(self.runs_dir / f"sac_seed_{seed}")

    def _resolve_load_dir(
        self,
        load_mode: str | None,
        model_type: str,
        seed: int,
    ) -> str | None:
        """Resolve load_mode to actual directory path.

        Args:
            load_mode: null (from scratch), "auto" (standard path),
                or explicit path string.
            model_type: "q_store" or "sac".
            seed: Training seed.

        Returns:
            Directory path or None.
        """
        if load_mode is None:
            return None
        if load_mode == "auto":
            if model_type == "q_store":
                return self._q_model_dir(seed)
            if model_type == "sac":
                return self._sac_model_dir(seed)
            return None
        return str(load_mode)

    def _save_meta(
        self,
        model_type: str,
        seed: int | None,
        stats: dict[str, Any],
        meshes: list[str],
    ) -> None:
        """Save training metadata to meta/ directory.

        Args:
            model_type: "q_store", "sac", or "bc".
            seed: Training seed (None for BC).
            stats: Training statistics.
            meshes: List of meshes trained on.
        """
        meta_dir = self.runs_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        meta_name = (
            f"{model_type}_seed_{seed}.json"
            if seed is not None
            else f"{model_type}.json"
        )
        meta = {
            "model_type": model_type,
            "seed": seed,
            "created": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "meshes": meshes,
            "stats": stats,
        }
        meta_path = meta_dir / meta_name
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Meta saved to %s", meta_path)

    @staticmethod
    def _balance_bc_transitions(
        transitions: list,
        total_target: int = 100000,
        mesh_weights: dict[str, float] | None = None,
        level_weights: dict[int, float] | None = None,
        min_per_group: int = 50,
        action_min_pct: float = 0.01,
        action_min_absolute: int = 100,
    ) -> list:
        """Balance BC transitions by (mesh, level) with
        complexity weights, then balance action types on
        the full dataset.

        Pipeline:
        1. Group by (mesh_id, level)
        2. Weighted sample per group
        3. Concatenate
        4. Balance rare action types on full dataset
        5. Shuffle

        Args:
            transitions: All BC transitions.
            total_target: Target size before action balance.
            mesh_weights: Per-mesh complexity weights.
            level_weights: Per-level difficulty weights.
            min_per_group: Minimum samples to include group.
            action_min_pct: Min fraction for action balance.
            action_min_absolute: Min count for action balance.

        Returns:
            Balanced transitions.
        """
        if mesh_weights is None:
            mesh_weights = {}
        if level_weights is None:
            level_weights = {0: 1.0, 1: 1.5, 2: 2.0}

        type_names = (
            ExperienceExtractor.get_type_names()
        )
        mesh_id_to_name = {
            v: k
            for k, v in (
                ExperienceExtractor
                .MESH_NAME_TO_ID.items()
            )
        }

        # ═══ Step 1: Group by (mesh_id, level) ═══
        groups: dict[tuple[int, int], list] = {}
        for tr in transitions:
            mesh_id = getattr(tr, "mesh_id", -1)
            level = getattr(tr, "level", 0)
            key = (mesh_id, level)
            if key not in groups:
                groups[key] = []
            groups[key].append(tr)

        # ═══ Step 2: Log before ═══
        logger.info(
            "BC balance BEFORE (%d total):",
            len(transitions),
        )
        for (mid, level), trs in sorted(
            groups.items()
        ):
            mname = mesh_id_to_name.get(
                mid, f"mesh_{mid}"
            )
            action_counts: dict[str, int] = {}
            for tr in trs:
                aname = type_names.get(
                    tr.action_type,
                    f"type_{tr.action_type}",
                )
                action_counts[aname] = (
                    action_counts.get(aname, 0) + 1
                )
            top_actions = dict(
                sorted(
                    action_counts.items(),
                    key=lambda x: -x[1],
                )[:5]
            )
            logger.info(
                "  %s/L%d: %d transitions, "
                "actions: %s",
                mname,
                level,
                len(trs),
                top_actions,
            )

        # ═══ Step 3: Compute weighted targets ═══
        group_weights: dict[
            tuple[int, int], float
        ] = {}
        for mid, level in groups:
            mname = mesh_id_to_name.get(
                mid, f"mesh_{mid}"
            )
            mw = mesh_weights.get(mname, 1.0)
            lw = level_weights.get(level, 1.0)
            group_weights[(mid, level)] = mw * lw

        total_weight = max(
            sum(group_weights.values()), 1e-8
        )

        group_targets: dict[
            tuple[int, int], int
        ] = {}
        for key, weight in group_weights.items():
            group_targets[key] = max(
                min_per_group,
                int(
                    total_target
                    * weight
                    / total_weight
                ),
            )

        # ═══ Step 4: Weighted sample per group ═══
        after_weighted: list = []
        for (mid, level), trs in sorted(
            groups.items()
        ):
            mname = mesh_id_to_name.get(
                mid, f"mesh_{mid}"
            )
            target = group_targets[(mid, level)]

            if len(trs) < min_per_group:
                after_weighted.extend(trs)
                logger.info(
                    "  %s/L%d: kept all %d (too few)",
                    mname,
                    level,
                    len(trs),
                )
                continue

            if len(trs) > target:
                indices = np.random.permutation(
                    len(trs)
                )[:target]
                sampled = [trs[i] for i in indices]
            elif len(trs) < target:
                indices = np.random.choice(
                    len(trs), target, replace=True
                )
                sampled = [trs[i] for i in indices]
            else:
                sampled = list(trs)

            after_weighted.extend(sampled)

            mw = mesh_weights.get(mname, 1.0)
            lw = level_weights.get(level, 1.0)
            logger.info(
                "  %s/L%d: %d → %d "
                "(target=%d, mw=%.1f, lw=%.1f)",
                mname,
                level,
                len(trs),
                len(sampled),
                target,
                mw,
                lw,
            )

        logger.info(
            "After weighted sampling: "
            "%d transitions",
            len(after_weighted),
        )

        # ═══ Step 5: Balance action types ═══
        balanced = (
            RLGoalApproachExperiment
            ._balance_by_action_type(
                after_weighted,
                min_pct=action_min_pct,
                min_absolute=action_min_absolute,
            )
        )

        np.random.shuffle(balanced)

        # ═══ Step 6: Log after ═══
        after_mesh: dict[str, int] = {}
        after_level: dict[int, int] = {}
        after_action: dict[str, int] = {}
        for tr in balanced:
            mname = mesh_id_to_name.get(
                getattr(tr, "mesh_id", -1),
                "unknown",
            )
            level = getattr(tr, "level", 0)
            aname = type_names.get(
                tr.action_type,
                f"type_{tr.action_type}",
            )
            after_mesh[mname] = (
                after_mesh.get(mname, 0) + 1
            )
            after_level[level] = (
                after_level.get(level, 0) + 1
            )
            after_action[aname] = (
                after_action.get(aname, 0) + 1
            )

        total = max(len(balanced), 1)
        logger.info(
            "BC balance AFTER (%d total):",
            len(balanced),
        )

        logger.info("  By mesh:")
        for mname in sorted(after_mesh.keys()):
            count = after_mesh[mname]
            logger.info(
                "    %s: %d (%.1f%%)",
                mname,
                count,
                100.0 * count / total,
            )

        logger.info("  By level:")
        for level in sorted(after_level.keys()):
            count = after_level[level]
            logger.info(
                "    L%d: %d (%.1f%%)",
                level,
                count,
                100.0 * count / total,
            )

        logger.info("  By action type:")
        for aname in sorted(
            after_action.keys(),
            key=lambda x: -after_action[x],
        ):
            count = after_action[aname]
            logger.info(
                "    %s: %d (%.1f%%)",
                aname,
                count,
                100.0 * count / total,
            )

        logger.info(
            "BC balance: %d → %d transitions",
            len(transitions),
            len(balanced),
        )
        return balanced
    
    @staticmethod
    def _balance_by_action_type(
        transitions: list,
        min_pct: float = 0.01,
        min_absolute: int = 100,
    ) -> list:
        """Ensure minimum representation of each
        action type.

        Each type gets at least max(min_absolute,
        total * min_pct) examples.
        Only oversamples types that already have
        >= min_absolute examples.

        Args:
            transitions: BC transitions.
            min_pct: Minimum fraction of total (0.01=1%).
            min_absolute: Minimum raw examples to
                qualify for oversampling.

        Returns:
            Balanced transitions.
        """
        type_names = (
            ExperienceExtractor.get_type_names()
        )

        by_type: dict[int, list] = {}
        for tr in transitions:
            tid = tr.action_type
            if tid not in by_type:
                by_type[tid] = []
            by_type[tid].append(tr)

        total = len(transitions)
        min_target = max(
            min_absolute,
            int(total * min_pct),
        )

        logger.info(
            "Action type balance: total=%d, "
            "min_target=%d (%.1f%% or %d)",
            total,
            min_target,
            min_pct * 100,
            min_absolute,
        )
        logger.info("  BEFORE:")
        for tid in sorted(by_type.keys()):
            name = type_names.get(
                tid, f"type_{tid}"
            )
            logger.info(
                "    %s: %d (%.2f%%)",
                name,
                len(by_type[tid]),
                100.0 * len(by_type[tid])
                / max(total, 1),
            )

        balanced = list(transitions)
        for tid, trs in by_type.items():
            name = type_names.get(
                tid, f"type_{tid}"
            )
            if (
                len(trs) < min_target
                and len(trs) >= min_absolute
            ):
                need = min_target - len(trs)
                indices = np.random.choice(
                    len(trs), need, replace=True
                )
                extra = [trs[i] for i in indices]
                balanced.extend(extra)
                logger.info(
                    "    %s: oversampled %d → %d",
                    name,
                    len(trs),
                    min_target,
                )
            elif len(trs) < min_absolute:
                logger.info(
                    "    %s: skipped (%d < %d "
                    "min_absolute)",
                    name,
                    len(trs),
                    min_absolute,
                )

        logger.info(
            "  Action type balance: %d → %d",
            total,
            len(balanced),
        )

        return balanced
    # ══════════════════════════════════════════════════════
    # Context manager
    # ══════════════════════════════════════════════════════

    def __enter__(self) -> Self:
        """Set up experiment directories and demo meshes.

        Returns:
            Self for use in with-statement.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        prepare_demo_meshes(self.data_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> bool:
        """Clean up on context exit.

        Returns:
            False to not suppress exceptions.
        """
        return False

    def run(self) -> None:
        """Run all enabled pipeline stages."""
        start_time = time.time()

        if self.do_train:
            self._run_train()
        if self.do_eval:
            self._run_eval()
        if self.do_heur_eval:
            self._run_heuristic_eval()
        if self.do_bc_train:
            self._run_bc_train()
        if self.do_sac_train:
            self._run_sac_train()
        if self.do_sac_eval:
            self._run_sac_eval()
        if self.do_adaptive:
            self._run_adaptive()
        if self.do_ycb_eval:
            self._run_ycb_eval()
        if self.do_ycb_mujoco_eval:
            self._run_ycb_mujoco_eval()
        if self.do_ycb_mujoco_adaptive:
            self._run_ycb_mujoco_adaptive()
        if self.config.get("do_sleep", False):
            self._run_sleep()

        elapsed = time.time() - start_time
        logger.info("Experiment complete in %.1fs", elapsed)

    # ══════════════════════════════════════════════════════
    # Q-learning training
    # ══════════════════════════════════════════════════════

    def _run_train(self) -> None:
        """Run Q-learning training across all stages."""
        logger.info("=" * 60)
        logger.info("Q-Learning Training (all stages)")
        logger.info("=" * 60)

        trained_meshes: list[str] = []

        for stage_idx, stage in enumerate(
            self.training_stages
        ):
            mesh_name = stage["mesh"]
            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            episodes = stage["episodes"]
            epsilon_start = stage["epsilon_start"]
            load_mode = stage.get("load_mode")
            pool_seed = stage.get("pool_seed", None)

            # Per-stage overrides with global fallback
            stage_curriculum_levels = stage.get(
                "curriculum_levels", self.curriculum_levels
            )
            stage_curriculum_filters = stage.get(
                "curriculum_filters", self.curriculum_filters
            )
            stage_warmup = stage.get(
                "warmup_episodes",
                self.rl_config.get("warmup_episodes", 100),
            )
            stage_promote_threshold = stage.get(
                "promote_threshold", self.promote_threshold
            )
            stage_promote_window = stage.get(
                "promote_window", self.promote_window
            )

            logger.info(
                "STAGE %d/%d: %s, episodes=%d, eps=%.2f, "
                "load_mode=%s, pool_seed=%s, "
                "warmup=%d, promote=%.2f/%d",
                stage_idx + 1,
                len(self.training_stages),
                mesh_name,
                episodes,
                epsilon_start,
                load_mode,
                pool_seed,
                stage_warmup,
                stage_promote_threshold,
                stage_promote_window,
            )

            stage_cfg = {
                **self.rl_config,
                "epsilon_start": epsilon_start,
                "epsilon_min": stage.get(
                    "epsilon_min", 0.05
                ),
                "num_episodes": episodes,
                "warmup_episodes": stage_warmup,
                "unfreeze_normalization": (
                    load_mode is not None
                ),
                # Strategic epsilon follows tactical
                "strategic_epsilon_start": stage.get(
                    "strategic_epsilon_start",
                    epsilon_start,
                ),
                "strategic_epsilon_min": stage.get(
                    "strategic_epsilon_min",
                    stage.get("epsilon_min", 0.05),
                ),
            }

            # Determine seeds and prefix for pool generation
            if pool_seed is not None:
                pool_seeds = [pool_seed]
                pool_prefix = (
                    f"train_{mesh_name}_ps{pool_seed}"
                )
            else:
                pool_seeds = self.train_seeds
                pool_prefix = f"train_{mesh_name}"

            train_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=pool_seeds,
                episodes_per_level=episodes,
                scripts_dir=self.scripts_dir,
                curriculum_levels=stage_curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=pool_prefix,
                curriculum_filters=stage_curriculum_filters,
            )

            for seed in self.train_seeds:
                save_dir = self._q_model_dir(seed)
                load_dir = self._resolve_load_dir(
                    load_mode, "q_store", seed
                )

                lookup_seed = (
                    pool_seed
                    if pool_seed is not None
                    else seed
                )
                seed_pools = train_pools.get(lookup_seed)
                episode_pools = (
                    seed_pools.get("levels")
                    if seed_pools
                    else None
                )

                curriculum_config = {
                    "levels": stage_curriculum_levels,
                    "filters": stage_curriculum_filters,
                    "promote_threshold": (
                        stage_promote_threshold
                    ),
                    "promote_window": (
                        stage_promote_window
                    ),
                }

                run_result = run_episodes(
                    mesh_dir=str(self.data_dir),
                    save_dir=save_dir,
                    load_dir=load_dir,
                    num_episodes=episodes,
                    config=stage_cfg,
                    mesh_path=mesh_path,
                    seed=seed,
                    return_metrics=True,
                    curriculum_config=curriculum_config,
                    episode_pools=episode_pools,
                    visualise=self.visualise,
                )

                stage_output = (
                    self.data_dir
                    / f"train_result_{mesh_name}"
                    f"_seed_{seed}"
                    f"_stage_{stage_idx}.json"
                )
                stage_data = {
                    "stage": stage_idx,
                    "mesh": mesh_name,
                    "seed": seed,
                    "pool_seed": pool_seed,
                    "epsilon_start": epsilon_start,
                    "epsilon_min": stage.get(
                        "epsilon_min", 0.05
                    ),
                    "load_mode": load_mode,
                    "episodes": episodes,
                    "warmup_episodes": stage_warmup,
                    "promote_threshold": (
                        stage_promote_threshold
                    ),
                    "promote_window": (
                        stage_promote_window
                    ),
                    "curriculum_levels": (
                        stage_curriculum_levels
                    ),
                    "curriculum_filters": (
                        stage_curriculum_filters
                    ),
                    "success_rate": run_result.get(
                        "success_rate"
                    ),
                    "stats": run_result.get("stats"),
                    "curriculum_stats": run_result.get(
                        "curriculum_stats"
                    ),
                }
                with stage_output.open("w") as f:
                    json.dump(stage_data, f, indent=2)
                logger.info(
                    "Stage result: success_rate=%.4f",
                    run_result.get("success_rate", 0),
                )

            trained_meshes.append(mesh_name)

        for seed in self.train_seeds:
            self._save_meta(
                "q_store",
                seed,
                {"trained_meshes": trained_meshes},
                trained_meshes,
            )

    # ══════════════════════════════════════════════════════
    # Q-learning evaluation
    # ══════════════════════════════════════════════════════

    def _run_eval(self) -> None:
        """Run Q-learning eval on all meshes, save per-mesh BC data."""
        logger.info("=" * 60)
        logger.info("Eval on all meshes")
        logger.info("=" * 60)

        eval_cfg = {
            **self.rl_config,
            "mode": "eval",
        }

        all_eval_results: dict[str, Any] = {}

        # ═══ NEW: directory for per-mesh BC data ═══
        bc_per_mesh_dir = self.data_dir / "bc_per_mesh"
        bc_per_mesh_dir.mkdir(parents=True, exist_ok=True)

        for mesh_entry in self.eval_meshes:
            if isinstance(mesh_entry, dict):
                mesh_name = mesh_entry["mesh"]
                eval_pool_seed = mesh_entry.get("pool_seed", None)
                eval_curriculum_filters = mesh_entry.get(
                    "curriculum_filters", self.curriculum_filters
                )
                eval_curriculum_levels = mesh_entry.get(
                    "curriculum_levels", self.curriculum_levels
                )
            else:
                mesh_name = mesh_entry
                eval_pool_seed = None
                eval_curriculum_filters = self.curriculum_filters
                eval_curriculum_levels = self.curriculum_levels

            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            logger.info(
                "Evaluating: %s (pool_seed=%s)",
                mesh_name,
                eval_pool_seed,
            )

            if eval_pool_seed is not None:
                pool_seeds = [eval_pool_seed]
                pool_prefix = f"eval_{mesh_name}_ps{eval_pool_seed}"
            else:
                pool_seeds = self.eval_seeds
                pool_prefix = f"eval_{mesh_name}"

            eval_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=pool_seeds,
                episodes_per_level=(
                    self.eval_episodes_per_level
                ),
                scripts_dir=self.scripts_dir,
                curriculum_levels=eval_curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=pool_prefix,
                curriculum_filters=eval_curriculum_filters,
            )

            if eval_pool_seed is not None:
                remapped_pools: dict[int, Any] = {}
                for es in self.eval_seeds:
                    remapped_pools[es] = eval_pools[eval_pool_seed]
                eval_pools = remapped_pools

            eval_results, bc_transitions = (
                run_eval_per_seed(
                    data_dir=self.data_dir,
                    runs_dir=self.runs_dir,
                    mesh_path=mesh_path,
                    train_seeds=self.train_seeds,
                    eval_seeds=self.eval_seeds,
                    variant="q_store",
                    eval_cfg=eval_cfg,
                    eval_pools=eval_pools,
                    collect_bc=True,
                    episodes_per_level=(
                        self.eval_episodes_per_level
                    ),
                    mesh_name=mesh_name,
                    visualise=self.visualise,
                )
            )

            all_eval_results[mesh_name] = eval_results

            # ═══ NEW: save per-mesh BC data ═══
            if bc_transitions:
                bc_mesh_path = (
                    bc_per_mesh_dir
                    / f"bc_qstore_{mesh_name}.pkl"
                )
                with bc_mesh_path.open("wb") as f:
                    pickle.dump(bc_transitions, f)
                logger.info(
                    "  %s: %d transitions saved to %s",
                    mesh_name,
                    len(bc_transitions),
                    bc_mesh_path,
                )

            mesh_output = (
                self.data_dir
                / f"eval_result_{mesh_name}.json"
            )
            with mesh_output.open("w") as f:
                json.dump(eval_results, f, indent=2)

        eval_output = self.data_dir / "eval_result_all.json"
        with eval_output.open("w") as f:
            json.dump(all_eval_results, f, indent=2)
            
    def _run_heuristic_eval(self) -> list[Any]:
        """Run heuristic eval on all meshes, save per-mesh BC data."""
        logger.info("=" * 60)
        logger.info("Heuristic Expert Data Collection")
        logger.info("=" * 60)

        heuristic_cfg = {
            **self.rl_config,
            "mode": "eval",
            "eval_epsilon": 1.0,
            "temperature_override": 0.01,
            "strategic_eval_epsilon": 1.0,
            "air_start_enabled": False,
            "air_start_in_eval": False,
        }

        all_heuristic_results: dict[str, Any] = {}

        bc_per_mesh_dir = self.data_dir / "bc_per_mesh"
        bc_per_mesh_dir.mkdir(parents=True, exist_ok=True)

        for mesh_entry in self.eval_meshes:
            if isinstance(mesh_entry, dict):
                mesh_name = mesh_entry["mesh"]
                eval_pool_seed = mesh_entry.get("pool_seed", None)
                eval_curriculum_filters = mesh_entry.get(
                    "curriculum_filters", self.curriculum_filters
                )
                eval_curriculum_levels = mesh_entry.get(
                    "curriculum_levels", self.curriculum_levels
                )
            else:
                mesh_name = mesh_entry
                eval_pool_seed = None
                eval_curriculum_filters = self.curriculum_filters
                eval_curriculum_levels = self.curriculum_levels

            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            logger.info(
                "Heuristic eval: %s (pool_seed=%s)",
                mesh_name,
                eval_pool_seed,
            )

            if eval_pool_seed is not None:
                pool_seeds = [eval_pool_seed]
                pool_prefix = (
                    f"heuristic_{mesh_name}_ps{eval_pool_seed}"
                )
            else:
                pool_seeds = self.eval_seeds
                pool_prefix = f"heuristic_{mesh_name}"

            heuristic_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=pool_seeds,
                episodes_per_level=(
                    self.eval_episodes_per_level
                ),
                scripts_dir=self.scripts_dir,
                curriculum_levels=eval_curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=pool_prefix,
                curriculum_filters=eval_curriculum_filters,
            )

            if eval_pool_seed is not None:
                remapped_pools: dict[int, Any] = {}
                for es in self.eval_seeds:
                    remapped_pools[es] = heuristic_pools[
                        eval_pool_seed
                    ]
                heuristic_pools = remapped_pools

            heuristic_results, heuristic_transitions = (
                run_eval_per_seed(
                    data_dir=self.data_dir,
                    runs_dir=self.runs_dir,
                    mesh_path=mesh_path,
                    train_seeds=self.train_seeds,
                    eval_seeds=self.eval_seeds,
                    variant="q_store",
                    eval_cfg=heuristic_cfg,
                    eval_pools=heuristic_pools,
                    collect_bc=True,
                    episodes_per_level=(
                        self.eval_episodes_per_level
                    ),
                    mesh_name=mesh_name,
                    visualise=self.visualise,
                    log_prefix="heuristic",
                )
            )

            all_heuristic_results[mesh_name] = heuristic_results

            # ═══ NEW: save per-mesh heuristic BC data ═══
            if heuristic_transitions:
                bc_mesh_path = (
                    bc_per_mesh_dir
                    / f"bc_heuristic_{mesh_name}.pkl"
                )
                with bc_mesh_path.open("wb") as f:
                    pickle.dump(heuristic_transitions, f)
                logger.info(
                    "Heuristic %s: %d transitions saved to %s",
                    mesh_name,
                    len(heuristic_transitions),
                    bc_mesh_path,
                )

        # Save heuristic eval results
        heuristic_eval_path = (
            self.data_dir / "heuristic_eval_result_all.json"
        )
        with heuristic_eval_path.open("w") as f:
            json.dump(all_heuristic_results, f, indent=2)
        logger.info(
            "Heuristic eval results saved to %s",
            heuristic_eval_path,
        )

        # ═══ REMOVED: no more monolithic bc_data_heuristic.pkl ═══
        # Per-mesh files are in bc_per_mesh/ directory
        return []  # no need to return transitions anymore

    # ══════════════════════════════════════════════════════
    # Behavioral Cloning
    # ══════════════════════════════════════════════════════

    def _run_bc_train(self) -> None:
        """Train BC model from per-mesh eval data."""
        logger.info("=" * 60)
        logger.info("BC Training")
        logger.info("=" * 60)

        bc_data_combined_path = (
            self.data_dir / "bc_data_combined.pkl"
        )
        if bc_data_combined_path.exists():
            logger.info(
                "Loading cached bc_data_combined from %s",
                bc_data_combined_path,
            )
            with bc_data_combined_path.open("rb") as f:
                bc_transitions = pickle.load(f)
            logger.info(
                "Loaded %d cached transitions",
                len(bc_transitions),
            )
        else:
            # ═══ Load per-mesh BC data ═══
            bc_per_mesh_dir = (
                self.data_dir / "bc_per_mesh"
            )
            if not bc_per_mesh_dir.exists():
                logger.error(
                    "bc_per_mesh/ not found. "
                    "Run do_eval and do_heur_eval first."
                )
                return

            bc_mesh_names = self.bc_mesh_names
            heuristic_transitions = []
            q_store_transitions = []

            # Discover available files
            available_heuristic = sorted(
                bc_per_mesh_dir.glob(
                    "bc_heuristic_*.pkl"
                )
            )
            available_qstore = sorted(
                bc_per_mesh_dir.glob(
                    "bc_qstore_*.pkl"
                )
            )

            logger.info(
                "BC per-mesh directory: %s",
                bc_per_mesh_dir,
            )
            logger.info(
                "  Available heuristic: %s",
                [p.stem for p in available_heuristic],
            )
            logger.info(
                "  Available Q-store: %s",
                [p.stem for p in available_qstore],
            )
            logger.info(
                "  BC mesh filter: %s",
                (
                    list(bc_mesh_names)
                    if bc_mesh_names
                    else "ALL"
                ),
            )

            # Load heuristic data (primary source)
            for pkl_path in available_heuristic:
                mesh_name = pkl_path.stem.replace(
                    "bc_heuristic_", ""
                )
                if (
                    bc_mesh_names
                    and mesh_name not in bc_mesh_names
                ):
                    logger.info(
                        "  SKIP heuristic %s (held-out)",
                        mesh_name,
                    )
                    continue
                with pkl_path.open("rb") as f:
                    transitions = pickle.load(f)
                heuristic_transitions.extend(transitions)
                logger.info(
                    "  LOAD heuristic %s: %d transitions",
                    mesh_name,
                    len(transitions),
                )

            # Load Q-store data (optional fraction)
            q_store_fraction = float(
                self.config.get(
                    "bc_q_store_fraction", 0.0
                )
            )
            q_store_transitions_used = []

            if q_store_fraction > 0.0:
                for pkl_path in available_qstore:
                    mesh_name = pkl_path.stem.replace(
                        "bc_qstore_", ""
                    )
                    if (
                        bc_mesh_names
                        and mesh_name
                        not in bc_mesh_names
                    ):
                        logger.info(
                            "  SKIP Q-store %s (held-out)",
                            mesh_name,
                        )
                        continue
                    with pkl_path.open("rb") as f:
                        transitions = pickle.load(f)
                    q_store_transitions.extend(
                        transitions
                    )
                    logger.info(
                        "  LOAD Q-store %s: "
                        "%d transitions",
                        mesh_name,
                        len(transitions),
                    )

                max_q = int(
                    len(heuristic_transitions)
                    * q_store_fraction
                )
                if len(q_store_transitions) > max_q:
                    indices = np.random.permutation(
                        len(q_store_transitions)
                    )[:max_q]
                    q_store_transitions_used = [
                        q_store_transitions[i]
                        for i in indices
                    ]
                else:
                    q_store_transitions_used = (
                        q_store_transitions
                    )
                logger.info(
                    "Q-store data: %d → %d "
                    "(fraction=%.1f%%)",
                    len(q_store_transitions),
                    len(q_store_transitions_used),
                    q_store_fraction * 100,
                )

            # ═══ Combine ═══
            bc_transitions = (
                heuristic_transitions
                + q_store_transitions_used
            )
            logger.info(
                "BC data: %d heuristic + %d Q-store "
                "= %d total",
                len(heuristic_transitions),
                len(q_store_transitions_used),
                len(bc_transitions),
            )

            # ═══ Balance by (mesh, level) + action types ═══
            bc_total_target = self.config.get(
                "bc_total_target", None
            )

            bc_transitions = self._balance_bc_transitions(
                bc_transitions,
                total_target=(
                    int(bc_total_target)
                    if bc_total_target is not None
                    else len(bc_transitions)
                ),
                mesh_weights=self.config.get(
                    "bc_mesh_weights", {}
                ),
                level_weights={
                    int(k): float(v)
                    for k, v in self.config.get(
                        "bc_level_weights",
                        {0: 1.0, 1: 1.5, 2: 2.0},
                    ).items()
                },
                min_per_group=int(
                    self.config.get("bc_min_per_group", 50)
                ),
                action_min_pct=float(
                    self.config.get("bc_action_min_pct", 0.01)
                ),
                action_min_absolute=int(
                    self.config.get(
                        "bc_action_min_absolute", 100
                    )
                ),
            )

            np.random.shuffle(bc_transitions)

            logger.info(
                "BC data: %d total after balancing",
                len(bc_transitions),
            )
            # ═══ Save detailed BC composition report ═══
            mesh_id_to_name = {
                v: k
                for k, v in (
                    ExperienceExtractor
                    .MESH_NAME_TO_ID.items()
                )
            }
            type_names = ExperienceExtractor.get_type_names()

            bc_composition = {}
            for tr in bc_transitions:
                mname = mesh_id_to_name.get(
                    getattr(tr, "mesh_id", -1), "unknown"
                )
                level = getattr(tr, "level", 0)
                key = f"{mname}/L{level}"
                if key not in bc_composition:
                    bc_composition[key] = {
                        "count": 0,
                        "actions": {},
                        "mesh": mname,
                        "level": level,
                    }
                bc_composition[key]["count"] += 1
                aname = type_names.get(
                    tr.action_type,
                    f"type_{tr.action_type}",
                )
                bc_composition[key]["actions"][aname] = (
                    bc_composition[key]["actions"]
                    .get(aname, 0) + 1
                )

            total = max(len(bc_transitions), 1)
            bc_report = {
                "total_transitions": len(bc_transitions),
                "has_next_state": sum(
                    1 for tr in bc_transitions
                    if tr.next_state is not None
                ),
                "missing_next_state": sum(
                    1 for tr in bc_transitions
                    if tr.next_state is None
                ),
                "per_group": {
                    k: {
                        **v,
                        "pct": round(
                            100.0 * v["count"] / total, 1
                        ),
                    }
                    for k, v in sorted(
                        bc_composition.items()
                    )
                },
                "per_mesh_total": {},
                "per_level_total": {},
                "per_action_total": {},
            }

            # Aggregate by mesh
            for k, v in bc_composition.items():
                mname = v["mesh"]
                bc_report["per_mesh_total"][mname] = (
                    bc_report["per_mesh_total"]
                    .get(mname, 0) + v["count"]
                )
            # Aggregate by level
            for k, v in bc_composition.items():
                lkey = f"L{v['level']}"
                bc_report["per_level_total"][lkey] = (
                    bc_report["per_level_total"]
                    .get(lkey, 0) + v["count"]
                )
            # Aggregate by action
            for k, v in bc_composition.items():
                for aname, count in v["actions"].items():
                    bc_report["per_action_total"][aname] = (
                        bc_report["per_action_total"]
                        .get(aname, 0) + count
                    )

            bc_composition_path = (
                self.data_dir / "bc_composition_report.json"
            )
            with bc_composition_path.open("w") as f:
                json.dump(bc_report, f, indent=2)
            logger.info(
                "BC composition report saved to %s",
                bc_composition_path,
            )

            # Save combined
            with bc_data_combined_path.open("wb") as f:
                pickle.dump(bc_transitions, f)

        # ═══ Train BC ═══
        num_types = len(
            ExperienceExtractor.get_type_names()
        )
        trainer = BCTrainer(
            state_dim=self.rl_config.get(
                "state_dim", 23
            ),
            num_types=num_types,
        )
        trainer.train(
            bc_transitions, num_epochs=_BC_NUM_EPOCHS
        )
        trainer.save(str(self.runs_dir / "bc_model"))

        bc_stats = trainer.get_training_stats(
            bc_transitions
        )
        bc_stats_path = (
            self.data_dir / "bc_train_result.json"
        )
        with bc_stats_path.open("w") as f:
            json.dump(bc_stats, f, indent=2)

        # ═══ Strategic BC Training ═══
        logger.info("=" * 40)
        logger.info(
            "Strategic BC Training (two separate)"
        )
        logger.info("=" * 40)

        for seed in self.train_seeds:
            q_dir = self._q_model_dir(seed)
            if not (
                Path(q_dir) / "config.json"
            ).exists():
                logger.warning(
                    "Q-store not found at %s, "
                    "skipping Strategic BC",
                    q_dir,
                )
                continue

            controller = (
                RLGoalApproachController.load(
                    q_dir,
                    agent_id=(
                        f"strategic_bc_seed_{seed}"
                    ),
                    config={
                        **self.rl_config,
                        "mode": "eval",
                    },
                )
            )

            strategic_bc_dir = str(
                self.runs_dir / "strategic_bc_model"
            )
            Path(strategic_bc_dir).mkdir(
                parents=True, exist_ok=True
            )

            # 1. Detach BC
            detach_bc = StrategicBCTrainer(state_dim=5)
            has_detach = (
                detach_bc.prepare_data_from_q_store(
                    controller.strategic_detach,
                )
            )
            if has_detach:
                detach_bc.train(num_epochs=100)
                torch.save(
                    detach_bc.get_actor_weights(),
                    Path(strategic_bc_dir)
                    / "strategic_detach_actor.pt",
                )
                s_mean, s_std = (
                    detach_bc.get_normalization()
                )
                np.savez(
                    Path(strategic_bc_dir)
                    / "strategic_detach_norm.npz",
                    state_mean=s_mean,
                    state_std=s_std,
                )
                logger.info(
                    "Strategic Detach BC saved to %s",
                    strategic_bc_dir,
                )
            else:
                logger.warning(
                    "No data for Strategic Detach BC"
                )

            # 2. Direction BC
            direction_bc = StrategicBCTrainer(
                state_dim=5
            )
            has_direction = (
                direction_bc.prepare_data_from_q_store(
                    controller.strategic_direction,
                )
            )
            if has_direction:
                direction_bc.train(num_epochs=100)
                torch.save(
                    direction_bc.get_actor_weights(),
                    Path(strategic_bc_dir)
                    / "strategic_direction_actor.pt",
                )
                s_mean, s_std = (
                    direction_bc.get_normalization()
                )
                np.savez(
                    Path(strategic_bc_dir)
                    / "strategic_direction_norm.npz",
                    state_mean=s_mean,
                    state_std=s_std,
                )
                logger.info(
                    "Strategic Direction BC saved "
                    "to %s",
                    strategic_bc_dir,
                )
            else:
                logger.warning(
                    "No data for Strategic "
                    "Direction BC"
                )

            # Save strategic BC stats
            strategic_bc_stats = {
                "seed": seed,
                "detach": {
                    "has_data": has_detach,
                    "store_points": len(
                        controller.strategic_detach
                        .points
                    ),
                    "train_size": (
                        len(detach_bc.train_states)
                        if has_detach
                        and hasattr(
                            detach_bc, "train_states"
                        )
                        else 0
                    ),
                    "val_size": (
                        len(detach_bc.val_states)
                        if has_detach
                        and hasattr(
                            detach_bc, "val_states"
                        )
                        else 0
                    ),
                },
                "direction": {
                    "has_data": has_direction,
                    "store_points": len(
                        controller.strategic_direction
                        .points
                    ),
                    "train_size": (
                        len(
                            direction_bc.train_states
                        )
                        if has_direction
                        and hasattr(
                            direction_bc,
                            "train_states",
                        )
                        else 0
                    ),
                    "val_size": (
                        len(direction_bc.val_states)
                        if has_direction
                        and hasattr(
                            direction_bc,
                            "val_states",
                        )
                        else 0
                    ),
                },
            }
            strategic_stats_path = (
                self.data_dir
                / "strategic_bc_train_result.json"
            )
            with strategic_stats_path.open("w") as f:
                json.dump(
                    strategic_bc_stats, f, indent=2
                )

        self._save_meta(
            "bc",
            None,
            {
                **bc_stats,
                "bc_mesh_names": list(
                    self.bc_mesh_names
                ),
                "heuristic_transitions": len(
                    bc_transitions
                ),
                "q_store_fraction": float(
                    self.config.get(
                        "bc_q_store_fraction", 0.0
                    )
                ),
            },
            (
                list(self.bc_mesh_names)
                if self.bc_mesh_names
                else list(self.eval_meshes)
            ),
        )
        logger.info(
            "BC complete: val_acc=%.4f, "
            "transitions=%d",
            bc_stats["val_accuracy"],
            bc_stats["total_transitions"],
        )
        
    # ══════════════════════════════════════════════════════
    # SAC training
    # ══════════════════════════════════════════════════════

    def _run_sac_train(self) -> None:
        """Train P-SAC model sequentially on all meshes."""
        logger.info("=" * 60)
        logger.info("P-SAC Training (seed=%d)", self.sac_seed)
        logger.info("=" * 60)

        num_types = len(
            ExperienceExtractor.get_type_names()
        )

        sac_cfg = {
            k: v
            for k, v in self.sac_config.items()
            if k != "load_mode"
        }
        trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 20),
            num_types=num_types,
            **sac_cfg,
        )

        load_mode = self.sac_config.get("load_mode")
        load_dir = self._resolve_load_dir(
            load_mode, "sac", self.sac_seed
        )

        if load_dir:
            trainer.load(load_dir)
            bc_combined_path = (
                self.data_dir
                / "bc_data_combined.pkl"
            )
            bc_path = (
                bc_combined_path
                if bc_combined_path.exists()
                else self.data_dir / "bc_data.pkl"
            )
            if bc_path.exists():
                with bc_path.open("rb") as f:
                    bc_data = pickle.load(f)
                
                bc_norm = (
                    trainer._normalize_bc_transitions(  # noqa: SLF001
                        bc_data
                    )
                )
                trainer.buffer.load_bc_data(bc_norm)
                trainer.bc_data = bc_data
            logger.info(
                "SAC loaded from %s for fine-tuning",
                load_dir,
            )
        else:
            # Use combined BC data (heuristic + optional Q-store)
            bc_combined_path = (
                self.data_dir / "bc_data_combined.pkl"
            )
            bc_data_path = (
                str(bc_combined_path)
                if bc_combined_path.exists()
                else str(
                    self.data_dir / "bc_data.pkl"
                )
            )
            trainer.load_bc(
                bc_model_dir=str(
                    self.runs_dir / "bc_model"
                ),
                bc_data_path=bc_data_path,
            )
            
        sac_model_dir = self._sac_model_dir(self.sac_seed)

        # ═══ Load Strategic SACs (only if enabled) ═══
        use_strategic = self.sac_config.get(
            "use_strategic_override", False
        )
        if use_strategic:
            # Load Strategic BC into controller's Strategic SAC
            # ═══ Load Strategic SACs from BC ═══
            strategic_bc_dir = str(
                self.runs_dir / "strategic_bc_model"
            )

            strategic_detach_sac = None
            strategic_direction_sac = None

            # Detach SAC
            detach_actor_path = (
                Path(strategic_bc_dir)
                / "strategic_detach_actor.pt"
            )
            if detach_actor_path.exists():
                strategic_detach_sac = StrategicSAC(
                    state_dim=5
                )
                strategic_detach_sac.actor.load_state_dict(
                    torch.load(
                        detach_actor_path,
                        weights_only=True,
                    )
                )
                norm = np.load(
                    Path(strategic_bc_dir)
                    / "strategic_detach_norm.npz"
                )
                strategic_detach_sac._state_mean = (
                    norm["state_mean"]
                )
                strategic_detach_sac._state_std = (
                    norm["state_std"]
                )
                strategic_detach_sac._norm_frozen = True

                # Warm start buffer from Q-store
                for seed in self.train_seeds:
                    q_dir = self._q_model_dir(seed)
                    if (
                        Path(q_dir) / "config.json"
                    ).exists():
                        q_ctrl = (
                            RLGoalApproachController.load(
                                q_dir,
                                agent_id="strat_loader",
                                config={
                                    **self.rl_config,
                                    "mode": "eval",
                                },
                            )
                        )
                        strategic_detach_sac.warm_start_from_q_store(
                            q_ctrl.strategic_detach
                        )
                        break

                logger.info(
                    "Strategic Detach SAC loaded "
                    "(buf=%d)",
                    strategic_detach_sac.buffer_size,
                )

            # Direction SAC
            direction_actor_path = (
                Path(strategic_bc_dir)
                / "strategic_direction_actor.pt"
            )
            if direction_actor_path.exists():
                strategic_direction_sac = StrategicSAC(
                    state_dim=5
                )
                strategic_direction_sac.actor.load_state_dict(
                    torch.load(
                        direction_actor_path,
                        weights_only=True,
                    )
                )
                norm = np.load(
                    Path(strategic_bc_dir)
                    / "strategic_direction_norm.npz"
                )
                strategic_direction_sac._state_mean = (
                    norm["state_mean"]
                )
                strategic_direction_sac._state_std = (
                    norm["state_std"]
                )
                strategic_direction_sac._norm_frozen = True

                # Warm start buffer
                for seed in self.train_seeds:
                    q_dir = self._q_model_dir(seed)
                    if (
                        Path(q_dir) / "config.json"
                    ).exists():
                        q_ctrl = (
                            RLGoalApproachController.load(
                                q_dir,
                                agent_id="strat_loader2",
                                config={
                                    **self.rl_config,
                                    "mode": "eval",
                                },
                            )
                        )
                        strategic_direction_sac.warm_start_from_q_store(
                            q_ctrl.strategic_direction
                        )
                        break

                logger.info(
                    "Strategic Direction SAC loaded "
                    "(buf=%d)",
                    strategic_direction_sac.buffer_size,
                )

            # Pass to trainer
            trainer.load_strategic(
                strategic_detach_sac=strategic_detach_sac,
                strategic_direction_sac=(
                    strategic_direction_sac
                ),
            )

        for mesh_name, mesh_cfg in self.sac_episodes_per_mesh.items():
            # Support both formats:
            #   mug: 3000
            #   mug: {episodes: 3000, pool_seed: 301}
            if isinstance(mesh_cfg, dict):
                num_episodes = mesh_cfg["episodes"]
                pool_seed = mesh_cfg.get("pool_seed", self.sac_seed)
            else:
                num_episodes = mesh_cfg
                pool_seed = self.sac_seed

            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            logger.info(
                "SAC Training: %s, %d episodes, pool_seed=%d",
                mesh_name,
                num_episodes,
                pool_seed,
            )
            env = LightweightEnv(mesh_path, mesh_name=Path(mesh_path).stem)
            controller = RLGoalApproachController(
                agent_id=f"sac_{mesh_name}",
                config={**self.rl_config, "mode": "eval"},
            )

            sac_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=[pool_seed],
                episodes_per_level=num_episodes,
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"sac_train_{mesh_name}_ps{pool_seed}",
                curriculum_filters=self.curriculum_filters,
            )

            trainer.start_mesh_tracking(mesh_name)

            is_first = mesh_name == self.sac_meshes[0]
            trainer.train(
                env=env,
                controller=controller,
                num_episodes=num_episodes,
                warmup_steps=(
                    _SAC_WARMUP_STEPS
                    if is_first and not load_dir
                    else 0
                ),
                save_dir=sac_model_dir,
                curriculum_levels=self.curriculum_levels,
                promote_threshold=self.promote_threshold,
                promote_window=self.promote_window,
                episode_pools=sac_pools[pool_seed],
                visualise=self.visualise,
                mesh_name=mesh_name,
            )

            mesh_stat = trainer._get_current_mesh_stats()
            result_path = (
                self.data_dir
                / f"sac_train_result_{mesh_name}.json"
            )
            with result_path.open("w") as f:
                json.dump(mesh_stat, f, indent=2)
            logger.info(
                "SAC %s: rate=%.3f",
                mesh_name,
                mesh_stat.get("success_rate", 0),
            )

        final_stats = trainer.get_training_stats()
        final_path = (
            self.data_dir / "sac_train_result_all.json"
        )
        with final_path.open("w") as f:
            json.dump(final_stats, f, indent=2)

        self._save_meta(
            "sac",
            self.sac_seed,
            final_stats,
            list(self.sac_meshes),
        )
        logger.info(
            "SAC training complete. Saved to %s",
            sac_model_dir,
        )

    # ══════════════════════════════════════════════════════
    # SAC evaluation
    # ══════════════════════════════════════════════════════

    def _run_sac_eval(self) -> None:
        """Evaluate P-SAC model on all meshes."""
        logger.info("=" * 60)
        logger.info("P-SAC Eval (seed=%d)", self.sac_seed)
        logger.info("=" * 60)

        num_types = len(
            ExperienceExtractor.get_type_names()
        )
        sac_trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 20),
            num_types=num_types,
            max_steps_per_goal=int(self.sac_config.get("max_steps_per_goal", 500)),
            goal_threshold=float(self.sac_config.get("goal_threshold", 4.0)),
        )
        sac_trainer.load(
            self._sac_model_dir(self.sac_seed)
        )

        all_results: dict[str, Any] = {}

        for mesh_entry in self.eval_meshes:
            # ═══ Parse mesh entry (dict or string) ═══
            if isinstance(mesh_entry, dict):
                mesh_name = mesh_entry["mesh"]
                eval_pool_seed = mesh_entry.get(
                    "pool_seed", None
                )
                eval_curriculum_filters = mesh_entry.get(
                    "curriculum_filters",
                    self.curriculum_filters,
                )
                eval_curriculum_levels = mesh_entry.get(
                    "curriculum_levels",
                    self.curriculum_levels,
                )
            else:
                mesh_name = mesh_entry
                eval_pool_seed = None
                eval_curriculum_filters = self.curriculum_filters
                eval_curriculum_levels = self.curriculum_levels

            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            logger.info(
                "SAC Eval: %s (pool_seed=%s)",
                mesh_name,
                eval_pool_seed,
            )

            # ═══ Determine pool seeds and prefix ═══
            if eval_pool_seed is not None:
                pool_seeds = [eval_pool_seed]
                pool_prefix = (
                    f"sac_eval_{mesh_name}_ps{eval_pool_seed}"
                )
            else:
                pool_seeds = self.sac_eval_seeds
                pool_prefix = f"sac_eval_{mesh_name}"

            sac_eval_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=pool_seeds,
                episodes_per_level=(
                    self.sac_eval_episodes_per_level
                ),
                scripts_dir=self.scripts_dir,
                curriculum_levels=eval_curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=pool_prefix,
                curriculum_filters=eval_curriculum_filters,
            )

            # ═══ Remap pools if using shared pool_seed ═══
            if eval_pool_seed is not None:
                remapped_pools: dict[int, Any] = {}
                for es in self.sac_eval_seeds:
                    remapped_pools[es] = sac_eval_pools[
                        eval_pool_seed
                    ]
                sac_eval_pools = remapped_pools

            results = self._eval_sac_on_pools(
                sac_trainer=sac_trainer,
                sac_eval_pools=sac_eval_pools,
                mesh_path=mesh_path,
                curriculum_levels=eval_curriculum_levels,
                visualise=self.visualise,              # NEW
                save_dir=str(self.data_dir),           # NEW (куда складывать)
                mesh_name=mesh_name,                   # NEW (лейбл)
            )
            all_results[mesh_name] = results

            for key in sorted(results.keys()):
                data = results[key]
                logger.info(
                    "  %s (%smm): success=%.4f, "
                    "timeout=%.4f, collision=%.4f",
                    key,
                    data.get("bounds_mm", []),
                    data["success_rate"],
                    data["timeout_rate"],
                    data["collision_rate"],
                )

            eval_out = (
                self.data_dir
                / f"sac_eval_result_{mesh_name}.json"
            )
            with eval_out.open("w") as f:
                json.dump(results, f, indent=2)

        all_out = (
            self.data_dir / "sac_eval_result_all.json"
        )
        with all_out.open("w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(
            "Saved all SAC eval results to %s", all_out
        )

    def _eval_sac_on_pools(
        self,
        sac_trainer: PSACTrainer,
        sac_eval_pools: dict[int, dict[str, Any]],
        mesh_path: str,
        curriculum_levels: list[tuple[float, float]] | None = None,
        visualise=None,
        save_dir: str | None = None,
        mesh_name: str = "",
    ) -> dict[str, Any]:
        """Run SAC evaluation on episode pools + (optional) save EpisodeVisualizer logs.

        Visualisation/logging logic mirrors PSACTrainer.train():
        - create EpisodeVisualizer if visualise and save_dir
        - collect current_poses + action_explanations (+ optional short)
        - call visualizer.save_episode() for each episode
        """
        if curriculum_levels is None:
            curriculum_levels = self.curriculum_levels

        # ─────────────────────────────────────────────
        # Optional visualizer (same pattern as SAC train)
        # ─────────────────────────────────────────────
        visualizer = None
        if visualise and save_dir:
            from .visualize_env import EpisodeVisualizer

            visualizer = EpisodeVisualizer(
                output_dir=Path(save_dir),
                mesh_name=(mesh_name or Path(mesh_path).stem),
                stage="sac_eval",
                max_per_type_per_level=5,
                timeout_frame_interval=50,
                visualize_mode=visualise,
                num_levels=len(curriculum_levels) if curriculum_levels else 1,
            )

        type_names = ExperienceExtractor.get_type_names()
        results: dict[str, Any] = {}
        sample_seed = self.sac_eval_seeds[0]
        num_levels = len(sac_eval_pools[sample_seed].get("levels", []))

        # Global monotonically increasing episode index for visualizer filenames
        global_ep_idx = 0

        for level_idx in range(num_levels):
            successes = 0
            timeouts = 0
            collisions = 0
            total = 0

            action_counts: dict[int, int] = {}
            collision_counts: dict[int, int] = {}
            total_steps = 0

            episode_steps_list: list[int] = []
            success_steps_list: list[int] = []

            for eval_seed in self.sac_eval_seeds:
                pool = sac_eval_pools[eval_seed]["levels"][level_idx]

                np.random.seed(eval_seed)  # noqa: NPY002
                torch.manual_seed(eval_seed)

                env = LightweightEnv(mesh_path, seed=eval_seed, mesh_name=Path(mesh_path).stem)
                controller = RLGoalApproachController(
                    agent_id=f"sac_eval_L{level_idx}_{eval_seed}",
                    config={**self.rl_config, "mode": "eval"},
                )
                interpreter = ActionInterpreter(env)

                for ep_data in pool:
                    global_ep_idx += 1

                    start_pos = np.array(ep_data["start_pos"])
                    start_rot = np.array(ep_data["start_rot"])
                    env.reset(position=start_pos, rotation=start_rot)

                    goal_pose = np.concatenate(
                        [np.array(ep_data["goal_pos"]), np.array(ep_data["goal_rot"])]
                    )
                    controller.set_new_goal(goal_pose, start_pos)
                    env.set_goal(goal_pose)

                    success = False
                    collision = False

                    # Per-episode logs (like SAC train)
                    current_poses: list[np.ndarray] = [env.get_pose().copy()]
                    action_explanations: list[str] = []
                    actions_short: list[str] = []

                    for step_idx in range(sac_trainer.max_steps_per_goal):
                        pose = env.get_pose()
                        sensor = env.get_sensor_data()

                        state_raw = controller._compute_state(pose, sensor)
                        state = sac_trainer.normalize_state(state_raw)

                        # ─── Select action (same logic as your current SAC eval) ───
                        strategic_type = None
                        strategic_source = None

                        if sac_trainer.use_strategic_override:
                            strategic_type, strategic_source = sac_trainer._strategic_detach_check(
                                state_raw, controller, sensor
                            )

                        if strategic_type is not None:
                            atype = int(strategic_type)
                            aparams = np.zeros(3, dtype=np.float32)
                        else:
                            if sac_trainer.use_strategic_override:
                                dir_phase = sac_trainer._strategic_direction_check(
                                    state_raw, controller, sensor, pose
                                )
                                if dir_phase is not None:
                                    controller._current_phase = dir_phase

                            state_t = torch.FloatTensor(state.astype(np.float32)).unsqueeze(0)
                            with torch.no_grad():
                                at, ap, _, _ = sac_trainer.actor.sample_eval(state_t)
                            atype = int(at[0].item())
                            aparams = ap[0].numpy()

                            # Action masks
                            atype, aparams = sac_trainer._apply_action_masks(
                                atype, aparams, state_raw, state, controller
                            )

                        # ─── Per-step accounting ───
                        action_counts[atype] = action_counts.get(atype, 0) + 1
                        total_steps += 1

                        # ─── Log step (BEFORE executing action) ───
                        # (avoid UnboundLocalError and match "train-style" logging)
                        dist_before = float(np.linalg.norm(goal_pose[:3] - pose[:3]))
                        act_name = type_names.get(atype, f"type_{atype}")
                        param_str = self._format_action_params(atype, aparams)
                        state_debug = controller.get_state_debug_info(state_raw, pose, sensor)

                        source_tag = (
                            f"[{strategic_source}]" if strategic_source is not None else ""
                        )
                        action_explanations.append(
                            f"{act_name}({param_str}) {state_debug} {source_tag}".strip()
                        )
                        actions_short.append(
                            f"Step {step_idx+1:03d} | dist={dist_before:.1f}mm | {act_name}"
                        )

                        # ─── Execute action ───
                        sensor_after = interpreter.execute(atype, aparams)
                        current_poses.append(env.get_pose().copy())
                        pose_after = env.get_pose()

                        # Update controller tracking (important for next step strategic logic)
                        next_raw = controller._compute_state(pose_after, sensor_after)
                        sac_trainer._update_controller_tracking(
                            controller, next_raw, sensor_after, pose_after, atype
                        )

                        # Termination checks
                        dist_after = float(np.linalg.norm(goal_pose[:3] - pose_after[:3]))
                        if dist_after < sac_trainer.goal_threshold:
                            success = True
                            break

                        depth = sensor_after.get("depth", 100.0)
                        if depth < _COLLISION_DEPTH_THRESHOLD:
                            collision = True
                            collision_counts[atype] = collision_counts.get(atype, 0) + 1
                            break

                    ep_step_count = step_idx + 1
                    episode_steps_list.append(ep_step_count)
                    total += 1

                    if success:
                        successes += 1
                        success_steps_list.append(ep_step_count)
                        ep_result = "success"
                    elif collision:
                        collisions += 1
                        ep_result = "collision"
                    else:
                        timeouts += 1
                        ep_result = "timeout"

                    # Save episode visualization/logs (same as SAC train)
                    if visualizer is not None:
                        visualizer.save_episode(
                            env=env,
                            episode=global_ep_idx,
                            level=level_idx,
                            result=ep_result,
                            goal_pose=goal_pose,
                            poses=current_poses,
                            actions=action_explanations,
                            actions_short=actions_short,
                            extra_info={
                                "policy": "sac",
                                "eval_seed": int(eval_seed),
                            },
                        )

            count = max(total, 1)
            bounds = curriculum_levels[level_idx]
            total_act = max(sum(action_counts.values()), 1)

            collision_rate_per_type = {}
            for tid, cc in collision_counts.items():
                tc = action_counts.get(tid, 0)
                name = type_names.get(tid, f"type_{tid}")
                collision_rate_per_type[name] = {
                    "collisions": cc,
                    "total_calls": tc,
                    "rate": round(cc / max(tc, 1), 4),
                }

            results[f"level_{level_idx}"] = {
                "bounds_mm": list(bounds),
                "total_episodes": total,
                "success_count": successes,
                "timeout_count": timeouts,
                "collision_count": collisions,
                "success_rate": successes / count,
                "timeout_rate": timeouts / count,
                "collision_rate": collisions / count,
                "mean_episode_steps": round(
                    float(np.mean(episode_steps_list)) if episode_steps_list else 0, 1
                ),
                "mean_success_steps": round(
                    float(np.mean(success_steps_list)) if success_steps_list else 0, 1
                ),
                "total_steps_per_goal": round(total_steps / max(successes, 1), 1),
                "action_distribution": {
                    type_names.get(k, f"type_{k}"): {
                        "count": v,
                        "rate": round(v / total_act, 4),
                    }
                    for k, v in sorted(action_counts.items())
                },
                "collision_stats": {
                    type_names.get(k, f"type_{k}"): v for k, v in collision_counts.items()
                },
                "collision_rate_per_type": collision_rate_per_type,
            }

        return results

    # ══════════════════════════════════════════════════════
    # Adaptive (generic, environment-agnostic)
    # ══════════════════════════════════════════════════════

    def _run_adaptive_generic(
        self,
        online_env,
        offline_env,
        object_name: str,
        num_episodes: int,
        curriculum_levels: list,
        curriculum_filters: list,
        adapt_q_dir: str,
        adapt_sac_dir: str,
        adapt_seed: int,
        log_dir: Path,
        mesh_path: str,
        visualizer=None,
    ) -> dict:
        """Unified adaptive loop for any environment.

        Online interaction uses online_env (MuJoCo, robot, trimesh).
        Offline retrain uses offline_env (always trimesh — fast).
        State is frame-invariant: relative features in agent's local frame.

        Args:
            online_env: Environment for online interaction.
                Must implement RLEnvironment protocol.
            offline_env: Fast environment for offline retrain.
                Must have supports_offline_retrain=True.
            object_name: Name for logging and file paths.
            num_episodes: Total adaptive episodes.
            curriculum_levels: Distance ranges per level.
            curriculum_filters: Per-level filters.
            adapt_q_dir: Save path for adaptive Q-store.
            adapt_sac_dir: Save path for adaptive SAC.
            adapt_seed: Random seed.
            log_dir: Directory for logs/snapshots.
            mesh_path: Path to mesh STL (for offline retrain).
            visualizer: Optional EpisodeVisualizer instance.

        Returns:
            Final results dict.
        """
        from collections import Counter
        import pickle

        # ═══ Sleep trails collection ═══
        _MAX_SUCCESS_TRAILS_PER_LEVEL = 200
        _MAX_FAILURE_TRAILS_PER_LEVEL = 100
        _SLEEP_TRAILS_SAVE_EVERY = 200  # save periodically

        # ═══ Sleep-Q normalization data (NEW) ═══
        _SLEEP_Q_SAVE_EVERY = _SLEEP_TRAILS_SAVE_EVERY
        _SLEEP_Q_START_K = 5
        _SLEEP_Q_STRIDE = 5
        _SLEEP_Q_MAX_STATES_PER_LEVEL = 50_000
        _SLEEP_Q_MAX_START_PER_LEVEL = 10_000

        sleep_q_norm = {
            "mesh_name": object_name,
            "state_dim": int(self.rl_config.get("state_dim", 0)),
            "meta": {
                "start_k": _SLEEP_Q_START_K,
                "stride": _SLEEP_Q_STRIDE,
            },
            "by_level": {},  # lvl -> dict
        }

        def _ensure_q_level(lvl: int) -> dict:
            if lvl not in sleep_q_norm["by_level"]:
                sleep_q_norm["by_level"][lvl] = {
                    "start": {"states": [], "prox": [], "q_conf": []},
                    "sample": {"free": [], "surface": [], "bypass": []},
                    "counts": {"episodes": 0, "steps_seen": 0},
                }
            return sleep_q_norm["by_level"][lvl]

        def _cap_list(lst, cap: int) -> None:
            if len(lst) > cap:
                del lst[:-cap]

        sleep_success_trails: dict[int, list] = {}
        sleep_failure_trails: dict[int, list] = {}

        logger.info("=" * 60)
        logger.info(
            "Adaptive Generic: %s (%d episodes)",
            object_name, num_episodes,
        )
        logger.info("=" * 60)

        np.random.seed(adapt_seed)
        random.seed(adapt_seed)

        num_types = len(ExperienceExtractor.get_type_names())
        type_names = ExperienceExtractor.get_type_names()

        adaptive_cfg = {**self.rl_config, "mode": "adaptive"}
        adaptive_max_steps = self.rl_config.get(
            "max_steps_per_goal", 400
        )

        # ═══ Load controller ═══
        q_load_dir = (
            adapt_q_dir
            if (Path(adapt_q_dir) / "config.json").exists()
            else self._q_model_dir(adapt_seed)
        )
        logger.info("Loading Q-store from: %s", q_load_dir)

        controller = RLGoalApproachController.load(
            q_load_dir,
            agent_id=f"{object_name}_adaptive",
            config=adaptive_cfg,
        )
        controller._collision_stats = {}

        # ═══ Q-store diagnostics: initial sizes ═══
        q_store_initial_free = len(controller.q_store_free.points)
        q_store_initial_surface = len(controller.q_store_surface.points)
        q_store_initial_bypass = len(controller.q_store_surface_bypass.points)
        logger.info(
            "Q-store initial: free=%d, surface=%d",
            q_store_initial_free, q_store_initial_surface,
        )

        # ═══ Load SAC ═══
        sac_trainer = None
        sac_model_dir = self._sac_model_dir(self.sac_seed)
        if (Path(sac_model_dir) / "sac_actor.pt").exists():
            sac_trainer = PSACTrainer(
                state_dim=self.rl_config.get("state_dim", 20),
                num_types=num_types,
            )
            if (Path(adapt_sac_dir) / "sac_actor.pt").exists():
                sac_trainer.load(adapt_sac_dir)
                logger.info("SAC loaded from adaptive: %s", adapt_sac_dir)
            else:
                sac_trainer.load(sac_model_dir)
                logger.info("SAC loaded from training: %s", sac_model_dir)
            sac_trainer.buffer.set_current_mesh(object_name)

        # ═══ Manager (uses offline_env for retrain) ═══
        manager = AdaptiveTrainingManager(
            controller=controller,
            env=offline_env,
            config=adaptive_cfg,
            runs_dir=str(self.runs_dir),
            mesh_path=mesh_path,
            q_save_dir=adapt_q_dir,
            sac_save_dir=adapt_sac_dir,
            offline_check_window=self.config.get(
                "offline_check_window", 50
            ),
            promote_threshold=self.promote_threshold,
            promote_window=self.promote_window,
            online_sac_update_every=100,
            online_sac_update_steps=40,
            enable_online_sac_updates=False,
            enable_offline_sac_retrain=False,
        )
        manager.sac_trainer = sac_trainer

        # ═══ Sliding window RunningQStats ═══
        _ADAPTIVE_Q_STATS_WINDOW = 200_000
        manager.arbitrator._running_q_stats_free = RunningQStats(
            warmup=200, window=_ADAPTIVE_Q_STATS_WINDOW
        )
        manager.arbitrator._running_q_stats_surface = RunningQStats(
            warmup=200, window=_ADAPTIVE_Q_STATS_WINDOW
        )
        manager.arbitrator._warmup_running_stats()

        # ═══ Strategic SAC ═══
        if sac_trainer is not None:
            if sac_trainer.strategic_detach_sac is not None:
                manager.arbitrator._sac_strategic_detach = (
                    sac_trainer.strategic_detach_sac
                )
            if sac_trainer.strategic_direction_sac is not None:
                manager.arbitrator._sac_strategic_direction = (
                    sac_trainer.strategic_direction_sac
                )

        # ═══ Tracking state ═══
        episode_log: list[dict[str, Any]] = []
        snapshot_log: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {}
        collision_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {
            "q_store": 0, "sac": 0, "blend": 0, "heuristic": 0,
        }
        total_steps_adaptive = 0
        rolling_successes: list[bool] = []
        all_episode_steps: list[int] = []
        all_success_steps: list[int] = []
        level_episode_steps: dict[int, list[int]] = {}
        level_success_steps: dict[int, list[int]] = {}
        per_level_source_stats: dict[int, dict[str, dict]] = {}
        ep_successes: list[bool] = []

        # Curriculum state
        adaptive_level = 0
        adaptive_promote_window: list[bool] = []
        collision_stats_before: dict = dict(controller._collision_stats)

        log_dir.mkdir(parents=True, exist_ok=True)

        def _get_level_source(lvl, src):
            if lvl not in per_level_source_stats:
                per_level_source_stats[lvl] = {}
            if src not in per_level_source_stats[lvl]:
                per_level_source_stats[lvl][src] = {
                    "success": 0, "collision": 0, "timeout": 0,
                    "final_distances": [], "episode_steps": [],
                    "success_steps": [],
                }
            return per_level_source_stats[lvl][src]

        # ═══ Main loop ═══
        for episode in range(num_episodes):
            online_env.reset()
            start_pos = online_env.get_pose()[:3]

            # ═══ Goal generation with filters ═══
            min_dist, max_dist = curriculum_levels[adaptive_level]
            level_filter = (
                curriculum_filters[adaptive_level]
                if adaptive_level < len(curriculum_filters)
                else {}
            )
            require_same_side = level_filter.get("same_side", None)
            require_path_blocked = level_filter.get("path_blocked", None)
            max_goal_attempts = (
                50
                if (
                    require_same_side is not None
                    or require_path_blocked is not None
                )
                else 1
            )

            goal_pose = None
            is_fallback_episode = False

            handle_between = level_filter.get("handle_between", False)
            handle_info = None
            if handle_between:
                handle_info = _get_handle_info(online_env)
                if handle_info is None:
                    handle_between = False

            for _attempt in range(max_goal_attempts):
                online_env.reset()
                start_pos = online_env.get_pose()[:3]

                candidate = online_env.get_random_surface_point(
                    reference_pos=start_pos,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    max_attempts=2000,
                    mesh_sample=True,
                )

                if handle_between:
                    if not _check_handle_between(
                        online_env, start_pos,
                        candidate[:3], handle_info,
                    ):
                        continue
                    
                if require_same_side is not None or require_path_blocked is not None:
                    online_env.set_goal(candidate)
                    sensor_check = online_env.get_sensor_data()

                    if (
                        require_same_side is not None
                        and sensor_check.get("same_side", True)
                        != require_same_side
                    ):
                        continue
                    if (
                        require_path_blocked is not None
                        and sensor_check.get("path_blocked", False)
                        != require_path_blocked
                    ):
                        continue

                goal_pose = candidate
                break

            if goal_pose is None:
                logger.warning(
                    "Adaptive ep %d: L%d filter not satisfied, "
                    "using unfiltered goal",
                    episode, adaptive_level,
                )
                online_env.reset()
                start_pos = online_env.get_pose()[:3]
                goal_pose = online_env.get_random_surface_point(
                    reference_pos=start_pos,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    max_attempts=2000,
                    mesh_sample=True,
                )
                is_fallback_episode = True

            controller.set_new_goal(goal_pose, start_pos)
            online_env.set_goal(goal_pose)
            manager.arbitrator.start_episode(level=adaptive_level)

            goals_before = controller._total_goals_reached
            ep_steps = 0
            ep_sources: list[str] = []
            current_poses: list[np.ndarray] = [online_env.get_pose().copy()]
            action_explanations: list[str] = []
            last_transitions: list = []
            collision_stats_before = dict(controller._collision_stats)
            action_short_labels: list[str] = []
            ep_q_confidences: list[float] = []
            ep_q_reliable_steps = 0
            ep_q_total_steps = 0

            # ═══ Step loop ═══
            for step in range(adaptive_max_steps):
                pose = online_env.get_pose()
                sensor = online_env.get_sensor_data()
                state = controller._compute_state(pose, sensor)

                # Arbitrator decides action
                action_type, action_params, source, q_values, confidence_info = (
                    manager.get_action(state, pose, sensor)
                )
                controller._current_source = source

                # ═══ Collect sleep-Q normalization stats (NEW) ═══
                lvl_bucket = _ensure_q_level(adaptive_level)
                lvl_bucket["counts"]["steps_seen"] += 1

                # Start-of-episode states (first K steps)
                if step < _SLEEP_Q_START_K:
                    lvl_bucket["start"]["states"].append(
                        state.astype(np.float32, copy=False)
                    )
                    if confidence_info is not None:
                        lvl_bucket["start"]["prox"].append(
                            float(confidence_info.get("proximity", 0.0))
                        )
                        lvl_bucket["start"]["q_conf"].append(
                            float(confidence_info.get("overall", 0.0))
                        )
                    else:
                        lvl_bucket["start"]["prox"].append(0.0)
                        lvl_bucket["start"]["q_conf"].append(0.0)

                    _cap_list(lvl_bucket["start"]["states"], _SLEEP_Q_MAX_START_PER_LEVEL)
                    _cap_list(lvl_bucket["start"]["prox"], _SLEEP_Q_MAX_START_PER_LEVEL)
                    _cap_list(lvl_bucket["start"]["q_conf"], _SLEEP_Q_MAX_START_PER_LEVEL)

                # Sparse samples across episode for global normalization coverage
                if (step % _SLEEP_Q_STRIDE) == 0:
                    on_object = float(state[11]) > 0.5

                    # same_side stored in state (ssS) at index 27 (state_dim=28)
                    same_side_state = True
                    if len(state) > 27:
                        same_side_state = bool(float(state[27]) > 0.5)

                    if not on_object:
                        lvl_bucket["sample"]["free"].append(
                            state.astype(np.float32, copy=False)
                        )
                        _cap_list(lvl_bucket["sample"]["free"], _SLEEP_Q_MAX_STATES_PER_LEVEL)
                    else:
                        if same_side_state:
                            lvl_bucket["sample"]["surface"].append(
                                state.astype(np.float32, copy=False)
                            )
                            _cap_list(lvl_bucket["sample"]["surface"], _SLEEP_Q_MAX_STATES_PER_LEVEL)
                        else:
                            lvl_bucket["sample"]["bypass"].append(
                                state.astype(np.float32, copy=False)
                            )
                            _cap_list(lvl_bucket["sample"]["bypass"], _SLEEP_Q_MAX_STATES_PER_LEVEL)

                # ═══ Per-episode Q diagnostics ═══
                # Q confidence уже посчитан внутри arbitrator._get_q_action
                # Берём из последних значений arbitrator
                if manager.arbitrator.q_confidence_history:
                    ep_q_confidences.append(manager.arbitrator.q_confidence_history[-1])
                ep_q_total_steps += 1

                # Discrete index for Q-store learning
                discrete_idx = sac_to_discrete(action_type, action_params)

                # ═══ State debug BEFORE update_only (goal still set) ═══
                state_debug = controller.get_state_debug_info(
                    state, pose, sensor, q_values=q_values, q_confidence=confidence_info
                )

                # Save transitions before update_only clears them
                last_transitions = controller._episode_transitions.copy()

                # ═══ KEY: use continuous params when available ═══
                if online_env.supports_continuous:
                    sensor_after = online_env.step_continuous(
                        action_type, action_params
                    )
                else:
                    sensor_after = online_env.step_discrete(
                        discrete_idx, controller.action_space
                    )

                ep_steps += 1
                total_steps_adaptive += 1
                current_poses.append(online_env.get_pose().copy())

                # Track action type
                act_name = type_names.get(
                    action_type, f"type_{action_type}"
                )
                action_counts[act_name] = (
                    action_counts.get(act_name, 0) + 1
                )

                # Track source
                # Track source — explicit mapping by reason name
                SOURCE_MAP = {
                    "forced_heuristic": "heuristic",
                    "calibration_sac": "sac",
                    "calibration_heuristic": "heuristic",
                    "q_NOT_reliable_sac_proven": "sac",
                    "q_NOT_reliable_heuristic_default": "heuristic",
                    "blend_crawl": "blend",
                    "q_crawl_sac_other": "q_store",
                    "q_crawl_disagree": "q_store",
                    "sac_non_crawl": "sac",
                    "q_non_crawl": "q_store",
                    "blend_agree": "blend",
                    "sac_moderate_agree": "sac",
                    "sac_proven": "sac",
                    "sac_proven_disagree": "sac",
                    "q_reliable_disagree": "q_store",
                    "q_agree": "q_store",
                    "q_moderate": "q_store",
                    "heuristic_default": "heuristic",
                    "heuristic_disagree": "heuristic",
                    "heuristic_fallback": "heuristic",
                    "q_explore": "q_store",
                }
                
                # Extract reason name (before first parenthesis)
                reason_name = source.split("(")[0] if "(" in source else source
                source_key = SOURCE_MAP.get(reason_name, "unknown")
                
                if reason_name not in SOURCE_MAP:
                    logger.warning(
                        "Unknown source reason: %s, defaulting to unknown",
                        reason_name,
                    )

                source_counts[source_key] = (
                    source_counts.get(source_key, 0) + 1
                )
                ep_sources.append(source_key)

                # Q-store learns from discrete action
                pose_after = online_env.get_pose()
                _, done, td_debug = controller.update_only(
                    pose_after, sensor_after, discrete_idx, source_key
                )

                # ═══ Action explanations ═══
                dist_to_goal = float(
                    np.linalg.norm(goal_pose[:3] - pose[:3])
                )

                # Full debug (actions.txt)
                params_str = self._format_action_params(
                    action_type, action_params
                )
                act_full = (
                    f"{act_name}-{discrete_idx} ({params_str})" if params_str
                    else act_name
                )

                # Post-step sensor info
                new_sensor = online_env.get_sensor_data()
                env_pos = online_env.get_pose()[:3]

                action_full = (
                    f"{act_full} "
                    f"| {state_debug} "
                    f"| {source} "
                    f"|mj=[{env_pos[0]:.4f},"
                    f"{env_pos[1]:.4f},"
                    f"{env_pos[2]:.4f}]"
                    f" |depth={new_sensor.get('depth', 0):.1f}"
                    f" |on={int(new_sensor.get('on_object', False))}"
                    f" |td_debug={td_debug} "
                )
                action_explanations.append(action_full)

                # Short label (картинка/видео)
                source_label = source_key.upper().replace("_", " ")
                action_short_labels.append(
                    f"Step {step + 1:03d} | dist={dist_to_goal:.1f}mm"
                    f" | {act_name} | {source_label}"
                )

                if done:
                    break

            # ═══ Episode result ═══
            success = (
                controller._total_goals_reached > goals_before
            )
            ep_successes.append(success)
            
            # Count episodes for sleep-Q stats (NEW)
            _ensure_q_level(adaptive_level)["counts"]["episodes"] += 1

            # ═══ Collect sleep trails ═══
            if last_transitions:
                if success:
                    sleep_success_trails.setdefault(adaptive_level, [])
                    # sleep_success_trails[adaptive_level].append(last_transitions)
                    sleep_success_trails[adaptive_level].extend(
                        [t for t in last_transitions if t.get("source") == "heuristic"]
                    )
                    if len(sleep_success_trails[adaptive_level]) > _MAX_SUCCESS_TRAILS_PER_LEVEL:
                        sleep_success_trails[adaptive_level] = (
                            sleep_success_trails[adaptive_level][-_MAX_SUCCESS_TRAILS_PER_LEVEL:]
                        )
                else:
                    # Failures for critic: sample ~30%
                    if np.random.random() < 0.3:
                        sleep_failure_trails.setdefault(adaptive_level, [])
                        sleep_failure_trails[adaptive_level].append(last_transitions)
                        #sleep_failure_trails[adaptive_level].extend(
                        #    [t for t in last_transitions if t.get("source") == "heuristic"]
                        #)
                        if len(sleep_failure_trails[adaptive_level]) > _MAX_FAILURE_TRAILS_PER_LEVEL:
                            sleep_failure_trails[adaptive_level] = (
                                sleep_failure_trails[adaptive_level][-_MAX_FAILURE_TRAILS_PER_LEVEL:]
                            )

            all_episode_steps.append(ep_steps)
            if success:
                all_success_steps.append(ep_steps)

            if adaptive_level not in level_episode_steps:
                level_episode_steps[adaptive_level] = []
                level_success_steps[adaptive_level] = []
            level_episode_steps[adaptive_level].append(ep_steps)
            if success:
                level_success_steps[adaptive_level].append(ep_steps)

            # Termination type
            if success:
                termination = "success"
            elif step == (adaptive_max_steps - 1):
                termination = "timeout"
            else:
                termination = "collision"

            # Collision tracking
            if termination == "collision":
                for act_name_c, count in (
                    controller._collision_stats.items()
                ):
                    prev = collision_stats_before.get(act_name_c, 0)
                    if count > prev:
                        collision_counts[act_name_c] = (
                            collision_counts.get(act_name_c, 0)
                            + (count - prev)
                        )

            # Per-source per-level tracking
            source_counter = Counter(ep_sources)
            dominant_source_key = (
                source_counter.most_common(1)[0][0]
                if ep_sources
                else "heuristic"
            )

            src_stats = _get_level_source(
                adaptive_level, dominant_source_key
            )
            src_stats[termination] = (
                src_stats.get(termination, 0) + 1
            )
            final_pose = online_env.get_pose()
            final_dist = float(
                np.linalg.norm(goal_pose[:3] - final_pose[:3])
            )
            src_stats["final_distances"].append(round(final_dist, 1))
            src_stats["episode_steps"].append(ep_steps)
            if success:
                src_stats["success_steps"].append(ep_steps)

            # Manager update (offline retrain uses offline_env)
            manager.on_episode_complete(
                success=success, transitions=last_transitions,
            )
            if manager._offline_just_completed:
                rolling_successes.clear()
                manager._offline_just_completed = False

            # Mode change logging
            while manager.mode_changes:
                change = manager.mode_changes.pop(0)
                change["recent_episodes"] = episode_log[-10:]
                change_idx = len([
                    f for f in log_dir.iterdir()
                    if f.name.startswith("mode_change_")
                ])
                change_path = (
                    log_dir
                    / f"mode_change_{change_idx:04d}"
                    f"_ep_{change['episode']:05d}.json"
                )
                with change_path.open("w") as f:
                    json.dump(change, f, indent=2)

            manager.arbitrator.on_episode_end(success)

            # Curriculum promote
            if not is_fallback_episode:
                adaptive_promote_window.append(success)
            if len(adaptive_promote_window) > self.promote_window:
                adaptive_promote_window.pop(0)
            if (
                len(adaptive_promote_window) == self.promote_window
                and adaptive_level < len(curriculum_levels) - 1
            ):
                promote_rate = (
                    sum(adaptive_promote_window)
                    / self.promote_window
                )
                if promote_rate >= self.promote_threshold:
                    adaptive_level += 1
                    adaptive_promote_window = []
                    logger.info(
                        "Promoted to level %d at ep %d (rate=%.3f)",
                        adaptive_level, episode + 1, promote_rate,
                    )

            # Rolling success rate
            rolling_successes.append(success)
            if len(rolling_successes) > 100:
                rolling_successes.pop(0)
            rolling_rate = (
                sum(rolling_successes) / len(rolling_successes)
            )
            total_rate = sum(ep_successes) / len(ep_successes)

            # Episode log entry
            start_dist = float(
                np.linalg.norm(goal_pose[:3] - start_pos)
            )
            episode_log.append({
                "episode": episode,
                "success": success,
                "termination": termination,
                "steps": ep_steps,
                "start_distance": round(start_dist, 1),
                "final_distance": round(final_dist, 1),
                "dominant_source": dominant_source_key,
                "last_source_detail": source,
                "rolling_success_rate": round(rolling_rate, 3),
                "mode": manager.mode,
                "curriculum_level": adaptive_level,
                "ep_q_confidence_mean": round(
                    float(np.mean(ep_q_confidences)) if ep_q_confidences else 0.0, 4
                ),
                "ep_q_confidence_max": round(
                    float(np.max(ep_q_confidences)) if ep_q_confidences else 0.0, 4
                ),
            })

            # Visualization
            if visualizer is not None:
                visualizer.save_episode(
                    env=online_env,
                    episode=episode,
                    level=adaptive_level,
                    result=termination,
                    goal_pose=goal_pose,
                    poses=current_poses,
                    actions=action_explanations,
                    actions_short=action_short_labels,
                    extra_info={
                        "object": object_name,
                        "mode": manager.mode,
                        "source": dominant_source_key,
                        "rolling_rate": round(rolling_rate, 3),
                    },
                )

            # ═══ Periodic snapshot ═══
            if (episode + 1) % _ADAPTIVE_LOG_INTERVAL == 0:
                stats = manager.get_stats()
                arb_stats = manager.arbitrator.get_stats()
                total_src = max(sum(source_counts.values()), 1)
                total_act = max(sum(action_counts.values()), 1)
                successes_count = sum(
                    1 for e in episode_log if e["success"]
                )
                # ═══ Q-store diagnostics for snapshot ═══
                q_store_diagnostics = {
                    "q_store_free_size": len(controller.q_store_free.points),
                    "q_store_surface_size": len(controller.q_store_surface.points),
                    "q_store_bypass_size": len(controller.q_store_surface_bypass.points),
                    "q_store_free_growth": len(controller.q_store_free.points) - q_store_initial_free,
                    "q_store_surface_growth": len(controller.q_store_surface.points) - q_store_initial_surface,
                    "q_store_bypass_growth": len(controller.q_store_surface_bypass.points) - q_store_initial_bypass,
                    "store_stats": {
                        "free": controller.q_store_free.get_stats(),
                        "surface": controller.q_store_surface.get_stats(),
                        "bypass": controller.q_store_surface_bypass.get_stats(),
                    },
                    # Per-level from arbitrator
                    "per_level": arb_stats.get("q_diagnostics_per_level", {}),
                }
                snapshot = self._build_adaptive_snapshot(
                    episode=episode + 1,
                    rolling_rate=rolling_rate,
                    total_rate=total_rate,
                    adaptive_level=adaptive_level,
                    stats=stats,
                    arb_stats=arb_stats,
                    total_steps_adaptive=total_steps_adaptive,
                    episode_log=episode_log,
                    source_counts=source_counts,
                    action_counts=action_counts,
                    collision_counts=collision_counts,
                    all_episode_steps=all_episode_steps,
                    all_success_steps=all_success_steps,
                    per_level_source_stats=per_level_source_stats,
                )
                snapshot["q_store_diagnostics"] = q_store_diagnostics
                snapshot_log.append(snapshot)

                snap_path = (
                    log_dir / f"snapshot_ep_{episode + 1:05d}.json"
                )
                with snap_path.open("w") as f:
                    json.dump(snapshot, f, indent=2)

                # ═══ Periodic sleep trails save ═══
                if (episode + 1) % _SLEEP_TRAILS_SAVE_EVERY == 0:
                    self._save_sleep_trails(
                        log_dir, object_name, episode + 1,
                        sleep_success_trails, sleep_failure_trails,
                        ep_successes, adaptive_level,
                        level_episode_steps, level_success_steps,
                        sleep_q_norm=sleep_q_norm,
                    )
                logger.info(
                    "Adaptive ep %d: rate=%.3f, mode=%s",
                    episode + 1, rolling_rate, stats["mode"],
                )

        # ═══ Final sleep trails save ═══
        self._save_sleep_trails(
            log_dir, object_name, len(episode_log),
            sleep_success_trails, sleep_failure_trails,
            ep_successes, adaptive_level,
            level_episode_steps, level_success_steps,
            sleep_q_norm=sleep_q_norm,
        )
        # ═══ Final Q-store diagnostics ═══
        q_final_free = len(controller.q_store_free.points)
        q_final_surface = len(controller.q_store_surface.points)
        q_final_bypass = len(controller.q_store_surface_bypass.points)
        logger.info(
            "Q-store final: free=%d (+%d), surface=%d (+%d), bypass=%d (+%d)",
            q_final_free, q_final_free - q_store_initial_free,
            q_final_surface, q_final_surface - q_store_initial_surface,
            q_final_bypass, q_final_bypass - q_store_initial_bypass,
        )
        # ═══ Save models ═══
        controller.save(adapt_q_dir)
        logger.info("Adaptive Q-store saved to %s", adapt_q_dir)

        if manager.sac_trainer:
            manager.sac_trainer.save(adapt_sac_dir)
            logger.info("Adaptive SAC saved to %s", adapt_sac_dir)

        # ═══ Build final results ═══
        total_ep = len(episode_log)
        total_src = max(sum(source_counts.values()), 1)
        total_act = max(sum(action_counts.values()), 1)
        successes_total = sum(
            1 for e in episode_log if e["success"]
        )
        collisions_total = sum(
            1 for e in episode_log
            if e["termination"] == "collision"
        )
        timeouts_total = sum(
            1 for e in episode_log
            if e["termination"] == "timeout"
        )

        final_results = {
            "object": object_name,
            "total_episodes": total_ep,
            "total_steps": total_steps_adaptive,
            "success_rate": round(
                successes_total / max(total_ep, 1), 4
            ),
            "collision_rate": round(
                collisions_total / max(total_ep, 1), 4
            ),
            "timeout_rate": round(
                timeouts_total / max(total_ep, 1), 4
            ),
            "mean_episode_steps": round(
                float(np.mean(all_episode_steps))
                if all_episode_steps else 0, 1,
            ),
            "mean_success_steps": round(
                float(np.mean(all_success_steps))
                if all_success_steps else 0, 1,
            ),
            "total_steps_per_goal": round(
                total_steps_adaptive / max(successes_total, 1), 1,
            ),
            "final_level": adaptive_level,
            "per_level_steps": {
                f"level_{lvl}": {
                    "mean_episode_steps": round(
                        float(np.mean(steps_list))
                        if steps_list else 0, 1,
                    ),
                    "mean_success_steps": round(
                        float(np.mean(
                            level_success_steps.get(lvl, [])
                        ))
                        if level_success_steps.get(lvl) else 0, 1,
                    ),
                    "episodes": len(steps_list),
                    "successes": len(
                        level_success_steps.get(lvl, [])
                    ),
                }
                for lvl, steps_list
                in sorted(level_episode_steps.items())
            },
            "source_distribution": {
                k: round(v / total_src, 4)
                for k, v in source_counts.items()
            },
            "action_distribution": {
                k: {"count": v, "rate": round(v / total_act, 4)}
                for k, v in sorted(
                    action_counts.items(), key=lambda x: -x[1]
                )
            },
            "collision_stats": dict(collision_counts),
            "snapshots": snapshot_log,
            "episode_log": episode_log,
            "per_source_analysis": self._build_per_source_analysis(
                per_level_source_stats
            ),
            "q_store_final": {
                "free_size": q_final_free,
                "surface_size": q_final_surface,
                "bypass_size": q_final_bypass,
                "free_growth": q_final_free - q_store_initial_free,
                "surface_growth": q_final_surface - q_store_initial_surface,
                "bypass_growth": q_final_bypass - q_store_initial_bypass,
            },
        }

        # Save results
        result_path = (
            self.data_dir / f"adaptive_result_{object_name}.json"
        )
        with result_path.open("w") as f:
            json.dump(final_results, f, indent=2)

        self._save_meta(
            f"adaptive_{object_name}",
            adapt_seed,
            {
                "success_rate": final_results["success_rate"],
                "total_episodes": total_ep,
                "source_distribution": final_results[
                    "source_distribution"
                ],
            },
            [object_name],
        )

        logger.info(
            "Adaptive %s complete: success=%.3f, "
            "collision=%.3f, timeout=%.3f",
            object_name,
            final_results["success_rate"],
            final_results["collision_rate"],
            final_results["timeout_rate"],
        )

        return final_results

    # ══════════════════════════════════════════════════════
    # Adaptive helpers (extracted from inline code)
    # ══════════════════════════════════════════════════════
    @staticmethod
    def _save_sleep_trails(
        log_dir: Path,
        mesh_name: str,
        episode: int,
        success_trails: dict[int, list],
        failure_trails: dict[int, list],
        ep_successes: list[bool],
        current_level: int,
        level_episode_steps: dict[int, list],
        level_success_steps: dict[int, list],
        sleep_q_norm: dict | None = None,
    ) -> None:
        """Save sleep trails to disk for offline SAC consolidation."""
        # Per-level success rates
        success_rate_by_level = {}
        trails_count_by_level = {}
        
        for lvl in sorted(set(list(success_trails.keys()) + list(failure_trails.keys()))):
            s_count = len(success_trails.get(lvl, []))
            f_count = len(failure_trails.get(lvl, []))
            total = s_count + f_count
            success_rate_by_level[lvl] = round(s_count / max(total, 1), 3)
            trails_count_by_level[lvl] = {"success": s_count, "failure": f_count}

        sleep_data = {
            "mesh_name": mesh_name,
            "episode": episode,
            "current_level": current_level,
            "success_trails_by_level": success_trails,
            "failure_trails_by_level": failure_trails,
            "stats": {
                "total_episodes": episode,
                "overall_success_rate": round(
                    sum(ep_successes) / max(len(ep_successes), 1), 3
                ),
                "success_rate_by_level": success_rate_by_level,
                "trails_count_by_level": trails_count_by_level,
            },
        }

        path = log_dir / "sleep_trails.pkl"
        with path.open("wb") as f:
            pickle.dump(sleep_data, f)

        logger.info(
            "Sleep trails saved: ep=%d, levels=%s, path=%s",
            episode, trails_count_by_level, path,
        )

        # ═══ Save sleep-Q normalization data (NEW) ═══
        if sleep_q_norm is not None:
            import pickle as _pickle
            q_path = log_dir / "sleep_q_norm.pkl"
            with q_path.open("wb") as f:
                _pickle.dump(sleep_q_norm, f)
            logger.info(
                "Sleep Q-norm data saved: ep=%d, path=%s",
                episode, q_path,
            )

    @staticmethod
    def _format_action_params(action_type: int, action_params: np.ndarray) -> str:
        """Human-readable action parameters for logging."""
        if action_type == 0:
            # MoveTangentially: [sin, cos, distance]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            angle = float(np.degrees(np.arctan2(sin_a, cos_a))) % 360
            dist = float(np.clip(action_params[2], 0.5, 15.0))
            return f"angle={angle:.0f}°,step={dist:.1f}mm"
        elif action_type == 1:
            # MoveLinear: [distance]
            dist = float(np.clip(action_params[0], -25.0, 25.0))
            return f"step={dist:.1f}mm"
        elif action_type == 2:
            # Turn (yaw): [sin, cos]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            angle = float(np.degrees(np.arctan2(sin_a, cos_a)))
            angle = float(np.clip(angle, -45.0, 45.0))
            return f"yaw={angle:.1f}°"
        elif action_type == 3:
            # Look (pitch): [sin, cos]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            angle = float(np.degrees(np.arctan2(sin_a, cos_a)))
            angle = float(np.clip(angle, -45.0, 45.0))
            return f"pitch={angle:.1f}°"
        elif action_type == 4:
            # Rotate (roll): [sin, cos]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            angle = float(np.degrees(np.arctan2(sin_a, cos_a)))
            angle = float(np.clip(angle, -45.0, 45.0))
            return f"roll={angle:.1f}°"
        elif action_type == 5:
            # OrientHorizontal: [rotation, left_dist, fwd_dist]
            return (
                f"rot={action_params[0]:.1f}°,"
                f"left={action_params[1]:.1f}mm,"
                f"fwd={action_params[2]:.1f}mm"
            )
        elif action_type == 6:
            # OrientVertical: [rotation, down_dist, fwd_dist]
            return (
                f"rot={action_params[0]:.1f}°,"
                f"down={action_params[1]:.1f}mm,"
                f"fwd={action_params[2]:.1f}mm"
            )
        elif action_type == 7:
            return ""
        return ""

    @staticmethod
    def _build_per_source_analysis(
        per_level_source_stats: dict,
    ) -> dict:
        """Build per-source analysis dict for results JSON."""
        result = {}
        for lvl, level_data in sorted(
            per_level_source_stats.items()
        ):
            level_key = f"level_{lvl}"
            result[level_key] = {}
            for src_name, src_data in level_data.items():
                total = (
                    src_data.get("success", 0)
                    + src_data.get("collision", 0)
                    + src_data.get("timeout", 0)
                )
                result[level_key][src_name] = {
                    "terminations": {
                        k: v
                        for k, v in src_data.items()
                        if k in ("success", "collision", "timeout")
                    },
                    "total_episodes": total,
                    "success_rate": round(
                        src_data.get("success", 0)
                        / max(total, 1), 3,
                    ),
                    "mean_final_distance": round(
                        float(np.mean(src_data["final_distances"]))
                        if src_data["final_distances"] else 0, 1,
                    ),
                    "near_miss_count": sum(
                        1 for d in src_data["final_distances"]
                        if 2.0 < d <= 5.0
                    ),
                    "mean_success_steps": round(
                        float(np.mean(
                            src_data.get("success_steps", [])
                        ))
                        if src_data.get("success_steps") else 0, 1,
                    ),
                }
        return result

    @staticmethod
    def _build_adaptive_snapshot(
        episode: int,
        rolling_rate: float,
        total_rate: float,
        adaptive_level: int,
        stats: dict,
        arb_stats: dict,
        total_steps_adaptive: int,
        episode_log: list,
        source_counts: dict,
        action_counts: dict,
        collision_counts: dict,
        all_episode_steps: list,
        all_success_steps: list,
        per_level_source_stats: dict,
    ) -> dict:
        """Build periodic snapshot dict."""
        total_src = max(sum(source_counts.values()), 1)
        total_act = max(sum(action_counts.values()), 1)
        successes_count = sum(
            1 for e in episode_log if e["success"]
        )

        return {
            "episode": episode,
            "rolling_success_rate": round(rolling_rate, 3),
            "total_episodes_success_rate": round(total_rate, 3),
            "curriculum_level": adaptive_level,
            "mode": stats["mode"],
            "total_episodes": episode,
            "total_steps": total_steps_adaptive,
            "success_count": successes_count,
            "collision_count": sum(
                1 for e in episode_log
                if e["termination"] == "collision"
            ),
            "timeout_count": sum(
                1 for e in episode_log
                if e["termination"] == "timeout"
            ),
            "mean_episode_steps": round(
                float(np.mean(all_episode_steps))
                if all_episode_steps else 0, 1,
            ),
            "mean_success_steps": round(
                float(np.mean(all_success_steps))
                if all_success_steps else 0, 1,
            ),
            "total_steps_per_goal": round(
                total_steps_adaptive
                / max(successes_count, 1), 1,
            ),
            "source_distribution": {
                k: round(v / total_src, 3)
                for k, v in source_counts.items()
            },
            "action_distribution": {
                k: {"count": v, "rate": round(v / total_act, 4)}
                for k, v in sorted(
                    action_counts.items(), key=lambda x: -x[1]
                )
            },
            "collision_stats": dict(collision_counts),
            "arbitrator": {
                "q_store_rate": arb_stats.get("q_store_rate", 0),
                "sac_rate": arb_stats.get("sac_rate", 0),
                "blend_rate": arb_stats.get("blend_rate", 0),
                "heuristic_rate": arb_stats.get("heuristic_rate", 0),
                "agreement_rate": arb_stats.get("agreement_rate", 0),
                "q_spread_mean": arb_stats.get("q_spread_mean", 0),
                "q_confidence_mean": arb_stats.get(
                    "q_confidence_mean", 0
                ),
                "q_success_rate": arb_stats.get("q_success_rate", 0),
                "sac_success_rate": arb_stats.get(
                    "sac_success_rate", 0
                ),
                "heuristic_success_rate": arb_stats.get(
                    "heuristic_success_rate", 0
                ),
                "blend_success_rate": arb_stats.get(
                    "blend_success_rate", 0
                ),
                "arbitrage_only": arb_stats.get(
                    "arbitrage_only", {}
                ),
                "per_level_track_record": arb_stats.get(
                    "per_level_track_record", {}
                ),
                "is_calibrating": arb_stats.get(
                    "is_calibrating", False
                ),
                "agreement_analysis": arb_stats.get(
                    "agreement_analysis", {}
                ),
                "disagreement_summary": arb_stats.get(
                    "disagreement_summary", {}
                ),
                "recent_disagreements": arb_stats.get(
                    "recent_disagreements", {}
                ),
            },
            "manager": {
                "sac_updates": stats.get("total_sac_updates", 0),
                "offline_iterations": stats.get(
                    "total_offline_iterations", 0
                ),
            },
            "per_source_analysis": (
                RLGoalApproachExperiment._build_per_source_analysis(
                    per_level_source_stats
                )
            ),
        }

    # ══════════════════════════════════════════════════════
    # Adaptive wrappers (delegate to _run_adaptive_generic)
    # ══════════════════════════════════════════════════════

    def _run_adaptive(self) -> None:
        """Run adaptive mode on trimesh (LightweightEnv).

        Both online and offline use the same trimesh env
        since it's fast enough for both.
        """
        logger.info("=" * 60)
        logger.info("Adaptive (trimesh): %s", self.adaptive_mesh)
        logger.info("=" * 60)

        mesh_path = str(
            self.data_dir / f"{self.adaptive_mesh}.stl"
        )
        adapt_seed = self.train_seeds[0]

        np.random.seed(self.adapt_seed)
        random.seed(self.adapt_seed)

        env = LightweightEnv(mesh_path, seed=adapt_seed, mesh_name=self.adaptive_mesh)

        adapt_q_dir = str(
            self.runs_dir
            / f"adaptive_q_seed_{adapt_seed}"
        )
        adapt_sac_dir = str(
            self.runs_dir
            / f"adaptive_sac_seed_{self.sac_seed}"
        )
        log_dir = (
            self.data_dir
            / f"adaptive_logs_{self.adaptive_mesh}"
        )

        # Visualizer
        visualizer = None
        if self.visualise:
            visualizer = EpisodeVisualizer(
                output_dir=self.data_dir,
                mesh_name=self.adaptive_mesh,
                stage="adaptive",
                max_per_type_per_level=5,
                num_levels=len(self.curriculum_levels),
                visualize_mode=self.visualise,
            )

        self._run_adaptive_generic(
            online_env=env,
            offline_env=env,  # trimesh is fast — same env for both
            object_name=self.adaptive_mesh,
            num_episodes=self.adaptive_episodes,
            curriculum_levels=self.curriculum_levels,
            curriculum_filters=self.curriculum_filters,
            adapt_q_dir=adapt_q_dir,
            adapt_sac_dir=adapt_sac_dir,
            adapt_seed=adapt_seed,
            log_dir=log_dir,
            mesh_path=mesh_path,
            visualizer=visualizer,
        )

    # ══════════════════════════════════════════════════════
    # SLEEP
    # ══════════════════════════════════════════════════════
    def _run_sleep(self) -> None:
        """Sleep consolidation: offline SAC fine-tune from adaptive trails.

        Reads sleep_trails.pkl from last adaptive run,
        checks conditions, runs SGD-only SAC consolidation.
        """
        import pickle

        sleep_cfg = self.config.get("sleep_config", {})
        success_threshold = sleep_cfg.get("success_threshold", 0.80)
        min_success_trails = sleep_cfg.get("min_success_trails", 20)

        def _cap_list(lst, cap: int):
            if len(lst) > cap:
                return lst[-cap:]
            return lst

        def _to_np(states_list):
            if not states_list:
                return None
            return np.asarray(states_list, dtype=np.float32)

        def _compute_mean_std(states_arr: np.ndarray, min_std: float = 1e-4):
            mean = states_arr.mean(axis=0)
            std = states_arr.std(axis=0)
            std = np.maximum(std, min_std)
            return mean, std

        def _sample_mix(current_list, replay_list, total: int, current_ratio: float):
            """Return list of states (python list), sampled from current/replay."""
            current_list = list(current_list) if current_list else []
            replay_list = list(replay_list) if replay_list else []
            if total <= 0:
                return []

            n_cur = int(total * current_ratio)
            n_rep = total - n_cur

            if len(current_list) <= n_cur:
                cur = current_list
            else:
                idx = np.random.choice(len(current_list), n_cur, replace=False)
                cur = [current_list[i] for i in idx]

            if len(replay_list) <= n_rep:
                rep = replay_list
            else:
                idx = np.random.choice(len(replay_list), n_rep, replace=False)
                rep = [replay_list[i] for i in idx]

            return cur + rep

        mesh_name = self.adaptive_mesh
        log_dir = self.data_dir / f"adaptive_logs_{mesh_name}"
        trails_path = log_dir / "sleep_trails.pkl"

        # ═══ Load sleep-Q norm data ═══
        q_norm_path = log_dir / "sleep_q_norm.pkl"
        sleep_q_norm = None
        if q_norm_path.exists():
            with q_norm_path.open("rb") as f:
                sleep_q_norm = pickle.load(f)
            logger.info("Sleep Q-norm: loaded %s", q_norm_path)
            for lvl, d in sleep_q_norm["by_level"].items():
                logger.info(
                    "sleep_q_norm L%d: start=%d sample_free=%d sample_surface=%d sample_bypass=%d episodes=%d",
                    lvl,
                    len(d["start"]["states"]),
                    len(d["sample"]["free"]),
                    len(d["sample"]["surface"]),
                    len(d["sample"]["bypass"]),
                    d["counts"]["episodes"],
                )
        else:
            logger.info("Sleep Q-norm: no file at %s (skip Q-norm)", q_norm_path)

        # ═══════════════════════════════════════
        # Sleep Q-norm decision (NEW)
        # ═══════════════════════════════════════
        sleep_cfg = self.config.get("sleep_config", {})
        qnorm_cfg = sleep_cfg.get("sleep_q_norm", {})
        qnorm_enabled = bool(qnorm_cfg.get("enabled", True))

        do_qnorm = False
        qnorm_levels = []

        if qnorm_enabled and sleep_q_norm is not None:
            min_start_states = int(qnorm_cfg.get("min_start_states", 500))
            prox_thr = float(qnorm_cfg.get("prox_median_threshold", 0.15))

            by_level = sleep_q_norm.get("by_level", {})
            for lvl, data in by_level.items():
                start_prox = data.get("start", {}).get("prox", [])
                start_states = data.get("start", {}).get("states", [])
                if len(start_states) < min_start_states:
                    continue
                if start_prox:
                    med_prox = float(np.median(start_prox))
                else:
                    med_prox = 0.0

                if med_prox < prox_thr:
                    qnorm_levels.append(int(lvl))

            if qnorm_levels:
                do_qnorm = True
                logger.info(
                    "Sleep Q-norm: will run for levels=%s (low start prox)",
                    qnorm_levels,
                )
            else:
                logger.info("Sleep Q-norm: skip (start prox OK / not enough data)")
        else:
            logger.info("Sleep Q-norm: disabled or no data")

        # ═══════════════════════════════════════
        # Sleep Q-norm consolidation (NEW)
        # ═══════════════════════════════════════
        if do_qnorm:
            # 1) Load controller (Q-store) for rebuilding
            # Prefer adaptive_q if exists, else training q dir
            q_dir = str(self.runs_dir / f"adaptive_q_seed_{self.train_seeds[0]}")
            if not (Path(q_dir) / "config.json").exists():
                # fallback: use training q dir if you have it
                q_dir = str(self._q_model_dir(self.train_seeds[0]))

            controller = RLGoalApproachController.load(
                q_dir,
                agent_id=f"{mesh_name}_sleep_qnorm",
                config={**self.rl_config, "mode": "adaptive"},
            )

            # 2) Build per-level state pools (current mesh only)
            # We'll mix: (start_states oversampled) + sample_states
            current_mesh_ratio = float(qnorm_cfg.get("current_mesh_ratio", 0.7))
            start_oversample = int(qnorm_cfg.get("start_oversample", 3))
            max_total = int(qnorm_cfg.get("max_states_total", 50000))

            by_level = sleep_q_norm.get("by_level", {})

            # We'll rebuild all stores using data from selected levels only
            # If you want all levels, replace qnorm_levels with by_level.keys()
            use_levels = qnorm_levels

            # Accumulators per store
            cur_free = []
            cur_surface = []
            cur_bypass = []

            for lvl in use_levels:
                lvl_data = by_level.get(lvl, {})
                start_states = lvl_data.get("start", {}).get("states", [])
                sample = lvl_data.get("sample", {})

                # oversample start states to emphasize early-episode proximity
                if start_states:
                    for _ in range(max(1, start_oversample)):
                        # classify by on_object + same_side in state
                        for s in start_states:
                            on_obj = float(s[11]) > 0.5
                            ss = bool(float(s[27]) > 0.5) if len(s) > 27 else True
                            if not on_obj:
                                cur_free.append(s)
                            else:
                                if ss:
                                    cur_surface.append(s)
                                else:
                                    cur_bypass.append(s)

                # add sparse samples already split
                cur_free.extend(sample.get("free", []))
                cur_surface.extend(sample.get("surface", []))
                cur_bypass.extend(sample.get("bypass", []))

            # cap current pools
            cur_free = _cap_list(cur_free, max_total)
            cur_surface = _cap_list(cur_surface, max_total)
            cur_bypass = _cap_list(cur_bypass, max_total)

            # 3) Replay pool from primitive-trained Q-stores (base model)
            rep_free, rep_surface, rep_bypass = [], [], []
            try:
                from .hnsw_state_store import HNSWStateStore

                base_q_dir = str(self._q_model_dir(self.train_seeds[0]))
                logger.info("Sleep Q-norm: loading replay states from base Q-dir: %s", base_q_dir)

                # Load free store
                base_free = HNSWStateStore.load_with_index(
                    os.path.join(base_q_dir, "q_store_free"),
                    extra_cfg={**self.rl_config, "feature_weights": self.rl_config.get("feature_weights_free", None)},
                    name="free_replay",
                )
                rep_free = [p.raw_state.astype(np.float32, copy=False) for p in base_free.points.values()]

                # Load surface store
                base_surface = HNSWStateStore.load_with_index(
                    os.path.join(base_q_dir, "q_store_surface"),
                    extra_cfg={**self.rl_config, "feature_weights": self.rl_config.get("feature_weights_surface", None)},
                    name="surface_replay",
                )
                rep_surface = [p.raw_state.astype(np.float32, copy=False) for p in base_surface.points.values()]

                # Load bypass store if exists
                bypass_path = os.path.join(base_q_dir, "q_store_surface_bypass")
                if Path(bypass_path + ".npz").exists():
                    base_bypass = HNSWStateStore.load_with_index(
                        bypass_path,
                        extra_cfg={**self.rl_config, "feature_weights": self.rl_config.get("feature_weights_bypass", None)},
                        name="bypass_replay",
                    )
                    rep_bypass = [p.raw_state.astype(np.float32, copy=False) for p in base_bypass.points.values()]
                else:
                    rep_bypass = []

                # Cap replay sizes (avoid huge memory)
                replay_cap = int(qnorm_cfg.get("replay_per_store", max_total))
                if len(rep_free) > replay_cap:
                    idx = np.random.choice(len(rep_free), replay_cap, replace=False)
                    rep_free = [rep_free[i] for i in idx]
                if len(rep_surface) > replay_cap:
                    idx = np.random.choice(len(rep_surface), replay_cap, replace=False)
                    rep_surface = [rep_surface[i] for i in idx]
                if len(rep_bypass) > replay_cap:
                    idx = np.random.choice(len(rep_bypass), replay_cap, replace=False)
                    rep_bypass = [rep_bypass[i] for i in idx]

                logger.info(
                    "Sleep Q-norm: replay states loaded: free=%d, surface=%d, bypass=%d",
                    len(rep_free), len(rep_surface), len(rep_bypass),
                )

            except Exception as exc:
                logger.warning("Sleep Q-norm: could not load replay states: %s", exc)
                rep_free, rep_surface, rep_bypass = [], [], []

            # 4) Sample mixed pools
            mix_free = _sample_mix(cur_free, rep_free, max_total, current_mesh_ratio)
            mix_surface = _sample_mix(cur_surface, rep_surface, max_total, current_mesh_ratio)
            mix_bypass = _sample_mix(cur_bypass, rep_bypass, max_total, current_mesh_ratio)

            # 5) Recompute mean/std + rebuild index for each store
            def _apply_norm_and_rebuild(store, states_list, store_label: str):
                if not states_list:
                    logger.warning("Sleep Q-norm: no states for %s, skip", store_label)
                    return

                arr = _to_np(states_list)
                mean, std = _compute_mean_std(arr, min_std=store.norm_min_std)

                store._state_mean = mean
                store._state_std = std

                store._norm_frozen = True
                store._freeze_done = True

                # Rebuild index with new normalization
                store._rebuild_index_with_renorm()

                logger.info(
                    "Sleep Q-norm rebuilt %s: n_states=%d, mean_std_ok",
                    store_label, len(states_list),
                )

            _apply_norm_and_rebuild(controller.q_store_free, mix_free, "free")
            _apply_norm_and_rebuild(controller.q_store_surface, mix_surface, "surface")
            _apply_norm_and_rebuild(controller.q_store_surface_bypass, mix_bypass, "surface_bypass")

            # 6) Save controller back
            controller.save(q_dir)
            logger.info("Sleep Q-norm: controller saved to %s", q_dir)


        # ═══ Load sleep-SAC ═══
        if not trails_path.exists():
            logger.warning("Sleep skipped: no trails file at %s", trails_path)
            return

        # ═══ Load trails ═══
        with trails_path.open("rb") as f:
            sleep_data = pickle.load(f)

        stats = sleep_data.get("stats", {})
        success_trails_by_level = sleep_data.get("success_trails_by_level", {})
        failure_trails_by_level = sleep_data.get("failure_trails_by_level", {})

        logger.info(
            "Sleep: loaded trails for %s — %s",
            mesh_name, stats.get("trails_count_by_level", {}),
        )

        # ═══ Check conditions per level ═══
        levels_to_consolidate = []
        rate_by_level = stats.get("success_rate_by_level", {})

        for level in sorted(success_trails_by_level.keys()):
            level_rate = rate_by_level.get(level, 0.0)
            level_trails = success_trails_by_level.get(level, [])

            if level_rate >= success_threshold:
                logger.info(
                    "Sleep L%d: skipped (rate=%.3f >= %.3f, already good)",
                    level, level_rate, success_threshold,
                )
                continue

            if len(level_trails) < min_success_trails:
                logger.info(
                    "Sleep L%d: skipped (trails=%d < %d)",
                    level, len(level_trails), min_success_trails,
                )
                continue

            levels_to_consolidate.append(level)
            logger.info(
                "Sleep L%d: will consolidate (rate=%.3f, trails=%d)",
                level, level_rate, len(level_trails),
            )

        if not levels_to_consolidate:
            logger.info("Sleep: nothing to consolidate for %s", mesh_name)
            return

        # ═══ Load SAC trainer ═══
        num_types = len(ExperienceExtractor.get_type_names())
        adapt_sac_dir = str(
            self.runs_dir / f"adaptive_sac_seed_{self.sac_seed}"
        )
        sac_model_dir = self._sac_model_dir(self.sac_seed)

        sac_trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 20),
            num_types=num_types,
        )

        # Prefer adaptive SAC (already fine-tuned), fallback to training SAC
        if (Path(adapt_sac_dir) / "sac_actor.pt").exists():
            sac_trainer.load(adapt_sac_dir)
            logger.info("Sleep: SAC loaded from adaptive: %s", adapt_sac_dir)
        elif (Path(sac_model_dir) / "sac_actor.pt").exists():
            sac_trainer.load(sac_model_dir)
            logger.info("Sleep: SAC loaded from training: %s", sac_model_dir)
        else:
            logger.warning("Sleep: no SAC model found, skipping")
            return

        # ═══ Create manager for consolidation ═══
        adaptive_cfg = {**self.rl_config, "mode": "adaptive"}
        adapt_q_dir = str(
            self.runs_dir / f"adaptive_q_seed_{self.train_seeds[0]}"
        )

        manager = AdaptiveTrainingManager(
            controller=None,
            env=None,
            config=adaptive_cfg,
            runs_dir=str(self.runs_dir),
            mesh_path=str(self.data_dir / f"{mesh_name}.stl"),
            q_save_dir=adapt_q_dir,
            sac_save_dir=adapt_sac_dir,
        )

        manager.sac_trainer = sac_trainer

        # ═══ Consolidate each level ═══
        sleep_results = []

        for level in levels_to_consolidate:
            success_trails = success_trails_by_level.get(level, [])
            failure_trails = failure_trails_by_level.get(level, [])

            result = manager.sleep_sac_consolidation(
                object_name=mesh_name,
                success_trails=success_trails,
                failure_trails=failure_trails,
                level=level,
                sac_update_steps=sleep_cfg.get("sac_update_steps", 3000),
                critic_warmup_steps=sleep_cfg.get("critic_warmup_steps", 200),
                actor_update_every=sleep_cfg.get("actor_update_every", 5),
                actor_lr_scale=sleep_cfg.get("actor_lr_scale", 0.3),
                bc_lambda_init=sleep_cfg.get("bc_lambda_init", 3.0),
                bc_lambda_decay=sleep_cfg.get("bc_lambda_decay", 0.999),
                bc_lambda_min=sleep_cfg.get("bc_lambda_min", 0.3),
                use_cql=sleep_cfg.get("use_cql", True),
                cql_alpha=sleep_cfg.get("cql_alpha", 0.5),
                current_mesh_ratio=sleep_cfg.get("current_mesh_ratio", 0.7),
                bc_ratio=sleep_cfg.get("bc_ratio", 0.2),
                elite_ratio=sleep_cfg.get("elite_ratio", 0.1),
            )
            sleep_results.append(result)

        # ═══ Save sleep report ═══
        import json
        report = {
            "mesh_name": mesh_name,
            "levels_consolidated": levels_to_consolidate,
            "results": sleep_results,
            "config": sleep_cfg,
        }
        report_path = log_dir / "sleep_report.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            "Sleep complete for %s: %d levels consolidated, report at %s",
            mesh_name, len(levels_to_consolidate), report_path,
        )

    # ══════════════════════════════════════════════════════
    # YCB Evaluation
    # ══════════════════════════════════════════════════════

    def _run_ycb_eval(self) -> None:
        """Evaluate trained agent on YCB objects.

        Converts YCB .glb → .stl (mm), then runs eval using
        LightweightEnv (same as standard eval).
        """
        logger.info("=" * 60)
        logger.info("YCB Evaluation")
        logger.info("=" * 60)

        ycb_src_dir = self.ycb_config.get("ycb_src_dir")
        if not ycb_src_dir:
            logger.error(
                "ycb_config.ycb_src_dir not set. "
                "Set path to YCB meshes directory."
            )
            return

        ycb_src_dir = Path(ycb_src_dir).expanduser()
        if not ycb_src_dir.exists():
            logger.error("YCB source dir not found: %s", ycb_src_dir)
            return

        # Parse which objects to evaluate
        ycb_objects = self.ycb_config.get("objects", None)
        # Format: {local_name: ycb_folder_name}
        # e.g. {"ycb_mug": "025_mug", "ycb_bowl": "024_bowl"}

        ycb_episodes_per_level = self.ycb_config.get(
            "episodes_per_level", 100
        )
        ycb_curriculum_levels = [
            tuple(level)
            for level in self.ycb_config.get(
                "curriculum_levels", self.curriculum_levels
            )
        ]
        ycb_curriculum_filters = self.ycb_config.get(
            "curriculum_filters", self.curriculum_filters
        )

        # Convert YCB objects
        ycb_data_dir = self.data_dir / "ycb"
        logger.info("Converting YCB objects from %s", ycb_src_dir)
        converted = convert_ycb_objects(
            ycb_src_dir=str(ycb_src_dir),
            output_dir=str(ycb_data_dir),
            objects=ycb_objects,
        )

        if not converted:
            logger.error("No YCB objects converted, aborting")
            return

        logger.info("Converted %d YCB objects", len(converted))

        # Eval config
        #eval_cfg = {
        #    **self.rl_config,
        #    "mode": "eval",
        #}
        eval_cfg = {
            **self.rl_config,
            "mode": "eval",
            "eval_epsilon": 1.0,
            "temperature_override": 0.01,
            "strategic_eval_epsilon": 1.0,
            "air_start_enabled": False,
            "air_start_in_eval": False,
        }

        all_ycb_results = {}

        for mesh_name, stl_path in converted.items():
            mesh_path = str(stl_path)
            logger.info(
                "YCB Eval: %s (%s)", mesh_name, stl_path
            )

            # Generate episode pools
            eval_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=self.eval_seeds,
                episodes_per_level=ycb_episodes_per_level,
                scripts_dir=self.scripts_dir,
                curriculum_levels=ycb_curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"ycb_eval_{mesh_name}",
                curriculum_filters=ycb_curriculum_filters,
            )

            # Run eval
            eval_results, _ = run_eval_per_seed(
                data_dir=self.data_dir,
                runs_dir=self.runs_dir,
                mesh_path=mesh_path,
                train_seeds=self.train_seeds,
                eval_seeds=self.eval_seeds,
                variant="q_store",
                eval_cfg=eval_cfg,
                eval_pools=eval_pools,
                collect_bc=False,
                episodes_per_level=ycb_episodes_per_level,
                mesh_name=mesh_name,
                visualise=self.visualise,
                log_prefix="heuristic",
            )

            all_ycb_results[mesh_name] = eval_results

            # Log per-level results
            for level_key, level_data in eval_results.items():
                if level_key == "overall":
                    continue
                logger.info(
                    "  %s %s: success=%.3f, timeout=%.3f, collision=%.3f",
                    mesh_name,
                    level_key,
                    level_data.get("mean_success_rate", 0),
                    level_data.get("mean_timeout_rate", 0),
                    level_data.get("mean_collision_rate", 0),
                )

            if "overall" in eval_results:
                logger.info(
                    "  %s OVERALL: success=%.3f",
                    mesh_name,
                    eval_results["overall"].get("mean_success_rate", 0),
                )

            # Save per-mesh result
            mesh_result_path = (
                self.data_dir / f"ycb_eval_result_{mesh_name}.json"
            )
            with mesh_result_path.open("w") as f:
                json.dump(eval_results, f, indent=2)

        # Save combined results
        ycb_result_path = self.data_dir / "ycb_eval_result_all.json"
        with ycb_result_path.open("w") as f:
            json.dump(all_ycb_results, f, indent=2)

        # Summary table
        logger.info("\n" + "=" * 60)
        logger.info("YCB EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info("%-20s %10s %10s %10s", "Object", "Success", "Timeout", "Collision")
        logger.info("-" * 60)
        for mesh_name, results in all_ycb_results.items():
            overall = results.get("overall", {})
            logger.info(
                "%-20s %10.3f %10.3f %10.3f",
                mesh_name,
                overall.get("mean_success_rate", 0),
                overall.get("mean_timeout_rate", 0),
                overall.get("mean_collision_rate", 0),
            )
        logger.info("=" * 60)

        logger.info("YCB eval results saved to %s", ycb_result_path)

    # ═════════════════════════════════════════════════════
    # Utilities
    # ═════════════════════════════════════════════════════

    @staticmethod
    def _filter_bc_transitions(
        transitions: list,
        total_target: int = 100000,
        min_samples_to_keep: int = 50,
    ) -> list:
        """Balance BC transitions by (mesh, level).

        Distributes total_target equally across groups.
        Groups with fewer samples are oversampled,
        groups with more are undersampled.

        Args:
            transitions: All BC transitions.
            total_target: Total desired transitions.
            min_samples_to_keep: Minimum for inclusion.

        Returns:
            Balanced transitions.
        """
        type_names = (
            ExperienceExtractor.get_type_names()
        )
        mesh_id_to_name = {
            v: k
            for k, v in (
                ExperienceExtractor
                .MESH_NAME_TO_ID.items()
            )
        }

        # Group by (mesh_id, level)
        groups: dict[tuple[int, int], list] = {}
        for tr in transitions:
            key = (
                getattr(tr, "mesh_id", -1),
                getattr(tr, "level", 0),
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(tr)

        # Calculate target per group
        num_groups = max(len(groups), 1)
        target_per_group = total_target // num_groups

        # Log before
        logger.info("BC balance BEFORE:")
        for (mid, level), trs in sorted(
            groups.items()
        ):
            mname = mesh_id_to_name.get(
                mid, f"mesh_{mid}"
            )
            type_counts: dict[str, int] = {}
            for tr in trs:
                tname = type_names.get(
                    tr.action_type,
                    f"type_{tr.action_type}",
                )
                type_counts[tname] = (
                    type_counts.get(tname, 0) + 1
                )
            logger.info(
                "  %s L%d: %d transitions, "
                "actions: %s",
                mname,
                level,
                len(trs),
                type_counts,
            )

        # Balance
        balanced: list = []
        for (mid, level), trs in groups.items():
            mname = mesh_id_to_name.get(
                mid, f"mesh_{mid}"
            )

            if len(trs) < min_samples_to_keep:
                balanced.extend(trs)
                logger.info(
                    "  %s L%d: kept all %d "
                    "(too few)",
                    mname,
                    level,
                    len(trs),
                )
            elif len(trs) >= target_per_group:
                indices = (
                    np.random.permutation(
                        len(trs)
                    )[:target_per_group]
                )
                balanced.extend(
                    [trs[i] for i in indices]
                )
                logger.info(
                    "  %s L%d: undersampled "
                    "%d → %d",
                    mname,
                    level,
                    len(trs),
                    target_per_group,
                )
            else:
                indices = np.random.choice(
                    len(trs),
                    target_per_group,
                    replace=True,
                )
                balanced.extend(
                    [trs[i] for i in indices]
                )
                logger.info(
                    "  %s L%d: oversampled "
                    "%d → %d",
                    mname,
                    level,
                    len(trs),
                    target_per_group,
                )

        np.random.shuffle(balanced)

        # Log after
        after_groups: dict[str, int] = {}
        for tr in balanced:
            mname = mesh_id_to_name.get(
                getattr(tr, "mesh_id", -1),
                "unknown",
            )
            key = (
                f"{mname}_L"
                f"{getattr(tr, 'level', 0)}"
            )
            after_groups[key] = (
                after_groups.get(key, 0) + 1
            )

        logger.info("BC balance AFTER:")
        for key in sorted(after_groups.keys()):
            logger.info(
                "  %s: %d transitions (%.1f%%)",
                key,
                after_groups[key],
                100.0 * after_groups[key]
                / max(len(balanced), 1),
            )

        logger.info(
            "BC filter: %d → %d transitions",
            len(transitions),
            len(balanced),
        )
        return balanced
