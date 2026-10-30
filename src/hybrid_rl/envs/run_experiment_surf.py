# Copyright 2026 Aleksey Isaev
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

"""CLI entry point for hybrid-rl-navigation experiments.

Usage::

    # Run with default config
    python -m hybrid_rl.run

    # Run with specific experiment config
    python -m hybrid_rl.run --config experiments/full_pipeline.yaml

    # Override parameters
    python -m hybrid_rl.run --config experiments/full_pipeline.yaml \\
        do_train=true do_eval=true seed=42

    # Override nested parameters
    python -m hybrid_rl.run \\
        logging.run_name=my_experiment \\
        rl_config.epsilon_start=0.5

    # List-style overrides (use quotes)
    python -m hybrid_rl.run \\
        "train_seeds=[11, 22, 33]"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from surface.experiment import RLGoalApproachExperiment

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# Config resolution
# ═══════════════════════════════════════════════════════

def _find_config_file(config_path: str) -> Path:
    """Resolve config path with fallback search order.

    Search order:
        1. Absolute or relative to CWD
        2. Relative to package configs/ directory
        3. Relative to package configs/experiments/

    Args:
        config_path: User-provided config path.

    Returns:
        Resolved absolute Path.

    Raises:
        FileNotFoundError: If config not found in any location.
    """
    # 1. Direct path (absolute or relative to CWD)
    direct = Path(config_path)
    if direct.is_file():
        return direct.resolve()

    # 2. Relative to package configs/
    package_configs = Path(__file__).parent.parent / "configs"
    pkg_path = package_configs / config_path
    if pkg_path.is_file():
        return pkg_path.resolve()

    # 3. Relative to configs/experiments/
    exp_path = package_configs / "experiments" / config_path
    if exp_path.is_file():
        return exp_path.resolve()

    # 4. Try adding .yaml extension
    for candidate in [direct, pkg_path, exp_path]:
        with_ext = candidate.with_suffix(".yaml")
        if with_ext.is_file():
            return with_ext.resolve()

    searched = [
        str(direct),
        str(pkg_path),
        str(exp_path),
    ]
    raise FileNotFoundError(
        f"Config '{config_path}' not found. "
        f"Searched:\n  " + "\n  ".join(searched)
    )


def load_config(
    config_path: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load experiment config with defaults and overrides.

    Pipeline:
        1. Load package default config (configs/default.yaml)
        2. Load user experiment config
        3. Merge (experiment overrides defaults)
        4. Apply CLI overrides
        5. Resolve interpolations

    Args:
        config_path: Path to experiment YAML config.
        overrides: List of "key=value" CLI overrides.

    Returns:
        Fully resolved DictConfig.
    """
    # 1. Load defaults
    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if default_path.is_file():
        default_cfg = OmegaConf.load(default_path)
        logger.debug("Loaded defaults from %s", default_path)
    else:
        default_cfg = OmegaConf.create({})
        logger.debug("No default.yaml found, using empty defaults")

    # 2. Load experiment config
    experiment_path = _find_config_file(config_path)
    experiment_cfg = OmegaConf.load(experiment_path)
    logger.info("Loaded experiment config: %s", experiment_path)

    # Handle nested 'experiment.config' structure (backward compat)
    if "experiment" in experiment_cfg and "config" in experiment_cfg["experiment"]:
        experiment_cfg = experiment_cfg["experiment"]["config"]

    # 3. Merge: experiment overrides defaults
    merged = OmegaConf.merge(default_cfg, experiment_cfg)

    # 4. Apply CLI overrides via OmegaConf
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli_cfg)
        logger.info(
            "Applied %d CLI override(s): %s",
            len(overrides),
            overrides,
        )

    # 5. Resolve interpolations (${...} references)
    OmegaConf.resolve(merged)

    return merged


# ═══════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════

def setup_logging(config: DictConfig) -> None:
    """Configure logging from config.

    Args:
        config: Experiment config with optional logging section.
    """
    log_level = OmegaConf.select(
        config, "logging.level", default="INFO"
    )
    log_format = OmegaConf.select(
        config, "logging.format",
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log_file = OmegaConf.select(
        config, "logging.file", default=None
    )

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path)))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing config
    )

    # Suppress noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "trimesh", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    """Run hybrid-rl experiment from CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        description="Hybrid RL Navigation — experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="rl_goal_approach.yaml",
        help=(
            "Path to experiment YAML config. "
            "Searched in: CWD, package configs/, "
            "configs/experiments/. "
            "(default: default.yaml)"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Config overrides in dot notation: "
            "key=value key.nested=value "
            "(e.g., do_train=true seed=42 "
            "rl_config.epsilon_start=0.3)"
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config and exit (dry run).",
    )

    args = parser.parse_args(argv)

    # ═══ Load config ═══
    try:
        config = load_config(args.config, args.overrides)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # ═══ Setup logging ═══
    setup_logging(config)

    # ═══ Print config (dry run) ═══
    if args.print_config:
        print(OmegaConf.to_yaml(config))
        return 0

    # ═══ Log experiment info ═══
    run_name = OmegaConf.select(
        config, "logging.run_name", default="experiment"
    )
    logger.info("=" * 60)
    logger.info("Experiment: %s", run_name)
    logger.info("Config: %s", args.config)
    if args.overrides:
        logger.info("Overrides: %s", args.overrides)
    logger.info("=" * 60)

    # ═══ Log active stages ═══
    stages = [
        ("do_train", "Q-Learning Training"),
        ("do_eval", "Q-Learning Evaluation"),
        ("do_heur_eval", "Heuristic Evaluation"),
        ("do_bc_train", "Behavioral Cloning"),
        ("do_sac_train", "SAC Training"),
        ("do_sac_eval", "SAC Evaluation"),
        ("do_adaptive", "Adaptive Mode"),
        ("do_ycb_eval", "YCB Evaluation"),
        ("do_ycb_mujoco_eval", "YCB MuJoCo Evaluation"),
        ("do_ycb_mujoco_adaptive", "YCB MuJoCo Adaptive"),
    ]
    active = [
        name for key, name in stages
        if OmegaConf.select(config, key, default=False)
    ]
    if active:
        logger.info("Active stages: %s", ", ".join(active))
    else:
        logger.warning(
            "No stages enabled! Set do_train=true, "
            "do_eval=true, etc."
        )
        return 0

    # ═══ Run experiment ═══
    start_time = time.time()

    try:
        with RLGoalApproachExperiment(config) as experiment:
            experiment.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Experiment failed")
        return 1

    elapsed = time.time() - start_time
    logger.info(
        "Experiment '%s' completed in %.1fs (%.1f min)",
        run_name,
        elapsed,
        elapsed / 60,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
