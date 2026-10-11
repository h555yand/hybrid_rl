"""Run drone RL experiment from YAML config.

Usage:
    python run_experiment.py configs/drone_vel_experiment.yaml
    python run_experiment.py configs/drone_vel_experiment.yaml --override seed=123
    python run_experiment.py configs/drone_vel_experiment.yaml --override do_train=false do_eval=true
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml
import os

from drone.pybullet_drone_env import PyBulletDroneEnv
from drone.drone_controller_vel import DroneControllerVel
from drone.curriculum import CurriculumManager, TRAP_SCENARIOS
from drone.step_logger import StepLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiment")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════

def load_config(yaml_path: str) -> dict:
    # 1. Получаем абсолютный путь к текущему файлу run_experiment.py
    current_file = Path(__file__).resolve()
    
    # 2. Поднимаемся к главной папке библиотеки (src/hybrid_rl/)
    # current_file.parent — это папка envs/
    # current_file.parent.parent — это папка hybrid_rl/
    library_root = current_file.parent.parent
    
    # 3. Строим абсолютный путь к файлу конфигурации относительно корня библиотеки
    # Если папка configs/ лежит прямо внутри src/hybrid_rl/
    yaml_path = library_root / yaml_path
    
    # Для отладки можно распечатать получившийся путь, если файл не найдут:
    # print(f"Ищу конфигурацию по пути: {yaml_path}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return raw["experiment"]["config"]


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)

        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() in ("null", "none"):
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        keys = key.split(".")
        target = config
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
        logger.info("Override: %s = %s", key, value)

    return config


def build_controller_config(config: dict, stage: dict) -> dict:
    """Build flat controller config from rl_config + stage."""
    ctrl_cfg = {}
    ctrl_cfg.update(config.get("rl_config", {}))

    # Stage overrides
    for key in ("epsilon_start", "epsilon_min", "warmup_episodes",
                "epsilon_decay"):
        if key in stage:
            ctrl_cfg[key] = stage[key]

    # Demo config
    demo_cfg = config.get("demo_config", {})
    for key in ("demo_path", "demo_learning_rate", "demo_reward_bonus"):
        if key in demo_cfg:
            ctrl_cfg[key] = demo_cfg[key]

    # Environment params
    for key in ("flight_dome_size", "ctrl_freq", "pyb_freq",
                "initial_altitude"):
        if key in config:
            ctrl_cfg[key] = config[key]

    return ctrl_cfg

# ═══════════════════════════════════════════════════════
# Goal & obstacle generation (from stage config)
# ═══════════════════════════════════════════════════════

def generate_goal(
    stage: dict,
    episode_idx: int,
    reference_pos: np.ndarray,
    dome_size: float,
    base_seed: int,
) -> np.ndarray:
    """Generate goal from stage params."""
    ep_seed = base_seed + episode_idx
    rng = np.random.RandomState(ep_seed)

    min_dist = stage["min_dist"]
    max_dist = stage["max_dist"]
    min_alt = stage["min_altitude"]
    max_alt = stage["max_altitude"]
    dome_margin = dome_size * 0.8

    for _ in range(100):
        direction = rng.randn(3)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        direction /= norm

        dist = rng.uniform(min_dist, max_dist)
        point = reference_pos + direction * dist
        point[2] = np.clip(point[2], min_alt, max_alt)

        if np.linalg.norm(point) < dome_margin:
            return np.concatenate([point, np.zeros(3)])

    point = reference_pos + np.array([min_dist, 0.0, 0.5])
    point[2] = max(point[2], min_alt)
    return np.concatenate([point, np.zeros(3)])


def generate_obstacles(
    stage: dict,
    episode_idx: int,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    base_seed: int,
) -> list[dict]:
    """Generate obstacles from stage params.

    All obstacles start from ground level (z=0) with height
    guaranteed to cover drone flight altitude.
    """
    if not stage.get("obstacles", False):
        return []

    num_obs = stage.get("num_obstacles", 3)
    ep_seed = base_seed + episode_idx + 999
    rng = np.random.RandomState(ep_seed)

    obstacles = []
    direction = goal_pos - start_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return []

    dir_norm = direction / dist

    if abs(dir_norm[2]) < 0.9:
        perp1 = np.cross(dir_norm, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(dir_norm, np.array([1, 0, 0]))
    perp1 /= np.linalg.norm(perp1) + 1e-12
    perp2 = np.cross(dir_norm, perp1)

    for i in range(num_obs):
        t = rng.uniform(0.2, 0.8)
        center = start_pos + dir_norm * dist * t

        offset1 = rng.uniform(-1.0, 1.0)
        offset2 = rng.uniform(-0.5, 0.5)
        pos = center + perp1 * offset1 + perp2 * offset2
        pos[2] = 0.0  # always from ground

        radius = rng.uniform(0.3, 0.8)
        height = rng.uniform(2.0, 4.0)

        obstacles.append({
            "position": pos.tolist(),
            "radius": round(float(radius), 3),
            "height": round(float(height), 3),
        })

    return obstacles


def is_trap_episode(
    stage: dict, episode_idx: int, base_seed: int
) -> bool:
    """Decide if this episode should be a trap."""
    trap_ratio = stage.get("trap_ratio", 0.0)
    if trap_ratio <= 0:
        return False
    rng = np.random.RandomState(base_seed + episode_idx + 777)
    return rng.random() < trap_ratio


def generate_trap(
    episode_idx: int,
    start_pos: np.ndarray,
    base_seed: int,
) -> tuple[np.ndarray, list[dict], str]:
    """Generate trap scenario.

    Obstacles placed from ground level, offset only in XY
    from drone start position.
    """
    rng = np.random.RandomState(base_seed + episode_idx + 888)
    trap = TRAP_SCENARIOS[rng.randint(len(TRAP_SCENARIOS))]

    goal_offset = np.array(trap["goal_offset"], dtype=float)
    goal_pos = start_pos + goal_offset
    goal_pos[2] = max(goal_pos[2], 0.3)
    goal_pose = np.concatenate([goal_pos, np.zeros(3)])

    obstacles = []
    for obs in trap["obstacles"]:
        pos = np.array(obs["position"], dtype=float)
        pos[0] += start_pos[0]
        pos[1] += start_pos[1]
        pos[2] = 0.0  # always from ground

        obstacles.append({
            "position": pos.tolist(),
            "radius": obs["radius"],
            "height": obs.get("height", 3.0),
        })

    return goal_pose, obstacles, trap["name"]


# ═══════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════

def run_episode(
    env: PyBulletDroneEnv,
    controller: DroneControllerVel,
    stage: dict,
    episode_idx: int,
    base_seed: int,
    max_steps: int = 6000,
) -> dict:
    """Run single episode using stage config for generation."""
    sensor_data = env.reset()
    pos = np.array(sensor_data["position"])

    # Check if trap episode
    trap = is_trap_episode(stage, episode_idx, base_seed)
    trap_name = ""

    if trap:
        goal_pose, obstacles, trap_name = generate_trap(
            episode_idx, pos, base_seed,
        )
    else:
        goal_pose = generate_goal(
            stage, episode_idx, pos,
            env._dome_size, base_seed,
        )
        obstacles = generate_obstacles(
            stage, episode_idx, pos,
            goal_pose[:3], base_seed,
        )

    env.set_goal(goal_pose)
    env.spawn_obstacles(obstacles)

    sensor_data = env.get_sensor_data()
    start_pos = pos.copy()
    start_dist = float(sensor_data["distance_to_goal"])

    controller.set_new_goal(goal_pose, start_pos)

    # Snapshot termination counts before episode
    prev_counts = dict(controller._termination_counts)

    episode_log = {
        "episode": episode_idx,
        "stage": stage["name"],
        "start_distance": round(start_dist, 4),
        "num_obstacles": len(obstacles),
        "is_trap": trap,
        "trap_name": trap_name,
        "result": None,
        "total_steps": 0,
        "reward": 0.0,
    }

    for step in range(max_steps):
        action_idx, explanation = controller.step(sensor_data)

        if action_idx is None:
            episode_log["total_steps"] = step
            break

        sensor_data = env.step_discrete(
            action_idx, controller.action_space
        )

    # Determine result by comparing counts
    current_counts = dict(controller._termination_counts)
    for term_type in ("goal_reached", "collision",
                      "out_of_bounds", "timeout"):
        if current_counts.get(term_type, 0) > prev_counts.get(
            term_type, 0
        ):
            episode_log["result"] = term_type
            break

    if episode_log["result"] is None:
        episode_log["result"] = "timeout"
        episode_log["total_steps"] = max_steps

    final_dist = float(sensor_data["distance_to_goal"])
    episode_log["final_distance"] = round(final_dist, 4)
    episode_log["reward"] = round(controller._episode_reward, 2)
    episode_log["epsilon"] = round(controller.epsilon, 4)

    return episode_log


# ═══════════════════════════════════════════════════════
# Demonstrations
# ═══════════════════════════════════════════════════════

def generate_demonstrations(config: dict):
    demo_cfg = config.get("demo_config", {})
    demo_path = demo_cfg.get("demo_path", "demos/trap_demonstrations.npz")

    if Path(demo_path).exists():
        logger.info("Demonstrations already exist: %s", demo_path)
        return

    logger.info("Generating demonstrations...")
    from drone.generate_demonstrations import main as gen_main

    import sys
    old_argv = sys.argv
    sys.argv = [
        "generate_demonstrations",
        "--output", demo_cfg.get("output_dir", "demos"),
        "--repeats", str(demo_cfg.get("repeats", 5)),
    ]
    try:
        gen_main()
    finally:
        sys.argv = old_argv

    logger.info("Demonstrations generated: %s", demo_path)


# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

def run_training_stage(
    config: dict,
    stage: dict,
    stage_idx: int,
    controller: DroneControllerVel | None,
    output_dir: Path,
    log_dir: Path,
) -> DroneControllerVel:
    """Run one training stage."""
    stage_name = stage["name"]
    episodes = stage["episodes"]
    max_steps = config["rl_config"]["max_steps_per_goal"]
    seed = config["seed"] + stage_idx * 10000

    logger.info(
        "\n%s\nSTAGE %d: %s (%d episodes)\n%s",
        "=" * 60, stage_idx, stage_name, episodes, "=" * 60,
    )

    np.random.seed(seed)

    ctrl_cfg = build_controller_config(config, stage)
    ctrl_cfg["mode"] = "train"

    # ── Auto-calculate epsilon_decay ──
    eps_start = ctrl_cfg.get("epsilon_start", 1.0)
    eps_min = ctrl_cfg.get("epsilon_min", 0.05)
    warmup = ctrl_cfg.get("warmup_episodes", 30)
    decay_episodes = max(episodes - warmup, 1)

    if eps_start > eps_min and decay_episodes > 0:
        auto_decay = (eps_min / eps_start) ** (1.0 / decay_episodes)
    else:
        auto_decay = 1.0

    # Stage can override with explicit decay
    ctrl_cfg["epsilon_decay"] = stage.get(
        "epsilon_decay", auto_decay
    )

    logger.info(
        "Epsilon: %.3f → %.3f over %d episodes "
        "(warmup=%d, decay=%.6f)",
        eps_start, eps_min, episodes,
        warmup, ctrl_cfg["epsilon_decay"],
    )

    # Create or continue controller
    if stage.get("load_mode") == "auto" and controller is not None:
        controller.epsilon = stage.get(
            "epsilon_start", controller.epsilon
        )
        controller.epsilon_min = stage.get(
            "epsilon_min", controller.epsilon_min
        )
        controller.config.update(ctrl_cfg)
        logger.info(
            "Continuing from previous stage (ε=%.3f)",
            controller.epsilon,
        )
    elif stage.get("load_mode") == "auto":
        ckpt = output_dir / "latest"
        if ckpt.exists():
            controller = DroneControllerVel.load(
                str(ckpt), agent_id="drone_0", config=ctrl_cfg,
            )
            controller.epsilon = stage.get(
                "epsilon_start", controller.epsilon
            )
            logger.info("Loaded checkpoint from %s", ckpt)
        else:
            controller = DroneControllerVel(
                agent_id="drone_0", config=ctrl_cfg,
            )
    else:
        controller = DroneControllerVel(
            agent_id="drone_0", config=ctrl_cfg,
        )

    # Load demonstrations (first stage only)
    if stage_idx == 0:
        demo_path = config.get("demo_config", {}).get(
            "demo_path", "demos/trap_demonstrations.npz"
        )
        if Path(demo_path).exists():
            controller.load_demonstrations(demo_path)

    # Environment
    env = PyBulletDroneEnv(
        render=config.get("render", False),
        flight_dome_size=config["flight_dome_size"],
        max_steps=max_steps,
        goal_reach_distance=ctrl_cfg["goal_reach_distance"],
        initial_altitude=config["initial_altitude"],
        ctrl_freq=config["ctrl_freq"],
        pyb_freq=config["pyb_freq"],
    )

    # Debug logger
    log_cfg = config.get("logging", {})
    if log_cfg.get("debug", False):
        step_logger = StepLogger(
            output_dir=str(log_dir / "debug" / stage_name),
            console_every=log_cfg.get("log_every", 1),
            enabled=True,
        )
        controller.set_step_logger(step_logger)

    # Tracking
    save_every = log_cfg.get("save_every", 100)
    recent_results = deque(maxlen=50)
    total_goals = 0
    t_start = time.time()
    stage_logs = []

    for ep in range(1, episodes + 1):
        episode_log = run_episode(
            env, controller, stage, ep, seed, max_steps,
        )

        result = episode_log["result"]
        is_success = result == "goal_reached"
        if is_success:
            total_goals += 1
        recent_results.append(1 if is_success else 0)

        success_rate = (
            sum(recent_results) / len(recent_results)
        )

        stage_logs.append(episode_log)

        # Print every episode
        elapsed = time.time() - t_start
        status = "✅" if is_success else "❌"
        trap_str = (
            f" 🪤{episode_log['trap_name']}"
            if episode_log.get("is_trap") else ""
        )
        print(
            f"[{stage_name}] "
            f"Ep {ep:4d}/{episodes} {status} "
            f"{result:15s}{trap_str} | "
            f"dist: {episode_log['start_distance']:.1f}"
            f"→{episode_log['final_distance']:.2f}m | "
            f"steps: {episode_log['total_steps']:5d} | "
            f"reward: {episode_log['reward']:+7.1f} | "
            f"ε={episode_log['epsilon']:.3f} | "
            f"SR50={success_rate:.0%} | "
            f"goals={total_goals}/{ep} | "
            f"Q={controller.q_store_go.next_id}+{controller.q_store_avoid.next_id}"
            f"{elapsed:.0f}s"
        )

        # Save checkpoint
        if ep % save_every == 0:
            controller.save(str(output_dir / "latest"))

    # Save stage checkpoint
    stage_dir = output_dir / stage_name
    controller.save(str(stage_dir))
    with (stage_dir / "stage_log.json").open("w") as f:
        json.dump(stage_logs, f, indent=2, cls=NumpyEncoder)

    # Save stage summary
    sr = total_goals / max(episodes, 1)
    collisions = sum(
        1 for l in stage_logs if l["result"] == "collision"
    )
    timeouts = sum(
        1 for l in stage_logs if l["result"] == "timeout"
    )
    oob = sum(
        1 for l in stage_logs if l["result"] == "out_of_bounds"
    )
    trap_episodes = sum(
        1 for l in stage_logs if l.get("is_trap")
    )
    trap_successes = sum(
        1 for l in stage_logs
        if l.get("is_trap") and l["result"] == "goal_reached"
    )
    avg_reward = (
        sum(l["reward"] for l in stage_logs) / max(episodes, 1)
    )
    avg_steps = (
        sum(l["total_steps"] for l in stage_logs) / max(episodes, 1)
    )

    stage_summary = {
        "stage_name": stage_name,
        "stage_index": stage_idx,
        "episodes": episodes,
        "total_goals": total_goals,
        "success_rate": round(sr, 3),
        "collisions": collisions,
        "timeouts": timeouts,
        "out_of_bounds": oob,
        "trap_episodes": trap_episodes,
        "trap_successes": trap_successes,
        "trap_success_rate": round(
            trap_successes / max(trap_episodes, 1), 3
        ),
        "avg_reward": round(avg_reward, 1),
        "avg_steps": round(avg_steps, 1),
        "final_epsilon": round(controller.epsilon, 4),
        "q_store_go_points": controller.q_store_go.next_id,
        "q_store_avoid_points": controller.q_store_avoid.next_id,
        "time_seconds": round(time.time() - t_start, 1),
        "sr_last_50": round(
            sum(recent_results) / max(len(recent_results), 1), 3
        ),
        "config": {
            "epsilon_start": stage.get("epsilon_start"),
            "epsilon_min": stage.get("epsilon_min"),
            "min_dist": stage.get("min_dist"),
            "max_dist": stage.get("max_dist"),
            "obstacles": stage.get("obstacles"),
            "num_obstacles": stage.get("num_obstacles"),
            "trap_ratio": stage.get("trap_ratio"),
        },
    }

    with (stage_dir / "stage_summary.json").open("w") as f:
        json.dump(stage_summary, f, indent=2)

    env.close()

    print(
        f"\n{'─'*60}\n"
        f"Stage '{stage_name}' complete: "
        f"{total_goals}/{episodes} goals ({sr:.0%}) | "
        f"col={collisions} tout={timeouts} oob={oob} | "
        f"traps={trap_successes}/{trap_episodes} | "
        f"avg_r={avg_reward:.0f} avg_s={avg_steps:.0f} | "
        f"Q={controller.q_store_go.next_id}+{controller.q_store_avoid.next_id}"
        f"{time.time() - t_start:.0f}s\n"
        f"{'─'*60}\n"
    )

    return controller


def train(config: dict):
    """Run all training stages sequentially."""
    log_cfg = config.get("logging", {})
    output_dir = Path(
        log_cfg.get("checkpoint_dir", "checkpoints/drone_vel")
    )
    log_dir = Path(log_cfg.get("output_dir", "results/drone_vel"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    stages = config.get("training_stages", [])
    controller = None
    t_total = time.time()

    all_stage_summaries = []

    for idx, stage in enumerate(stages):
        controller = run_training_stage(
            config, stage, idx, controller, output_dir, log_dir,
        )
        # Load stage summary
        stage_dir = output_dir / stage["name"]
        summary_path = stage_dir / "stage_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                all_stage_summaries.append(json.load(f))

    # Final save
    if controller:
        controller.save(str(output_dir / "final"))

    total_time = time.time() - t_total
    stats = controller.get_stats() if controller else {}

    # Print summary table
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(
        f"{'Stage':<25s} {'SR':>6s} {'Goals':>8s} "
        f"{'Col':>5s} {'Tout':>5s} {'Traps':>8s} "
        f"{'AvgR':>7s} {'Time':>6s}"
    )
    print("─" * 75)
    for ss in all_stage_summaries:
        print(
            f"{ss['stage_name']:<25s} "
            f"{ss['success_rate']:>5.0%} "
            f"{ss['total_goals']:>4d}/{ss['episodes']:<4d}"
            f"{ss['collisions']:>5d} "
            f"{ss['timeouts']:>5d} "
            f"{ss['trap_successes']:>3d}/{ss['trap_episodes']:<4d} "
            f"{ss['avg_reward']:>+7.0f} "
            f"{ss['time_seconds']:>5.0f}s"
        )
    print("─" * 75)
    print(f"Total time: {total_time:.0f}s")
    q_go = stats.get('q_store_go', {}).get('num_points', 0)
    q_avoid = stats.get('q_store_avoid', {}).get('num_points', 0)
    print(f"Q-store: GO={q_go}, AVOID={q_avoid} points")
    print(f"Checkpoints: {output_dir}/")

    # Save overall summary
    training_summary = {
        "config": config,
        "total_time": round(total_time, 1),
        "stages": all_stage_summaries,
        "final_stats": stats,
    }
    with (log_dir / "training_summary.json").open("w") as f:
        json.dump(training_summary, f, indent=2, cls=NumpyEncoder)

    return controller

# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════

def evaluate(config: dict):
    log_cfg = config.get("logging", {})
    checkpoint_dir = Path(
        log_cfg.get("checkpoint_dir", "checkpoints/drone_vel")
    ) / "final"

    eval_stages = config.get("eval_stages", [])
    eval_seeds = config.get("eval_seeds", [44])
    max_steps = config["rl_config"]["max_steps_per_goal"]

    # Build eval config
    ctrl_cfg = build_controller_config(
        config, eval_stages[0] if eval_stages else {}
    )
    ctrl_cfg["mode"] = "eval"

    controller = DroneControllerVel.load(
        str(checkpoint_dir), agent_id="drone_0",
        config=ctrl_cfg,
    )

    # ── Force eval settings ──
    controller.mode = "eval"
    # Reset counters for clean eval stats
    controller._total_episodes = 0
    controller._total_steps = 0
    controller._total_goals_reached = 0
    controller._termination_counts = {
        "goal_reached": 0, "collision": 0,
        "out_of_bounds": 0, "timeout": 0,
    }

    # ── Step logger ──
    if log_cfg.get("debug", False):
        log_dir = Path(log_cfg.get("output_dir", "results/drone_vel"))
        step_logger = StepLogger(
            output_dir=str(log_dir / "debug" / "eval"),
            console_every=log_cfg.get("log_every", 50),
            enabled=True,
        )
        controller.set_step_logger(step_logger)
        logger.info("Debug step logging ENABLED for eval")

    logger.info(
        "Eval controller: mode=%s, eval_epsilon=%.3f, "
        "Q-store GO=%d, AVOID=%d points",
        controller.mode,
        controller.config.get("eval_epsilon", 0.02),
        controller.q_store_go.next_id,
        controller.q_store_avoid.next_id,
    )

    env = PyBulletDroneEnv(
        render=config.get("render", False),
        flight_dome_size=config["flight_dome_size"],
        max_steps=max_steps,
        goal_reach_distance=ctrl_cfg["goal_reach_distance"],
        initial_altitude=config["initial_altitude"],
        ctrl_freq=config["ctrl_freq"],
        pyb_freq=config["pyb_freq"],
    )

    all_results = {}

    for stage in eval_stages:
        stage_name = stage["name"]
        episodes = stage.get("episodes", 100)

        logger.info(
            "Evaluating: %s (%d episodes)", stage_name, episodes
        )

        results = []

        for seed in eval_seeds:
            for ep in range(1, episodes + 1):
                # Snapshot counts before episode
                prev_counts = dict(controller._termination_counts)

                episode_log = run_episode(
                    env, controller, stage, ep, seed, max_steps,
                )

                result = episode_log["result"]
                results.append(result)

                status = "✅" if result == "goal_reached" else "❌"
                sr_so_far = sum(
                    1 for r in results if r == "goal_reached"
                ) / len(results)

                # Print every episode
                print(
                    f"  [{stage_name}] seed={seed} "
                    f"Ep {ep:3d}/{episodes} {status} "
                    f"{result:15s} | "
                    f"dist: {episode_log['start_distance']:.1f}"
                    f"→{episode_log['final_distance']:.2f}m | "
                    f"steps: {episode_log['total_steps']:5d} | "
                    f"reward: {episode_log['reward']:+7.1f} | "
                    f"SR={sr_so_far:.0%}"
                )

        successes = sum(1 for r in results if r == "goal_reached")
        total = len(results)
        sr = successes / max(total, 1)
        all_results[stage_name] = {
            "success_rate": round(sr, 3),
            "successes": successes,
            "total": total,
            "collisions": sum(
                1 for r in results if r == "collision"
            ),
            "timeouts": sum(
                1 for r in results if r == "timeout"
            ),
        }

        print(
            f"\n{'─'*40}\n"
            f"{stage_name}: {successes}/{total} ({sr:.0%})\n"
            f"{'─'*40}"
        )

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(
            f"  {name:30s}: SR={res['success_rate']:.0%} "
            f"({res['successes']}/{res['total']}) "
            f"col={res['collisions']} tout={res['timeouts']}"
        )

    log_dir = Path(log_cfg.get("output_dir", "results/drone_vel"))
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "eval_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    env.close()
    return all_results

# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run drone RL experiment from YAML config"
    )
    parser.add_argument(
        "config", type=str,
        nargs="?",
        default="configs/drone_vel_experiment.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values: key=value",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    run_name = config.get("logging", {}).get(
        "run_name", "experiment"
    )
    logger.info("Experiment: %s", run_name)
    logger.info("Config: %s", args.config)

    if config.get("do_generate_demos", False):
        generate_demonstrations(config)

    if config.get("do_train", False):
        train(config)

    if config.get("do_eval", False):
        evaluate(config)

    logger.info("Experiment complete: %s", run_name)


if __name__ == "__main__":
    main()
