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

"""Generate scripted demonstrations for trap scenarios.

Creates .npz files with (state, action, reward) tuples
that show how to escape trap scenarios.

Usage:
    python generate_demonstrations.py
    python generate_demonstrations.py --output demos/
    python generate_demonstrations.py --render
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from drone.pybullet_drone_env import PyBulletDroneEnv
from drone.drone_action_space_vel import DroneActionSpaceVel
from drone.drone_config_vel import DEFAULT_DRONE_CONFIG_VEL
from drone.curriculum import TRAP_SCENARIOS
from drone.drone_controller_vel import DroneControllerVel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_demos")


def fly_waypoints(
    env: PyBulletDroneEnv,
    controller: DroneControllerVel,
    waypoints: list[np.ndarray],
    reach_dist: float = 0.5,
    max_steps_per_wp: int = 500,
) -> list[dict]:
    """Fly through waypoints using simple direction matching.

    Returns list of (state, action, reward) transitions.
    """
    AS = DroneActionSpaceVel
    transitions = []

    for wp_idx, waypoint in enumerate(waypoints):
        for step in range(max_steps_per_wp):
            sensor_data = env.get_sensor_data()
            pos = np.array(sensor_data["position"])

            # Direction to waypoint
            delta = waypoint - pos
            dist = np.linalg.norm(delta)
            if dist < reach_dist:
                break

            direction = delta / dist

            # Pick best discrete action toward waypoint
            action_idx = _direction_to_action(direction)

            # Record state before action
            state = controller._compute_state(sensor_data)

            # Execute
            sensor_data = env.step_discrete(
                action_idx, controller.action_space
            )

            # Compute reward
            new_pos = np.array(sensor_data["position"])
            new_dist = np.linalg.norm(waypoint - new_pos)
            progress = dist - new_dist
            reward = progress * 5.0  # strong progress signal

            transitions.append({
                "state": state,
                "action": action_idx,
                "reward": reward,
            })

            if sensor_data.get("collision"):
                break

    return transitions


def _direction_to_action(direction: np.ndarray) -> int:
    """Map 3D direction to best discrete action index."""
    AS = DroneActionSpaceVel
    dx, dy, dz = direction

    # Prioritize horizontal movement
    best_action = AS.IDX_FORWARD
    best_score = -999.0

    candidates = [
        (AS.IDX_FORWARD, np.array([1, 0, 0])),
        (AS.IDX_BACKWARD, np.array([-1, 0, 0])),
        (AS.IDX_LEFT, np.array([0, -1, 0])),
        (AS.IDX_RIGHT, np.array([0, 1, 0])),
        (AS.IDX_UP, np.array([0, 0, 1])),
        (AS.IDX_DOWN, np.array([0, 0, -1])),
        (AS.IDX_FORWARD_LEFT, np.array([0.707, -0.707, 0])),
        (AS.IDX_FORWARD_RIGHT, np.array([0.707, 0.707, 0])),
        (AS.IDX_FORWARD_UP, np.array([0.707, 0, 0.707])),
        (AS.IDX_FORWARD_DOWN, np.array([0.707, 0, -0.707])),
    ]

    for idx, action_dir in candidates:
        score = np.dot(direction, action_dir)
        if score > best_score:
            best_score = score
            best_action = idx

    return best_action


def generate_trap_demo(
    env: PyBulletDroneEnv,
    controller: DroneControllerVel,
    trap: dict,
    start_pos: np.ndarray,
) -> list[dict]:
    """Generate demonstration for one trap scenario.

    Creates waypoints that navigate around the trap,
    then flies them.
    """
    obstacles = []
    for obs in trap["obstacles"]:
        pos = np.array(obs["position"], dtype=float) + start_pos
        obstacles.append({
            "position": pos.tolist(),
            "radius": obs["radius"],
            "height": obs["height"],
        })

    goal_offset = np.array(trap["goal_offset"], dtype=float)
    goal_pos = start_pos + goal_offset

    # Set up environment
    sensor_data = env.reset()
    env.spawn_obstacles(obstacles)
    goal_pose = np.concatenate([goal_pos, np.zeros(3)])
    env.set_goal(goal_pose)

    # Create waypoints to navigate around trap
    waypoints = _create_escape_waypoints(
        start_pos, goal_pos, trap
    )

    controller.set_new_goal(goal_pose, start_pos)

    transitions = fly_waypoints(
        env, controller, waypoints,
    )

    return transitions


def _create_escape_waypoints(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    trap: dict,
) -> list[np.ndarray]:
    """Create waypoints that navigate around a trap.

    Strategy: back up, go sideways, then approach goal.
    """
    name = trap["name"]
    s = start_pos.copy()

    if name == "U_wall_open_back":
        return [
            s + np.array([-2.0, 0.0, 0.0]),   # back out
            s + np.array([-2.0, 3.0, 0.0]),   # go sideways
            s + np.array([3.0, 3.0, 0.0]),    # go past wall
            goal_pos,                           # approach goal
        ]
    elif name == "corridor_dead_end":
        return [
            s + np.array([-1.0, 0.0, 0.0]),   # back up
            s + np.array([-1.0, 2.0, 0.0]),   # exit sideways
            s + np.array([3.0, 2.0, 0.0]),    # go around
            goal_pos,
        ]
    elif name == "L_corner":
        return [
            s + np.array([0.0, -2.0, 0.0]),   # go opposite side
            s + np.array([4.0, -2.0, 0.0]),   # go past corner
            goal_pos,
        ]
    else:
        # Generic: go up and over
        mid = (s + goal_pos) / 2
        mid[2] += 3.0
        return [mid, goal_pos]


def main():
    parser = argparse.ArgumentParser(
        description="Generate trap demonstrations"
    )
    parser.add_argument(
        "--output", default="demos",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Show GUI during generation",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Repetitions per trap scenario",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_DRONE_CONFIG_VEL.copy()
    config["mode"] = "eval"

    env = PyBulletDroneEnv(
        render=args.render,
        flight_dome_size=config["flight_dome_size"],
        max_steps=config["max_steps_per_goal"],
        goal_reach_distance=config["goal_reach_distance"],
        initial_altitude=config["initial_altitude"],
        ctrl_freq=config["ctrl_freq"],
        pyb_freq=config["pyb_freq"],
    )

    controller = DroneControllerVel(
        agent_id="demo_drone", config=config,
    )

    start_pos = np.array([0.0, 0.0, 1.0])

    all_states = []
    all_actions = []
    all_rewards = []

    for trap in TRAP_SCENARIOS:
        for rep in range(args.repeats):
            logger.info(
                "Generating: %s (rep %d/%d)",
                trap["name"], rep + 1, args.repeats,
            )

            # Slight variation in start position
            rng = np.random.RandomState(42 + rep)
            offset = rng.uniform(-0.3, 0.3, size=3)
            offset[2] = 0
            varied_start = start_pos + offset

            transitions = generate_trap_demo(
                env, controller, trap, varied_start,
            )

            if transitions:
                for tr in transitions:
                    all_states.append(tr["state"])
                    all_actions.append(tr["action"])
                    all_rewards.append(tr["reward"])

                logger.info(
                    "  → %d transitions recorded",
                    len(transitions),
                )
            else:
                logger.warning(
                    "  → No transitions (collision?)"
                )

    # Save combined demo file
    if all_states:
        demo_path = output_dir / "trap_demonstrations.npz"
        np.savez(
            str(demo_path),
            states=np.array(all_states),
            actions=np.array(all_actions),
            rewards=np.array(all_rewards),
        )
        logger.info(
            "Saved %d total transitions to %s",
            len(all_states), demo_path,
        )
    else:
        logger.warning("No transitions generated!")

    env.close()


if __name__ == "__main__":
    main()
