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

"""Configuration for VelocityAviary drone (gym-pybullet-drones v2.2.0).

Action semantics:
    [dir_x, dir_y, dir_z, speed_fraction]
    speed_fraction * SPEED_LIMIT(0.25) = actual velocity m/s
"""

DEFAULT_DRONE_CONFIG_VEL = {
    # Stuck detection (efficiency-based)
    "stuck_window": 200,
    "stuck_efficiency_threshold": 0.15,
    "stuck_min_path": 0.5,
    "stuck_max_steps": 200,

    # Stuck response: reduce heuristic, increase pure random
    "stuck_random_base": 0.15,
    "stuck_random_growth": 0.001,
    "stuck_random_max": 0.5,

    # Reward shaping
    "reward_oscillation_penalty": -0.3,
    "reward_stuck_penalty": -0.5,
    "oscillation_threshold": 5,

    # ═══════════════════════════════════════════════════════
    # Curriculum levels
    # ═══════════════════════════════════════════════════════
    "curriculum": {
        "enabled": True,
        "promote_threshold": 0.7,
        "demote_threshold": 0.3,
        "min_episodes_per_level": 30,
        "trap_ratio": 0.15,
        "trap_min_level": 2,
        "levels": [
            #{
            #    "name": "L0_close_easy",
            #    "min_dist": 1.0,
            #    "max_dist": 3.0,
            #    "min_altitude": 0.5,
            #    "max_altitude": 2.5,
            #    "obstacles": False,
            #    "num_obstacles": 0,
            #},
            #{
            #    "name": "L1_medium",
            #    "min_dist": 2.0,
            #    "max_dist": 5.0,
            #    "min_altitude": 0.3,
            #    "max_altitude": 3.5,
            #    "obstacles": False,
            #    "num_obstacles": 0,
            #},
            #{
            #    "name": "L2_far",
            #    "min_dist": 4.0,
            #    "max_dist": 8.0,
            #    "min_altitude": 0.3,
            #    "max_altitude": 4.5,
            #    "obstacles": False,
            #    "num_obstacles": 0,
            #},
            #{
            #    "name": "L3_close_obstacle",
            #    "min_dist": 1.5,
            #    "max_dist": 4.0,
            #    "min_altitude": 0.5,
            #    "max_altitude": 3.0,
            #    "obstacles": True,
            #    "num_obstacles": 1,
            #},
            #{
            #   "name": "L4_close_obstacles",
            #    "min_dist": 1.5,
            #    "max_dist": 4.0,
            #    "min_altitude": 0.5,
            #    "max_altitude": 3.0,
            #    "obstacles": True,
            #    "num_obstacles": 3,
            #},
            {
                "name": "L5_far_obstacles",
                "min_dist": 3.0,
                "max_dist": 7.0,
                "min_altitude": 0.3,
                "max_altitude": 4.5,
                "obstacles": True,
                "num_obstacles": 3,
            },
        ],
    },
    
    # HER (progress-based)
    "her_virtual_goals": 3,
    "her_backup_steps": 15,
    "her_alpha": 0.3,
    "her_goal_reward": 0.5,

    # Demonstrations
    "demo_path": "demos/trap_demonstrations.npz",
    "demo_learning_rate": 0.8,
    "demo_reward_bonus": 0.5,

    # ═══════════════════════════════════════════
    # State
    # ═══════════════════════════════════════════
    "state_dim": 28,  # 14 base + 14 obstacle

    # ═══════════════════════════════════════════
    # Obstacle avoidance
    # ═══════════════════════════════════════════
    "ray_count": 8,              # number of horizontal rays
    "ray_max_dist": 3.0,         # max raycast distance [m]
    "avoid_enter_dist": 1.5,     # switch to AVOID phase [m]
    "avoid_exit_dist": 2.0,      # switch back to GO_TO_GOAL [m] (hysteresis)
    "potential_field_attract": 0.3,   # goal attraction in AVOID phase
    "potential_field_repulse": 1.0,   # obstacle repulsion strength

    # ═══════════════════════════════════════════
    # Obstacle reward
    # ═══════════════════════════════════════════
    "reward_obstacle_proximity": -2.0,   # penalty for being close to obstacle
    "reward_avoid_alignment": 1.5,       # bonus for moving along avoidance dir
    "reward_obstacle_cleared": 5.0,      # bonus for exiting AVOID phase
    "reward_progress_in_avoid": 0.2,     # weak goal progress multiplier in AVOID

    # ═══════════════════════════════════════════
    # Actions
    # ═══════════════════════════════════════════
    "num_actions": 13,
    "cruise_speed": 1.0,
    "slow_speed": 0.4,

    # ═══════════════════════════════════════════
    # Environment (VelocityAviary)
    # ═══════════════════════════════════════════
    "ctrl_freq": 48,
    "pyb_freq": 240,
    "speed_limit": 0.25,
    "flight_dome_size": 30.0,
    "initial_altitude": 1.0,
    "goal_reach_distance": 0.5,
    "max_steps_per_goal": 6000,
    "ground_z": 0.02,

    # ═══════════════════════════════════════════
    # Heuristic PD
    # ═══════════════════════════════════════════
    "pd_damping_dist": 1.0,
    "pd_damping_factor": 0.2,
    "pd_vel_damping_threshold": 0.05,
    "pd_vel_damping_multiplier": 0.5,

    # ═══════════════════════════════════════════
    # Reward
    # ═══════════════════════════════════════════
    "reward_progress": 5.0,
    "reward_goal_reached": 50.0,
    "reward_step_penalty": -0.05,
    "reward_collision": -20.0,
    "reward_out_of_bounds": -10.0,
    "reward_timeout": -5.0,
    "reward_alignment": 1.0,
    "reward_excessive_speed": -0.3,
    "reward_hover_bonus": 0.3,
    "reward_altitude_penalty": -1.0,

    # ═══════════════════════════════════════════
    # Q-learning (HNSW store)
    # ═══════════════════════════════════════════
    "gamma": 0.95,
    "alpha": 0.1,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,

    "max_points": 100000,
    "k_neighbors": 7,
    "sigma": 1.0,
    "insert_threshold": 0.5,
    "adaptive_sigma": True,
    "norm_warmup_steps": 3000,
    "feature_weights": None,
    "evict_fraction": 0.1,
    "auto_calibrate": False,
    "calibration_percentile": 10.0,
    "min_calibration_samples": 100,
    "min_weight_threshold": 0.01,
    "rebuild_on_freeze": True,
    "norm_min_std": 1e-4,

    # ═══════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════
    "num_episodes": 2000,
    "warmup_episodes": 100,
    "mode": "train",
    "eval_epsilon": 0.02,
    "eval_alpha_multiplier": 0.1,

    "success_backup_enabled": True,
    "success_backup_steps": 3000,
    "success_backup_lambda": 0.9,
    "success_backup_alpha_multiplier": 0.5,

    # ═══════════════════════════════════════════
    # Heuristic strengths
    # ═══════════════════════════════════════════
    "heuristic_approach_strength": 8.0,
    "heuristic_stabilize_strength": 3.0,
    "heuristic_altitude_strength": 5.0,
    "heuristic_avoid_strength": 10.0,
}
