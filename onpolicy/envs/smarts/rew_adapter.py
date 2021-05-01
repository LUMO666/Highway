import numpy as np
from scipy.spatial import distance

from .obs_adapter import observation_adapter

def reward_adapter(adapter_type="vanilla", neighbor_num=3):
    def vanilla(env_obs, env_reward):
        return env_reward

    def standard(last_env_obs, env_obs, env_reward):
        penalty, bonus = 0.0, 0.0
        _lane_ttc_observation_adapter = observation_adapter(neighbor_num, False)
        obs = _lane_ttc_observation_adapter(env_obs)
        last_obs = _lane_ttc_observation_adapter(last_env_obs)

        neighbor_features = obs.get("neighbor", None)
        last_neighbor_feature = last_obs.get("neighbor", None)

        # dealing with neighbor_features
        if neighbor_features is not None:
            new_neighbor_feature = neighbor_features.reshape((-1, 5))
            last_neighbor_feature = last_neighbor_feature.reshape((-1, 5))
            mean_dist = np.mean(new_neighbor_feature[:, 0])
            mean_ttc = np.mean(new_neighbor_feature[:, 2])
            mean_dist2 = np.mean(last_neighbor_feature[:, 0])
            # mean_speed2 = np.mean(last_neighbor_feature[:, 1])
            mean_ttc2 = np.mean(last_neighbor_feature[:, 2])

            # this penalty should considering the speed
            # if speed is ...
            ego_speed = env_obs.ego_vehicle_state.speed
            penalty += np.tanh(ego_speed) * (
                0.03 * (mean_dist - mean_dist2)
                # - 0.01 * (mean_speed - mean_speed2)
                + 0.01 * (mean_ttc - mean_ttc2)
            )

        # ======== Penalty: distance to goal =========
        goal = env_obs.ego_vehicle_state.mission.goal

        last_ego_2d_pos = last_env_obs.ego_vehicle_state.position[:2]
        ego_2d_pos = env_obs.ego_vehicle_state.position[:2]

        if hasattr(goal, "position"):
            goal_pos = goal.position
            last_goal_dist = distance.euclidean(last_ego_2d_pos, goal_pos)
            goal_dist = distance.euclidean(ego_2d_pos, goal_pos)
            penalty += 0.1 * (last_goal_dist - goal_dist)
        else:
            raise ValueError(f"Goal type: {type(goal)} has no attr named: position.")

        # ======== Penalty: distance to the center
        if last_obs.get("distance_to_center") is not None:
            diff_dist_to_center_penalty = np.abs(
                last_obs["distance_to_center"]
            ) - np.abs(obs["distance_to_center"])
            penalty += 0.01 * diff_dist_to_center_penalty[0]

        # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
        ego_events = env_obs.events
        # ::collision
        penalty += -50.0 if len(ego_events.collisions) > 0 else 0.0
        # ::off road
        penalty += -50.0 if ego_events.off_road else 0.0
        # ::reach goal
        if ego_events.reached_goal:
            bonus += 20.0

        # ::reached max_episode_step
        if ego_events.reached_max_episode_steps:
            penalty += -0.5
        else:
            bonus += 0.5

        # ======== Penalty: penalise sharp turns done at high speeds =======
        if env_obs.ego_vehicle_state.speed > 60:
            steering_penalty = -pow(
                (env_obs.ego_vehicle_state.speed - 60)
                / 20
                * env_obs.ego_vehicle_state.steering
                / 4,
                2,
            )
        else:
            steering_penalty = 0
        penalty += 0.1 * steering_penalty

        # ========= Bonus: environment reward (distance travelled) ==========
        bonus += 0.05 * env_reward
        return bonus + penalty

    def cruising(env_obs, env_reward):
        global lane_crash_flag
        global intersection_crash_flag

        distance_from_center = get_distance_from_center(env_obs)

        center_penalty = -np.abs(distance_from_center)

        # penalise sharp turns done at high speeds
        if env_obs.ego_vehicle_state.speed * 3.6 > 60:
            steering_penalty = -pow(
                (env_obs.ego_vehicle_state.speed * 3.6 - 60)
                / 20
                * (env_obs.ego_vehicle_state.steering)
                * 45
                / 4,
                2,
            )
        else:
            steering_penalty = 0

        # penalise close proximity to lane cars
        if lane_crash_flag:
            crash_penalty = -5
        else:
            crash_penalty = 0

        # penalise close proximity to intersection cars
        if intersection_crash_flag:
            crash_penalty -= 5

        total_reward = np.sum([1.0 * env_reward])
        total_penalty = np.sum(
            [0.1 * center_penalty, 1 * steering_penalty, 1 * crash_penalty]
        )

        return (total_reward + total_penalty) / 200.0

    return {
        "vanilla": vanilla,
        "standard": standard,
        "cruising": cruising,
    }[adapter_type]

def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center