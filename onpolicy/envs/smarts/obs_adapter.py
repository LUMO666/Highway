import numpy as np
import math

from smarts.core.utils.math import vec_2d

def observation_adapter(neighbor_num, use_proximity):
    def _lane_ttc_observation_adapter(env_observation):

        ego = env_observation.ego_vehicle_state
        waypoint_paths = env_observation.waypoint_paths
        wps = [path[0] for path in waypoint_paths]

        # distance of vehicle from center of lane
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth

        ego_ttc, ego_lane_dist = _ego_ttc_lane_dist(env_observation, closest_wp.lane_index)

        if env_observation.neighborhood_vehicle_states is not None:
            neighbor = cal_neighbor(env_observation, neighbor_num)
        else:
            neighbor = [0] * (neighbor_num * 5)

        if use_proximity:
            if env_observation.occupancy_grid_map is not None:
                proximity = cal_proximity(env_observation)
            else:
                proximity = [0] * 8

        if use_proximity:
            return {
                "distance_to_center": np.array([norm_dist_from_center]),
                "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
                "speed": np.array([ego.speed]),
                "steering": np.array([ego.steering]),
                "ego_ttc": np.array(ego_ttc),
                "ego_lane_dist": np.array(ego_lane_dist),
                "neighbor": np.array(neighbor),
                "proximity": np.array(proximity)
            }
        else:
            return {
                "distance_to_center": np.array([norm_dist_from_center]),
                "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
                "speed": np.array([ego.speed]),
                "steering": np.array([ego.steering]),
                "ego_ttc": np.array(ego_ttc),
                "ego_lane_dist": np.array(ego_lane_dist),
                "neighbor": np.array(neighbor),
            }

    return _lane_ttc_observation_adapter

def _ego_ttc_lane_dist(env_observation, ego_lane_index):
    ttc_by_p, lane_dist_by_p = _ttc_by_path(env_observation)

    return _ego_ttc_calc(ego_lane_index, ttc_by_p, lane_dist_by_p)

def _cal_angle(vec):

    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle

def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)

    return groups

def cal_neighbor(env_obs, closest_neighbor_num):
    ego = env_obs.ego_vehicle_state
    neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
    # dist, speed, ttc, pos
    features = np.zeros((closest_neighbor_num, 5))
    # fill neighbor vehicles into closest_neighboor_num areas
    surrounding_vehicles = _get_closest_vehicles(
        ego, neighbor_vehicle_states, n=closest_neighbor_num
    )

    heading_angle = ego.heading + math.pi / 2.0
    ego_heading_vec = np.asarray([math.cos(heading_angle), math.sin(heading_angle)])
    for i, v in surrounding_vehicles.items():
        if v[0] is None:
            continue
        v = v[0]
        rel_pos = np.asarray(
            list(map(lambda x: x[0] - x[1], zip(v.position[:2], ego.position[:2])))
        )

        rel_dist = np.sqrt(rel_pos.dot(rel_pos))

        v_heading_angle = math.radians(v.heading)
        v_heading_vec = np.asarray(
            [math.cos(v_heading_angle), math.sin(v_heading_angle)]
        )

        ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
        rel_pos_norm_2 = rel_pos.dot(rel_pos)
        v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)

        ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
            ego_heading_norm_2 + rel_pos_norm_2
        )

        v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
            v_heading_norm_2 + rel_pos_norm_2
        )

        rel_speed = 0
        if ego_cosin <= 0 and v_cosin > 0:
            rel_speed = 0
        else:
            rel_speed = ego.speed * ego_cosin - v.speed * v_cosin

        ttc = min(rel_dist / max(1e-5, rel_speed), 1e3)

        features[i, :] = np.asarray(
            [rel_dist, rel_speed, ttc, rel_pos[0], rel_pos[1]]
        )

    return features.reshape((-1,))

def _ttc_by_path(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    neighborhood_vehicle_states = env_observation.neighborhood_vehicle_states

    # first sum up the distance between waypoints along a path
    # ie. [(wp1, path1, 0),
    #      (wp2, path1, 0 + dist(wp1, wp2)),
    #      (wp3, path1, 0 + dist(wp1, wp2) + dist(wp2, wp3))]

    wps_with_lane_dist = []
    for path_idx, path in enumerate(waypoint_paths):
        lane_dist = 0.0
        for w1, w2 in zip(path, path[1:]):
            wps_with_lane_dist.append((w1, path_idx, lane_dist))
            lane_dist += np.linalg.norm(w2.pos - w1.pos)
        wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

    # next we compute the TTC along each of the paths
    ttc_by_path_index = [1000] * len(waypoint_paths)
    lane_dist_by_path_index = [1] * len(waypoint_paths)
    if neighborhood_vehicle_states is not None:
        for v in neighborhood_vehicle_states:
            # find all waypoints that are on the same lane as this vehicle
            wps_on_lane = [
                (wp, path_idx, dist)
                for wp, path_idx, dist in wps_with_lane_dist
                if wp.lane_id == v.lane_id
            ]

            if not wps_on_lane:
                # this vehicle is not on a nearby lane
                continue

            # find the closest waypoint on this lane to this vehicle
            nearest_wp, path_idx, lane_dist = min(
                wps_on_lane, key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position))
            )

            if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                # this vehicle is not close enough to the path, this can happen
                # if the vehicle is behind the ego, or ahead past the end of
                # the waypoints
                continue

            relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
            if abs(relative_speed_m_per_s) < 1e-5:
                relative_speed_m_per_s = 1e-5

            ttc = lane_dist / relative_speed_m_per_s
            ttc /= 10
            if ttc <= 0:
                # discard collisions that would have happened in the past
                continue

            lane_dist /= 100
            lane_dist_by_path_index[path_idx] = min(
                lane_dist_by_path_index[path_idx], lane_dist
            )
            ttc_by_path_index[path_idx] = min(ttc_by_path_index[path_idx], ttc)

    return ttc_by_path_index, lane_dist_by_path_index

def _ego_ttc_calc(ego_lane_index, ttc_by_path, lane_dist_by_path):
    ego_ttc = [0] * 3
    ego_lane_dist = [0] * 3

    ego_ttc[1] = ttc_by_path[ego_lane_index]
    ego_lane_dist[1] = lane_dist_by_path[ego_lane_index]

    max_lane_index = len(ttc_by_path) - 1
    min_lane_index = 0
    if ego_lane_index + 1 > max_lane_index:
        ego_ttc[2] = 0
        ego_lane_dist[2] = 0
    else:
        ego_ttc[2] = ttc_by_path[ego_lane_index + 1]
        ego_lane_dist[2] = lane_dist_by_path[ego_lane_index + 1]
    if ego_lane_index - 1 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
    else:
        ego_ttc[0] = ttc_by_path[ego_lane_index - 1]
        ego_lane_dist[0] = lane_dist_by_path[ego_lane_index - 1]
    return ego_ttc, ego_lane_dist

def proximity_detection(OGM):
    """
    Detects other vehicles in the vicinity of the ego vehicle
    hard coded for OGM(64, 64, 0.25)
    """
    boxes = []
    boxes += [
        OGM[11:25, 23:27],  # front left
        OGM[11:25, 27:37],  # front center
        OGM[11:25, 37:41],  # front right
        OGM[25:39, 23:27],  # left
        OGM[25:39, 37:41],  # right
        OGM[41:53, 23:27],  # back left
        OGM[41:53, 27:37],  # back center
        OGM[41:53, 37:41],  # back right
    ]
    output = np.array([b.max() > 0 for b in boxes], np.float32)
    return output

def cal_proximity(env_obs):
    proximity = proximity_detection(env_obs.occupancy_grid_map[1])
    return proximity

