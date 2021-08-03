import numpy as np
import logging
from icecream import ic

from typing import List, Tuple, Union

import copy
from onpolicy.envs.highway.highway_env import utils
from onpolicy.envs.highway.highway_env.road.road import Road, LaneIndex, Route
from onpolicy.envs.highway.highway_env.envs.highway_env import HighwayEnv
from onpolicy.envs.highway.highway_env.types import Vector
from onpolicy.envs.highway.highway_env.vehicle.kinematics import Vehicle
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle
from onpolicy.envs.highway.highway_env.road.objects import RoadObject

from ..common.abstract import AbstractAgent

logger = logging.getLogger(__name__)



class IDMAgent():
    def __init__(self,
                 env,
                 dt = 1, #every 5 step check change lane
                 vehicle_id = 0,
                 target_lane_index: LaneIndex = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):

        self.vehicle_id = vehicle_id
        self.vehicle = env.road.vehicles[vehicle_id]
        self.dt = dt
        #for force change lane
        self.DISTANCE_CHANGE = 10.0 + ControlledVehicle.LENGTH

        self.target_lane_index = target_lane_index or self.vehicle.lane_index
        self.route = route

        # initialization for IDMVehicle
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.vehicle.position) * np.pi) % self.LANE_CHANGE_DELAY

    SPEED_COUNT: int = 3  # []
    SPEED_MIN: float = 25  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    ########### Parameter of IDMVehicle
    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]


    def act(self,env):
        # Longitudinal: IDM
        self.vehicle = env.road.vehicles[self.vehicle_id]
        ######### TODO: Where the hell did this target come from and keep changing???
        self.vehicle.target_speed = 23.5
        #print("veh:",env.road.vehicles[self.vehicle_id])
        #print("self.veh:",self.vehicle)
        action = {}
        front_vehicle, rear_vehicle = self.vehicle.road.neighbour_vehicles(self.vehicle)
        action['acceleration'] = self.acceleration(ego_vehicle=self.vehicle,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        #print("self.veh:",self.vehicle)
        #action['acceleration'] = self.acceleration(ego_vehicle=self.vehicle,
        #                                           front_vehicle=None,
        #                                           rear_vehicle=None)
        #print("return:acc",action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        if action['acceleration']>2:
            IDMaction = 3
        elif action['acceleration']<-2:
            IDMaction = 4
        else:
            
            IDMaction = 1
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
            if self.vehicle.target_lane_index[2]<self.vehicle.lane_index[2]:
                IDMaction = 0
            elif self.vehicle.target_lane_index[2]>self.vehicle.lane_index[2]:
                IDMaction = 2
        self.timer += self.dt

        # Force a bad change lane rule
        #front_vehicle, rear_vehicle = self.vehicle.road.neighbour_vehicles(self.vehicle)
        if rear_vehicle:    
            if rear_vehicle.front_distance_to(self.vehicle) < self.DISTANCE_CHANGE:
            #if self.vehicle.front_distance_to(front_vehicle) < self.DISTANCE_CHANGE:
                #print("############# FORCE LANE CHANGE ############")
                #force side lane crash
                if self.vehicle.lane_index[2] == 0:
                    IDMaction = 0
                    self.vehicle.crashed = True
                elif self.vehicle.lane_index[2] == 3:
                    IDMaction = 2
                    self.vehicle.crashed = True
                else:
                    '''
                    for lane_index in self.vehicle.road.network.side_lanes(self.vehicle.lane_index):
                        if self.vehicle.road.network.get_lane(lane_index).is_reachable_from(self.vehicle.position):
                            self.target_lane_index = lane_index
                    if self.vehicle.target_lane_index[2]<self.vehicle.lane_index[2]:
                        IDMaction = 0
                    elif self.vehicle.target_lane_index[2]>self.vehicle.lane_index[2]:
                        IDMaction = 2
                        '''
                    '''
                    flaw_list = [0,2]
                    IDMaction = flaw_list[np.random.randint(0,2)]
                    '''

        #print("act:",IDMaction)

        return IDMaction
        #return 0

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.vehicle.road.network.get_lane(self.vehicle.target_lane_index).after_end(self.vehicle.position):
            self.vehicle.target_lane_index = self.vehicle.road.network.next_lane(self.vehicle.target_lane_index,
                                                                 route=self.vehicle.route,
                                                                 position=self.vehicle.position,
                                                                 np_random=self.vehicle.road.np_random)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.
        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.
        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", 0))
        #ic(ego_target_speed)
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        #ic(ego_vehicle.speed)
        #ic(front_vehicle)
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.
        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.
        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.vehicle.lane_index != self.vehicle.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.vehicle.lane_index[:2] == self.vehicle.target_lane_index[:2]:
                for v in self.vehicle.road.vehicles:
                    if v is not self.vehicle \
                            and v.lane_index != self.vehicle.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.vehicle.target_lane_index:
                        d = self.vehicle.lane_distance_to(v)
                        d_star = self.desired_gap(self.vehicle, v)
                        if 0 < d < d_star:
                            self.vehicle.target_lane_index = self.vehicle.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.vehicle.road.network.side_lanes(self.vehicle.lane_index):
            # Is the candidate lane close enough?
            if not self.vehicle.road.network.get_lane(lane_index).is_reachable_from(self.vehicle.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.vehicle.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change
            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.
        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.vehicle.road.neighbour_vehicles(self.vehicle, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self.vehicle)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.vehicle.road.neighbour_vehicles(self.vehicle)
        self_pred_a = self.acceleration(ego_vehicle=self.vehicle, front_vehicle=new_preceding)
        if self.vehicle.route and self.vehicle.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.vehicle.target_lane_index[2]) != np.sign(
                    self.vehicle.route[0][2] - self.vehicle.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self.vehicle, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self.vehicle)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.
        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.vehicle.target_lane_index != self.vehicle.lane_index and self.vehicle.speed < stopped_speed:
            _, rear = self.vehicle.road.neighbour_vehicles(self.vehicle)
            _, new_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.road.network.get_lane(self.vehicle.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self.vehicle) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self.vehicle) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


    @staticmethod
    def is_finite_mdp(env):
        try:
            finite_mdp = __import__("finite_mdp.envs.finite_mdp_env")
            if isinstance(env, finite_mdp.envs.finite_mdp_env.FiniteMDPEnv):
                return True
        except (ModuleNotFoundError, TypeError):
            return False

    def plan_trajectory(self, state, horizon=10):
        action_value = self.get_state_action_value()
        states, actions = [], []
        for _ in range(horizon):
            action = np.argmax(action_value[state])
            states.append(state)
            actions.append(action)
            state = self.mdp.next_state(state, action)
            if self.mdp.terminal[state]:
                states.append(state)
                actions.append(None)
                break
        return states, actions

    def record(self, state, action, reward, next_state, done, info):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False
