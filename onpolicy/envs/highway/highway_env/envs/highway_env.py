import numpy as np
from typing import Tuple
from gym.envs.registration import register
import copy

from onpolicy.envs.highway.highway_env import utils
from onpolicy.envs.highway.highway_env.envs.common.abstract import AbstractEnv
from onpolicy.envs.highway.highway_env.envs.common.action import Action
from onpolicy.envs.highway.highway_env.road.road import Road, RoadNetwork
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle

import random

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.9
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""

    # class learning rank


    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self, dif) -> None:

        ######## Jianming Jan 29 New feature -> bubble test: control handover
        self.use_bubble = False
        self.max_bubble_length = self.config["bubble_length"]   ####max_length_bubble
        ########
        self._create_road()
        self._create_vehicles()
        self.acquire_attacker(True, diff=dif)


    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # the number of agent with initialized postions overlapping with each other
        self.controlled_vehicles = []
        number_overlap = 0
        for i in range(self.config["controlled_vehicles"]):

        #for i in range(self.config["n_defenders"]):
            # vehicle = self.action_type.vehicle_class.create_random(self.road,
            #                                                        speed=25,
            #                                                        lane_id=self.config["initial_lane_id"],
            #                                                        spacing=self.config["ego_spacing"],
            #                                                        )
            default_spacing = 12.5 # 0.5 * speed
            longitude_position = 40+5*np.random.randint(1)
            initial_lane_idx = random.choice( [4*i for i in range(self.config["lanes_count"])] )
            # To separate cars in different places to avoid collision
            for vehicle_ in self.controlled_vehicles:
                if abs(longitude_position - vehicle_.position[0]) < 5 and initial_lane_idx == 4*vehicle_.lane_index[2]:
                    longitude_position = longitude_position - (number_overlap+1)*default_spacing
                    number_overlap = number_overlap + 1

            vehicle = self.action_type.vehicle_class(road=self.road, position=[longitude_position, initial_lane_idx], heading=0, speed=25)

            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["npc_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            vehicle = vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
            self.road.vehicles.append(vehicle)
            #self.controlled_vehicles.append(vehicle)
            # observation size depents on the firstly controlled_vehicles in the initilization process
            # but after defined the observation type doesn't change.


    def enter_bubble(self):
        if self.use_bubble:
            if self.steps % self.max_bubble_length==0:
                self.acquire_attacker()

    def bubble_terminated(self):
        if self.steps % self.max_bubble_length == 0:
            return True
        else:
            return False

    def acquire_attacker(self,reset=False,diff=0):
        if reset:
            for i,v in enumerate(self.controlled_vehicles):
                if i < self.config["n_defenders"]:
                    continue
                else:
                    self.road.vehicles.remove(v)

        #### only one defender
        defender_pos=self.controlled_vehicles[0].position
        dis=[]
        # caculate distance to defender
        for i , v in enumerate(self.road.vehicles):
            if i < self.config["n_defenders"]:
                continue
            else:
                if (v.position - defender_pos)[0] <0:
                    dis.append(10000)
                else:
                    dis.append(np.linalg.norm(v.position - defender_pos))
        # sort and take vehicles most close to defender as npc in bubble
        from copy import deepcopy
        dis_sort=deepcopy(dis)
        dis_sort.sort()
        ins=[]
        ins_npc=[]
        npc_bubble=self.config["available_npc_bubble"]
        self.npcs_in_bubble=[]

        dis_in_bubble=deepcopy(dis_sort[:self.config["n_attackers"] + npc_bubble])
        #attacker_dis=random.sample(dis_in_bubble,self.config["n_attackers"])
        attacker_dis = dis_in_bubble[:self.config["n_attackers"]]
        npc_dis=[x for x in dis_in_bubble if x not in attacker_dis]

        # ins for attackers, ins_npc for npc in bubble
        for i in range(self.config["n_attackers"]):
            ins.append(dis.index(attacker_dis[i]))

        for i in range(npc_bubble):
            ins_npc.append(dis.index(npc_dis[i]))

        ##we must first define the attacker
        for i,index in enumerate(ins):
            if reset:
                self.road.vehicles[self.config["n_defenders"] + index].use_action_level_behavior = True
                self.controlled_vehicles[i+self.config["n_defenders"]]=self.road.vehicles[self.config["n_defenders"] + index]
            else:
                self.controlled_vehicles[i + self.config["n_defenders"]].use_action_level_behavior=False
                self.road.vehicles[self.config["n_defenders"] + index].use_action_level_behavior = True
                self.controlled_vehicles[i+self.config["n_defenders"]]=self.road.vehicles[self.config["n_defenders"]+index]
            #for render, exchange the index in the list
            self.road.vehicles[self.config["n_defenders"] + index],self.road.vehicles[self.config["n_defenders"] + i]=\
            self.road.vehicles[self.config["n_defenders"] + i],self.road.vehicles[self.config["n_defenders"] + index]

        for i,index in enumerate(ins_npc):
            if reset:
                self.road.vehicles[self.config["n_defenders"] + index].use_action_level_behavior = True
                self.controlled_vehicles[i + self.config["n_defenders"]+self.config["n_attackers"]] = self.road.vehicles[
                    self.config["n_defenders"] + index]
            else:
                self.controlled_vehicles[i + self.config["n_defenders"] + self.config["n_attackers"]].use_action_level_behavior=False
                self.road.vehicles[self.config["n_defenders"] + index].use_action_level_behavior = True
                self.controlled_vehicles[i + self.config["n_defenders"] + self.config["n_attackers"]] = self.road.vehicles[
                    self.config["n_defenders"] + index]

            self.npcs_in_bubble.append(self.road.vehicles[self.config["n_defenders"] + index])

            self.road.vehicles[self.config["n_defenders"] + index], self.road.vehicles[self.config["n_defenders"]+self.config["n_attackers"] + i] = \
            self.road.vehicles[self.config["n_defenders"]+self.config["n_attackers"] + i], self.road.vehicles[self.config["n_defenders"] + index]
        # should add reset distance for classlearning
        # Rank 0: defender set on side lane and attacker set behind defender on same lane for a fixed distance, level 0-3 5 11 20 30
        # Rank 1: defender set on side lane and attacker set on near lane of defender for a fixed distance level 0-3 5 11 20 30
        # Rank 2: defender set on random lane and attcker set behind defender on random lane but not on same lane for a fixed distance, level 0-3 5 11 20 30
        # Rank 3: no special reset
        if reset:
            if diff > 1:
                diff = 1
            #distance = 10 + 20*diff
            #test lane change
            distance = 10
            set_attacker_id = np.random.randint(self.config["n_attackers"])
            self.controlled_vehicles[0].position[1] = random.choice([0,1,2,3])
            #print("att_id:",set_attacker_id)
            
            self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position = copy.deepcopy(self.controlled_vehicles[0].position)
            self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[0] -= distance
            #print(self.controlled_vehicles[0].position)
            #print( self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position)
            #print("att:",self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position)
            #print("def:",self.controlled_vehicles[0].position)
            #print("dif:",diff)

        '''
        if reset: #position = [longitude pos, lane_index]
            distance_list = [5,11,20,30]
            if self.class_rank == 0:
                set_attacker_id = np.random.randint(self.config["n_attackers"])
                self.controlled_vehicles[0].position[1] = random.choice([0,3])
                self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position = self.controlled_vehicles[0].position
                self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[0] -= distance_list[self.class_level]
            elif self.class_rank == 1:
                set_attacker_id = np.random.randint(self.config["n_attackers"])
                self.controlled_vehicles[0].position[1] = random.choice([0,3])
                self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position = self.controlled_vehicles[0].position
                self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[0] -= distance_list[self.class_level]
                if self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[1] == 0:
                    self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[1] = 1
                elif self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[1] == 3:
                    self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[1] = 2
                else:
                    raise NotImplementedError
            elif self.class_rank == 2:
                set_attacker_id = np.random.randint(self.config["n_attackers"])
                self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position = self.controlled_vehicles[0].position
                self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[0] -= distance_list[self.class_level]
                while self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[1] == self.controlled_vehicles[0].position[1]:
                    self.controlled_vehicles[self.config["n_defenders"] + set_attacker_id].position[1] = np.random.randint(4)
            elif self.calss_rank == 3:
                pass
            else:
                raise NotImplementedError
        '''
        self.define_spaces()

    def _reward(self, action: Action) :#-> float: now we return a list
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """

        # -> float: now we change it to return a list!!!!!

        rewards=[]
        for i,vehicle in enumerate(self.controlled_vehicles):
        
            neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
            lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
                else vehicle.lane_index[2]
            scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
            reward = \
                 self.config["collision_reward"] * vehicle.crashed \
                + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
                + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
            #reward = utils.lmap(reward,
            #              [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
            #              [0, 1])
            #reward = 0 if not vehicle.on_road else reward
            reward = -1 if not vehicle.on_road else reward

            if self.config['task_type']=='attack':
                if i>=self.config['n_defenders'] and i <(self.config['n_defenders']+self.config['n_attackers']):
                    reward*=0
                    reward = -0.5 if not vehicle.on_road or vehicle.crashed else 0

            rewards.append(reward)

        return rewards

    def _is_terminal(self) :
        ### every agent done
        #now we change it to return a list
        dones = []
        for vehicle in self.controlled_vehicles:
            dones.append(vehicle.crashed or \
                         self.steps >= self.config["duration"] or \
                         (self.config["offroad_terminal"] and not vehicle.on_road))

        defender_done = dones[:self.config["n_defenders"]]
        attacker_done = dones[self.config["n_defenders"]:self.config["n_defenders"] + self.config["n_attackers"]]

        if np.all(defender_done):
            for i in range(len(dones)):
                dones[i] = True
        elif len(attacker_done) > 0 and np.all(attacker_done):
            for i in range(len(dones)):
                dones[i] = True
        return dones

    def adv_rew(self) :#-> bool:
        dones = []
        for vehicle in self.controlled_vehicles:
            dones.append(vehicle.crashed or \
                         (self.config["offroad_terminal"] and not vehicle.on_road))
        defender_done = dones[:self.config["n_defenders"]]#
        attacker_done = dones[self.config["n_defenders"]:self.config["n_defenders"] + self.config["n_attackers"]]

        ###check!
        if np.all(defender_done) and (not np.any(attacker_done)):
            ad_rew=2
        elif np.all(defender_done):
            ad_rew = 0.6
        #elif np.any(defender_done):
        #    ad_rew = 0.2
        else:
            ad_rew = 0

        return ad_rew

    def _is_done(self) :#-> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        ####env done!
        dones = []
        for vehicle in self.controlled_vehicles:
            dones.append(vehicle.crashed or \
                         self.steps >= self.config["duration"] or \
                         (self.config["offroad_terminal"] and not vehicle.on_road))

        defender_done = dones[:self.config["n_defenders"]]
        attacker_done = dones[self.config["n_defenders"]:self.config["n_defenders"] + self.config["n_attackers"]]

        if np.all(defender_done):
            return True
        elif len(attacker_done)>0 and np.all(attacker_done):
            return True
        else:
            return False


    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

    def get_available_actions(self):
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        from onpolicy.envs.highway.highway_env.envs.common.action import  DiscreteMetaAction,MultiAgentAction

        if isinstance(self.action_type, DiscreteMetaAction):
            actions = [self.action_type.actions_indexes['IDLE']]
            for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
                if l_index[2] < self.vehicle.lane_index[2] \
                        and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                        and self.action_type.lateral:
                    actions.append(self.action_type.actions_indexes['LANE_LEFT'])
                if l_index[2] > self.vehicle.lane_index[2] \
                        and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                        and self.action_type.lateral:
                    actions.append(self.action_type.actions_indexes['LANE_RIGHT'])
            if self.vehicle.speed_index < self.vehicle.SPEED_COUNT - 1 and self.action_type.longitudinal:
                actions.append(self.action_type.actions_indexes['FASTER'])
            if self.vehicle.speed_index > 0 and self.action_type.longitudinal:
                actions.append(self.action_type.actions_indexes['SLOWER'])
            return actions

        elif isinstance(self.action_type, MultiAgentAction):
            multi_actions=[]
            for vehicle,action_type in zip(self.controlled_vehicles,self.action_type.agents_action_types):
                actions = [action_type.actions_indexes['IDLE']]
                for l_index in self.road.network.side_lanes(vehicle.lane_index):
                    if l_index[2] < vehicle.lane_index[2] \
                            and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                            and action_type.lateral:
                        actions.append(action_type.actions_indexes['LANE_LEFT'])
                    if l_index[2] > vehicle.lane_index[2] \
                            and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                            and action_type.lateral:
                        actions.append(action_type.actions_indexes['LANE_RIGHT'])
                if vehicle.speed_index < vehicle.SPEED_COUNT - 1 and action_type.longitudinal:
                    actions.append(action_type.actions_indexes['FASTER'])
                if vehicle.speed_index > 0 and action_type.longitudinal:
                    actions.append(action_type.actions_indexes['SLOWER'])
                multi_actions.append(actions)
            return multi_actions

register(
    id='highway-v0',
    entry_point='onpolicy.envs.highway.highway_env.envs:HighwayEnv',
)
