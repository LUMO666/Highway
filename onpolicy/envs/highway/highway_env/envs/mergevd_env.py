import numpy as np
from gym.envs.registration import register

from onpolicy.envs.highway.highway_env import utils
from onpolicy.envs.highway.highway_env.envs.common.abstract import AbstractEnv
from onpolicy.envs.highway.highway_env.road.lane import LineType, StraightLane, SineLane
from onpolicy.envs.highway.highway_env.road.road import Road, RoadNetwork
from onpolicy.envs.highway.highway_env.envs.common.action import Action
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from onpolicy.envs.highway.highway_env.road.objects import Obstacle

import random

class MergevdEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    
    COLLISION_REWARD: float = -1
    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.2
    MERGING_SPEED_REWARD: float = -0.5
    LANE_CHANGE_REWARD: float = 0

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

    def _reset(self) -> None:
        self.use_bubble = False
        self.max_bubble_length = self.config["bubble_length"] 
        self._create_road()
        self._create_vehicles()
        self.acquire_attacker(True)

    def _create_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _create_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle

        self.road.vehicle = [defender vehicle,  merge vehicle, npc,...,npc,]
        
        self.controlled_vehicles = [defender vehicle, merge_vehicle, other attackers]

        """
        position_deviation = 10
        self.controlled_vehicles = []
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        npc_vehicles_type = utils.class_from_path(self.config["npc_vehicles_type"])
        merging_v = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(60, 0), speed=30)
        merging_v.target_speed = 30
        self.controlled_vehicles.append(merging_v)
        road.vehicles.append(merging_v)

        vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", random.choice([0,1]))).position(110+2*(random.random()-0.5)*position_deviation, 0), speed=29)
        self.controlled_vehicles.append(vehicle)
        road.vehicles.append(vehicle)

        vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 1)).position(60+2*(random.random()-0.5)*position_deviation, 0), speed=31)
        self.controlled_vehicles.append(vehicle)
        road.vehicles.append(vehicle)

        vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(130, 0), speed=31.5)
        self.controlled_vehicles.append(vehicle)
        road.vehicles.append(vehicle)

        road.vehicles.append(npc_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(70, 0), speed=28.5))

        road.vehicles.append(npc_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(10, 0), speed=31.5))

        #self.vehicle = ego_vehicle

    def enter_bubble(self):
        if self.use_bubble:
            if self.steps % self.max_bubble_length==0:
                self.acquire_attacker()

    def bubble_terminated(self):
        if self.steps % self.max_bubble_length == 0:
            return True
        else:
            return False

    def acquire_attacker(self,reset=False):
        #print(self.controlled_vehicles)
        '''
        if reset:
            for i,v in enumerate(self.controlled_vehicles):
                if i < self.config["n_defenders"]:
                    continue
                else:
                    self.road.vehicles.remove(v)
        '''
        #### only one defender
        
        defender_pos=self.controlled_vehicles[0].position
        dis=[]
        # caculate distance to defender except merge_vehicle
        for i , v in enumerate(self.road.vehicles):
            if i < self.config["n_defenders"]+1:
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
        #print(len(dis_sort))
        #print(npc_bubble)
        self.npcs_in_bubble=[]

        dis_in_bubble=deepcopy(dis_sort[:self.config["n_attackers"] - 1 + npc_bubble])
        #attacker_dis=random.sample(dis_in_bubble,self.config["n_attackers"])
        #attacker_dis do not contain merge_vehicle
        attacker_dis = dis_in_bubble[:self.config["n_attackers"]-1]
        #print(len(dis_in_bubble))
        npc_dis=[x for x in dis_in_bubble if (x not in attacker_dis) and (x != np.linalg.norm(self.road.vehicles[1].position - defender_pos))]

        # ins for attackers, ins_npc for npc in bubble
        for i in range(self.config["n_attackers"]-1):
            ins.append(dis.index(attacker_dis[i]))
        #ins = [index of chosen att in self.road.vehicles by distance]
        for i in range(npc_bubble):
            #print(i)
            #print(len(npc_dis))
            ins_npc.append(dis.index(npc_dis[i]))
        #ins_npc = [index of chosen npc in self.road.vehicles by distance]
        ##we must first define the attacker
        
        #take merge_vehicle as 1st att
        if reset:
            self.road.vehicles[self.config["n_defenders"] + 1].use_action_level_behavior = True
            #print("contorlled_v1:",len(self.controlled_vehicles))
            self.controlled_vehicles[1+self.config["n_defenders"]]=self.road.vehicles[self.config["n_defenders"] + 1]
        else:
            self.controlled_vehicles[1 + self.config["n_defenders"]].use_action_level_behavior=False
            self.road.vehicles[self.config["n_defenders"] + 1].use_action_level_behavior = True
            self.controlled_vehicles[1+self.config["n_defenders"]]=self.road.vehicles[self.config["n_defenders"]+1]

        for i,index in enumerate(ins):
            if reset:
                self.road.vehicles[self.config["n_defenders"]+ 1 + index].use_action_level_behavior = True
                self.controlled_vehicles[i+self.config["n_defenders"] + 1]=self.road.vehicles[self.config["n_defenders"] + 1 + index]
            else:
                self.controlled_vehicles[i + 1 + self.config["n_defenders"]].use_action_level_behavior=False
                self.road.vehicles[self.config["n_defenders"] + 1 + index].use_action_level_behavior = True
                self.controlled_vehicles[i + 1 + self.config["n_defenders"]]=self.road.vehicles[self.config["n_defenders"] + index + 1]
            #for render, exchange the index in the list
            self.road.vehicles[self.config["n_defenders"] + 1 + index],self.road.vehicles[self.config["n_defenders"] + i + 1]=\
            self.road.vehicles[self.config["n_defenders"] + i + 1],self.road.vehicles[self.config["n_defenders"] + index + 1]

        for i,index in enumerate(ins_npc):
            if reset:
                self.road.vehicles[self.config["n_defenders"] + 1 + index].use_action_level_behavior = True
                #print(len(self.road.vehicles))
                #print(self.config["n_defenders"] + 1 + index)
                self.controlled_vehicles[i + self.config["n_defenders"]+self.config["n_attackers"]] = self.road.vehicles[
                    self.config["n_defenders"] + 1 + index]
            else:
                self.controlled_vehicles[i + self.config["n_defenders"] + self.config["n_attackers"]].use_action_level_behavior=False
                self.road.vehicles[self.config["n_defenders"] + 1 + index].use_action_level_behavior = True
                self.controlled_vehicles[i + self.config["n_defenders"] + self.config["n_attackers"]] = self.road.vehicles[
                    self.config["n_defenders"] + 1 + index]

            self.npcs_in_bubble.append(self.road.vehicles[self.config["n_defenders"] + 1 + index])

            self.road.vehicles[self.config["n_defenders"] + 1 + index], self.road.vehicles[self.config["n_defenders"]  + self.config["n_attackers"] + i] = \
            self.road.vehicles[self.config["n_defenders"] +self.config["n_attackers"] + i], self.road.vehicles[self.config["n_defenders"] + 1 + index]
        # should add reset distance
        if reset:
            pass

        self.define_spaces()
        
        '''for i,v in enumerate(self.controlled_vehicles):
            if i>0 and i<self.config["n_attackers"]+1:
                v.SPEED_MAX=40'''


    """def acquire_attacker(self,reset=False):
        '''if reset:
            for i,v in enumerate(self.controlled_vehicles):
                if i < self.config["n_defenders"]:
                    continue
                else:
                    self.road.vehicles.remove(v)'''

        #### only one defender
        defender_pos=self.controlled_vehicles[0].position
        dis=[]
        # caculate distance to defender
        for i , v in enumerate(self.road.vehicles):
            if i < self.config["n_defenders"]:
                continue
            else:
                if v.lane_index[0]=='j' or v.lane_index[0]=='k' :
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
        # should add reset distance
        if reset:
            pass

        self.define_spaces()"""

    def _reward(self, action: Action):
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
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



    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
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
            ad_rew=10
        elif np.all(defender_done):
            ad_rew = 0
        #elif np.any(defender_done):
        #    ad_rew = 0.2
        else:
            ad_rew = 0

        return ad_rew
    def sigmoid(self,x):
    # 直接返回sigmoid函数
        return 1. / (1. + np.exp(-x))
    
    def dis2rew(self,dis) :
        longitudinal=np.abs(dis[0])
        lateral=np.abs(dis[1])
        lp=4
        x=np.power(np.power(longitudinal,lp)+np.power(2.5*lateral,lp),1/lp)-10
        reward=(self.sigmoid(x)*(1-self.sigmoid(x))*4.-0.5)*0.3
        return reward

    def dis_rew(self) :#-> bool:
        distance_rewards=[]
        for i,vehicle in enumerate(self.controlled_vehicles):
            if i < self.config["n_defenders"]:
                continue
            else:
                Distance = self.controlled_vehicles[0].position - vehicle.position
                distance_rew=self.dis2rew(Distance)
            distance_rewards.append(distance_rew)
        return distance_rewards

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
    id='mergevd-v0',
    entry_point='onpolicy.envs.highway.highway_env.envs:MergevdEnv',
)
