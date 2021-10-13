from typing import Tuple

from gym.envs.registration import register
import numpy as np

from onpolicy.envs.highway.highway_env import utils
from onpolicy.envs.highway.highway_env.envs.common.abstract import AbstractEnv
from onpolicy.envs.highway.highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from onpolicy.envs.highway.highway_env.road.road import Road, RoadNetwork
from onpolicy.envs.highway.highway_env.envs.common.action import Action
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from onpolicy.envs.highway.highway_env.road.objects import Obstacle

import random
import pdb


class RoundaboutvdEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    
    COLLISION_REWARD: float = -1
    RIGHT_LANE_REWARD: float = 0
    HIGH_SPEED_REWARD: float = 0.2
    LANE_CHANGE_REWARD: float = 0
    
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_count": 50,
            "initial_lane_id": None,
            "incoming_vehicle_destination": None,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
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
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _create_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle

        self.road.vehicle = [defender vehicle,  merge vehicle, npc,...,npc,]
        
        self.controlled_vehicles = [defender vehicle, merge_vehicle, other attackers]

        """
        position_deviation = 20
        speed_deviation = 2

        self.controlled_vehicles = []
        road = self.road

        ego_lane = self.road.network.get_lane(("ser", "ses", 0))

        ego_vehicle = self.action_type.vehicle_class(road,
                                                     ego_lane.position(125, 0),
                                                     speed=8,
                                                     heading=ego_lane.heading_at(140))
        try:
            ego_vehicle.plan_route_to("wxs")
        except AttributeError:
            pass
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 16
        MDPVehicle.SPEED_COUNT = 3
        ego_vehicle.use_action_level_behavior = True
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        destinations = ["exr", "sxr", "nxr"]
        npc_vehicles_type = utils.class_from_path(self.config["npc_vehicles_type"])

        for i in range(self.config['n_attackers']):
            '''vehicle = self.action_type.vehicle_class(road,
                                                       road.network.get_lane(("ne", "wx", i)).position(
                                                       20*(i+1) + 2*(random.random()-0.5)*position_deviation,0),
                                                       speed=16 + 2*(random.random()-0.5) * speed_deviation,
                                                       heading=2-i*0.5)'''
            '''attacker_lane=road.network.get_lane(("we", "sx", i))
            local_position=20*(i+1) + 2*(random.random()-0.5)*position_deviation
            Position=attacker_lane.position(local_position,0)
            vehicle = self.action_type.vehicle_class(road,
                                                       Position,
                                                       speed=16 + 2*(random.random()-0.5) * speed_deviation,
                                                       heading=attacker_lane.heading_at(local_position))'''
            attacker_lane=road.network.get_lane(("sx","se",random.randint(0,1)))
            local_position=2*(random.random()-0.5)*position_deviation
            Position=attacker_lane.position(local_position,0)                                           
            vehicle = self.action_type.vehicle_class(road,
                                                       Position,
                                                       speed=16 + 2*(random.random()-0.5) * speed_deviation,
                                                       heading=attacker_lane.heading_at(local_position))
            vehicle.use_action_level_behavior = True
            self.controlled_vehicles.append(vehicle)
            road.vehicles.append(vehicle)
        
        for i in list(range(-1, 0)):
            vehicle = npc_vehicles_type.make_on_lane(self.road,
                                                       ("ne", "wx", 0),
                                                       longitudinal=20*i + 2*(random.random()-0.5)*position_deviation,
                                                       speed=16 + 2*(random.random()-0.5) * speed_deviation)
            #vehicle.plan_route_to(random.choice(destinations))
            vehicle.plan_route_to("wxr")
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Incoming vehicle
        vehicle = npc_vehicles_type.make_on_lane(self.road,
                                                   ("ee", "nx", 1),
                                                   longitudinal=5 + 2*(random.random()-0.5)*position_deviation,
                                                   speed=16 + 2*(random.random()-0.5) * speed_deviation)

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = random.choice(destinations)
        
        #vehicle.plan_route_to(destination)
        vehicle.plan_route_to("wxr")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Entering vehicle
        '''vehicle = npc_vehicles_type.make_on_lane(self.road,
                                                   ("eer", "ees", 0),
                                                   longitudinal=50 + 2*(random.random()-0.5) * position_deviation,
                                                   speed=16 + 2*(random.random()-0.5) * speed_deviation)
        vehicle.plan_route_to(random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)'''
        #self.define_spaces()
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
            if i < self.config["n_defenders"]+2:
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

        dis_in_bubble=deepcopy(dis_sort[:self.config["n_attackers"] - 2 + npc_bubble])
        #attacker_dis=random.sample(dis_in_bubble,self.config["n_attackers"])
        #attacker_dis do not contain merge_vehicle
        attacker_dis = dis_in_bubble[:self.config["n_attackers"]-2]
        #print(len(dis_in_bubble))
        npc_dis=[x for x in dis_in_bubble if (x not in attacker_dis) and (x != np.linalg.norm(self.road.vehicles[1].position - defender_pos))]

        # ins for attackers, ins_npc for npc in bubble
        for i in range(self.config["n_attackers"]-2):
            ins.append(dis.index(attacker_dis[i]))
        #ins = [index of chosen att in self.road.vehicles by distance]
        for i in range(npc_bubble):
            #print(i)
            #print(len(npc_dis))
            ins_npc.append(dis.index(npc_dis[i]))
        #ins_npc = [index of chosen npc in self.road.vehicles by distance]
        ##we must first define the attacker
        
        #take merge_vehicle as 1st att
        '''if reset:
            self.road.vehicles[self.config["n_defenders"] + 2].use_action_level_behavior = True
            #print("contorlled_v1:",len(self.controlled_vehicles))
            self.controlled_vehicles[1+self.config["n_defenders"]]=self.road.vehicles[self.config["n_defenders"] + 1]
        else:
            self.controlled_vehicles[1 + self.config["n_defenders"]].use_action_level_behavior=False
            self.road.vehicles[self.config["n_defenders"] + 2].use_action_level_behavior = True
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
            self.road.vehicles[self.config["n_defenders"] + i + 1],self.road.vehicles[self.config["n_defenders"] + index + 1]'''
        
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
    id='roundaboutvd-v0',
    entry_point='onpolicy.envs.highway.highway_env.envs:RoundaboutvdEnv',
)
