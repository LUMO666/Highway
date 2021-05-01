import numpy as np
from typing import Tuple
from gym.envs.registration import register

from onpolicy.envs.highway.highway_env import utils
from onpolicy.envs.highway.highway_env.envs.common.abstract import AbstractEnv
from onpolicy.envs.highway.highway_env.envs.common.action import Action
from onpolicy.envs.highway.highway_env.road.road import Road, RoadNetwork
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle
import imageio
from copy import copy

class HighwayEnvBacktrack(AbstractEnv):
    """
    A highway driving environment with the backtrack enhancement.

    The backtrack enhancement function is sepcified by the "backward_step" hyperparameter.
    When the ego vehicle of the environment crash, the environment will backward "backward_step" before and save 
    rendered videos of both the crash case and the backtrack case.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.9
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""

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
            "npc_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": 0,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
    
  
            "enable_backtrack": True,
            "backward_step": 20
        })

        self._inner_controlled_vechiles_backup_config = []  # a list of dict [{},{},{},{}, ... ,{}], each dict = configuration of each step.
        self._inner_controlled_vechiles_backup_action = []  # a list of "list of int" [[0,1,...], [1,2,..], ... ]. each sub = actions of all controlled car at each step.
        self._inner_other_vechiles_backup_config = []       # a list of dict [{},{},{},{}, ... ,{}], each dict = configuration of each step.
        self._inner_backup_backtrack_step = 0 # the number of step that we wanna backup to 
        self._in_reward_backward_stage = False
        self._inner_reset_backward = False
        self.backtrack_time  = 0
        return config

    def _reset(self) -> None:
        if self._inner_reset_backward:
            self.road.vehicles = []
            self._reset_from_inner_backup()
        else:
            self._create_road()
            self._create_vehicles()
            self._inner_controlled_vechiles_backup_config = []
            self._inner_other_vechiles_backup_config = []
            self._inner_controlled_vechiles_backup_action = []

    def _reset_from_inner_backup(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        
        self.controlled_vehicles = []
        train_config=self._inner_controlled_vechiles_backup_config[self._inner_backup_backtrack_step]
        npc_config=self._inner_other_vechiles_backup_config[self._inner_backup_backtrack_step]
        print(f"train_config = {train_config}")
        print(f"npc_config = {npc_config}")

        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, train_config[_]["trained_vehicle_position"], train_config[_]["trained_vehicle_heading"], train_config[_]["trained_vehicle_speed"], ("0","1", train_config[_]["trained_vehicle_target_lane_index"]), train_config[_]["trained_vehicle_target_speed"])
            
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            
        vehicles_type = utils.class_from_path(self.config["npc_vehicles_type"])
        self.other_vehicles = []
        for _ in range(self.config["vehicles_count"]):
            other_vehicle = vehicles_type(road=self.road, position=npc_config[_]["npc_vehicle_position"], heading=npc_config[_]["npc_vehicle_heading"], speed=npc_config[_]["npc_vehicle_speed"], target_lane_index=("0","1", npc_config[_]["npc_vehicle_target_lane_index"]), target_speed=npc_config[_]["npc_vehicle_target_speed"], timer= npc_config[_]["npc_vehicle_timer"])
            self.other_vehicles.append(other_vehicle)
            self.road.vehicles.append(other_vehicle)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []

        for i in range(self.config["controlled_vehicles"]):

            vehicle = self.action_type.vehicle_class.create_random(self.road,
                                                                   speed=25,
                                                                   lane_id=self.config["initial_lane_id"],
                                                                   spacing=self.config["ego_spacing"],
                                                                   )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)


        vehicles_type = utils.class_from_path(self.config["npc_vehicles_type"])
        self.other_vehicles = []

        for _ in range(self.config["vehicles_count"]):
            other_vehicle = vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
            self.other_vehicles.append(other_vehicle)
            self.road.vehicles.append(other_vehicle)
            

    def _reward(self, action: Action) :#-> float: now we return a list
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """

        # -> float: now we change it to return a list!!!!!
        n_defenders=self.config["n_defenders"]
        n_attackers=self.config["n_attackers"]
        n_dummies=self.config["n_dummies"]
        rewards=[]
        for vehicle in self.controlled_vehicles:
        
            neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
            lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
                else vehicle.lane_index[2]
            scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
            reward = \
                + self.config["collision_reward"] * vehicle.crashed \
                + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
                + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
            #reward = utils.lmap(reward,
            #              [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
            #              [0, 1])
            #reward = 0 if not vehicle.on_road else reward
            reward = -1 if not vehicle.on_road else reward
            '''
            scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
            reward = \
                + self.config["collision_reward"] * vehicle.crashed \
                + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
            #reward = utils.lmap(reward,
            #                    [self.config["collision_reward"], self.HIGH_SPEED_REWARD ],
            #                    [0, 1])
            reward = -1 if not vehicle.on_road else reward
            '''
            rewards.append(reward)

            # save configuration to self._inner_backup_config
            if self.config["enable_backtrack"]:
                self._save_backtrack_configuration(action)
            
            # reward correction based on the last several steps' behavior
            # currently reward is not changed. Just implement backtrack
            if self.vehicle.crashed:
                reward = self._reward_backward_correction(reward)
    
         
        return rewards

    def _is_terminal(self) :#-> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
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
                         self.steps >= self.config["duration"] or \
                         (self.config["offroad_terminal"] and not vehicle.on_road))

        defender_done = dones[:self.config["n_defenders"]]
        attacker_done = dones[self.config["n_defenders"]:self.config["n_defenders"] + self.config["n_attackers"]]

        if np.all(defender_done) and (not np.all(attacker_done)):
            return 1
        else:
            return 0

    def _is_done(self) :#-> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
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
                dones[i]=True
            return True
        elif len(attacker_done)>0 and np.all(attacker_done):
            for i in range(len(dones)):
                dones[i]=True
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
    def _save_backtrack_configuration(self, action) -> None:
        temp_dict_list = []
        for _i in range(self.config["controlled_vehicles"]):
            temp_position = copy(self.controlled_vehicles[_i].position)
            temp_dict_list.append( dict(zip( ["trained_vehicle_lane_index", "trained_vehicle_position", "trained_vehicle_heading", "trained_vehicle_speed", "trained_vehicle_target_lane_index", "trained_vehicle_target_speed"], [self.controlled_vehicles[_i].lane_index[2], temp_position, self.controlled_vehicles[_i].heading, self.controlled_vehicles[_i].speed, self.controlled_vehicles[_i].target_lane_index[2], self.controlled_vehicles[_i].target_speed] )) )
        
        self._inner_controlled_vechiles_backup_config.append(temp_dict_list)

        if isinstance(action, list):
            self._inner_controlled_vechiles_backup_action.append(action)
        else:
            self._inner_controlled_vechiles_backup_action.append([action])
        temp_dict_list_other_vehicle = []

        for _ in range(self.config["vehicles_count"]):
            temp_npc_position = copy(self.other_vehicles[_].position)
            temp_dict_list_other_vehicle.append( dict(zip( ["npc_vehicle_lane_index", "npc_vehicle_position", "npc_vehicle_heading", "npc_vehicle_speed", "npc_vehicle_target_lane_index", "npc_vehicle_target_speed","npc_vehicle_timer"], [self.other_vehicles[_].lane_index[2], temp_npc_position, self.other_vehicles[_].heading, self.other_vehicles[_].speed, self.other_vehicles[_].target_lane_index[2] , self.other_vehicles[_].target_speed, self.other_vehicles[_].timer  ] )) )
        
        self._inner_other_vechiles_backup_config.append(temp_dict_list_other_vehicle)


    def _reward_backward_correction(self, reward) -> float:
        if self._inner_reset_backward:
            # print(f"pass")
            pass
        elif self.config["reward_shape_correction"]:
            # print(f"\n{self.backtrack_time}")
            self._inner_reset_backward = True
            reward_before = reward
            step_when_backward = self.steps # the number of step when env backups to "backward_step" before.
            self._inner_backup_backtrack_step = max(self.steps - self.config["backward_step"],0) # backward to 10 steps before.
            # print(f"current step = {self.steps}; backup step = {self._inner_backup_backtrack_step}")
            # print(f"self._inner_controlled_vechiles_backup_config = {self._inner_controlled_vechiles_backup_config}")
            # print(f"self._inner_other_vechiles_backup_config = {self._inner_other_vechiles_backup_config}")
            for _ in range(self.config["vehicles_count"]):
                # print(f"last npc position = {self.other_vehicles[_].position[0]}")
            for _i in range(self.config["controlled_vehicles"]):
                # print(f"last controlled vehicle={self.controlled_vehicles[_i].position[0]}")
            backup_action = self._inner_controlled_vechiles_backup_action
            # print(f"backup_action = {backup_action}")
            images = []
            rendered_image = self.get_rendered_image()
            for single_image in rendered_image:
                images.append(single_image)
            imageio.mimsave(uri="crash"+str(self.backtrack_time)+".gif", ims=images, format="GIF", duration=0.01) 
            
            self.reset()
            images = []
            for _ in range(self.config["vehicles_count"]):
                # print(f"at lane {self.other_vehicles[_].lane_index} npc position = {self.other_vehicles[_].position[0]}")
            for _i in range(self.config["controlled_vehicles"]):
                # print(f"at lane {self.controlled_vehicles[_i].lane_index} controlled vehicle={self.controlled_vehicles[_i].position[0]}")
            backtrack_action_list = []
            temp_terminal = False
            for _i in range(min(self.config["backward_step"]-1, step_when_backward-1)):
                if not temp_terminal:
                    taken_action = backup_action[self._inner_backup_backtrack_step + _i + 1]
                    temp_observation, temp_reward, temp_terminal, temp_info = self.step(taken_action[0])
                    backtrack_action_list.append(taken_action)
            rendered_image = self.get_rendered_image()
            for single_image in rendered_image:
                images.append(single_image)
            self._inner_reset_backward = False
            # print(f"backtrack action list = {backtrack_action_list}")
            for _ in range(self.config["vehicles_count"]):
                # print(f"at lane {self.other_vehicles[_].lane_index} npc position = {self.other_vehicles[_].position[0]}")
            for _i in range(self.config["controlled_vehicles"]):
                # print(f"at lane {self.controlled_vehicles[_i].lane_index} controlled vehicle={self.controlled_vehicles[_i].position[0]}")
            imageio.mimsave(uri="backtrack"+str(self.backtrack_time)+".gif", ims=images, format="GIF", duration=0.01) 
            self.backtrack_time += 1
        return reward
register(
    id='highwayBacktrack-v0',
    entry_point='onpolicy.envs.highway.highway_env.envs:HighwayEnvBacktrack',
)

