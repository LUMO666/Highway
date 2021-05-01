import numpy as np
import gym
import math
from scipy.spatial import distance
from functools import reduce
from typing import Callable
from dataclasses import dataclass

import smarts
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.core.utils.math import vec_2d
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.scenario import Scenario
from smarts.core.utils.visdom_client import VisdomClient
from envision.client import Client as Envision

from .obs_adapter import observation_adapter
from .rew_adapter import reward_adapter


@dataclass
class Adapter:
    space: gym.Space
    transform: Callable

class SMARTSEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""
    def __init__(self, all_args):
        self.all_args = all_args

        self._dones_registered = 0    

        self.neighbor_num = all_args.neighbor_num
        self.rews_mode = all_args.rews_mode
        self.n_agents = all_args.num_agents
        self.use_proximity = all_args.use_proximity
        self.use_discrete = all_args.use_discrete # default True
        self.use_centralized_V = all_args.use_centralized_V

        self.scenarios = [(all_args.scenario_path + all_args.scenario_name)]

        self.agent_ids = ["Agent %i" % i for i in range(self.n_agents)]

        self.obs_space_dict = self.get_obs_space_dict()
        self.obs_dim = self.get_obs_dim()
        # ! TODO:
        self.share_obs_dim = self.get_state_dim() if self.use_centralized_V else self.get_obs_dim()
        self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,))] * self.n_agents
        self.share_observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(self.share_obs_dim,))] * self.n_agents
        
        if self.use_discrete:
            self.act_dim = 4
            self.action_space = [gym.spaces.Discrete(self.act_dim)] * self.n_agents
            self.agent_type = AgentType.Vulner_with_proximity if self.use_proximity else AgentType.Vulner

        else:
            # TODO Add continous action space
            self.agent_type = AgentType.VulnerCon_with_proximity if self.use_proximity else AgentType.VulnerCon
            raise NotImplementedError
        
        self._agent_specs = {
                agent_id: AgentSpec(
                    interface=AgentInterface.from_type(self.agent_type, max_episode_steps=all_args.horizon),
                    observation_adapter=self.get_obs_adapter(),
                    reward_adapter=self.get_rew_adapter(self.rews_mode, self.neighbor_num),
                    action_adapter=self.get_act_adapter(),
                )
                for agent_id in self.agent_ids
            }

        self._scenarios_iterator = Scenario.scenario_variations(
            self.scenarios, list(self._agent_specs.keys()), all_args.shuffle_scenarios,
        )

        self.agent_interfaces = {
            agent_id: agent.interface for agent_id, agent in self._agent_specs.items()
        }

        self.envision_client = None
        if not all_args.headless:
            self.envision_client = Envision(
                endpoint=all_args.envision_endpoint, output_dir=all_args.envision_record_data_replay_path
            )

        self.visdom_client = None
        if all_args.visdom:
            self.visdom_client = VisdomClient()

        self._smarts = SMARTS(
            agent_interfaces=self.agent_interfaces,
            traffic_sim=SumoTrafficSimulation(
                headless=all_args.sumo_headless,
                time_resolution=all_args.timestep_sec,
                num_external_sumo_clients=all_args.num_external_sumo_clients,
                sumo_port=all_args.sumo_port,
                auto_start=all_args.sumo_auto_start,
                endless_traffic=all_args.endless_traffic,
            ),
            envision=self.envision_client,
            visdom=self.visdom_client,
            timestep_sec=all_args.timestep_sec,
            zoo_workers=all_args.zoo_workers,
            auth_key=all_args.auth_key,
        )

    def seed(self, seed):
        self.seed = seed
        smarts.core.seed(seed)
    
    def get_obs_space_dict(self):
        
        obs_config = {
                    "distance_to_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                    "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
                    "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                    "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                    "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                    "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                    "neighbor": gym.spaces.Box(low=-1e3, high=1e3, shape=(self.neighbor_num * 5,)),
                    }
        if self.use_proximity:
            obs_config.update({"proximity":gym.spaces.Box(low=-1e10, high=1e10, shape=(8,))})
        
        obs_space_dict = gym.spaces.Dict(obs_config)

        return obs_space_dict

    def get_obs_dim(self):
        dim = 0
        for key in self.obs_space_dict.spaces.keys():
            space = list(self.obs_space_dict[key].shape)
            dim += reduce(lambda x, y: x*y, space)
        return dim

    def get_obs_adapter(self):
        def obs_adapter(env_observation):
            adapter = Adapter(
                space=self.obs_space_dict, transform=observation_adapter(self.neighbor_num, self.use_proximity)
            )
            obs = adapter.transform(env_observation)
            obs_flatten = np.concatenate(list(obs.values()), axis=0)
            return obs_flatten
        return obs_adapter

    def get_act_adapter(self):
        def action_adapter(policy_action):
            if isinstance(policy_action, (list, tuple, np.ndarray)):
                action = np.argmax(policy_action)
            else:
                action = policy_action
            action_dict = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
            return action_dict[action]
        return action_adapter

    def get_rew_adapter(self, adapter_type="vanilla", neighbor_num=3):
        return reward_adapter(adapter_type, neighbor_num)

    def _reset(self):
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        env_observations = self._smarts.reset(scenario)
        self.last_obs = env_observations
        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in env_observations.items()
        }

        return observations

    def reset(self, choose=True):
        if choose:
            try:
                self.current_observations = self._reset()
            except:
                self.close()
                self._smarts = SMARTS(
                    agent_interfaces=self.agent_interfaces,
                    traffic_sim=SumoTrafficSimulation(
                        headless=self.all_args.sumo_headless,
                        time_resolution=self.all_args.timestep_sec,
                        num_external_sumo_clients=self.all_args.num_external_sumo_clients,
                        sumo_port=self.all_args.sumo_port,
                        auto_start=self.all_args.sumo_auto_start,
                        endless_traffic=self.all_args.endless_traffic,
                    ),
                    envision=self.envision_client,
                    visdom=self.visdom_client,
                    timestep_sec=self.all_args.timestep_sec,
                    zoo_workers=self.all_args.zoo_workers,
                    auth_key=self.all_args.auth_key,
                )
                self.current_observations = self._reset()
            return self.get_obs()
        else:
            return [np.zeros(self.obs_dim) for agent_id in self.agent_ids]

    def _step(self, agent_actions):
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }
        observations, rewards, agent_dones, extras = self._smarts.step(agent_actions)

        infos = {
            agent_id: {"scores": value}
            for agent_id, value in extras["scores"].items()
        }
    
        for agent_id in observations:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]

            if self.rews_mode=="vanilla":
                rewards[agent_id] = agent_spec.reward_adapter(observation, reward)

            elif self.rews_mode=="standard":

                rewards[agent_id] = agent_spec.reward_adapter(self.last_obs[agent_id], observation, reward)
            
            elif self.rews_mode == "cruising":
                rewards[agent_id] = agent_spec.reward_adapter(observation, reward)

            self.last_obs[agent_id] = observation
            observations[agent_id] = agent_spec.observation_adapter(observation)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)

        for done in agent_dones.values():
            self._dones_registered += 1 if done else 0

        agent_dones["__all__"] = self._dones_registered == len(self._agent_specs)

        return observations, rewards, agent_dones, infos

    def step(self, action_n):
        if not np.all(action_n == np.ones((self.n_agents,)).astype(np.int) * (-1)):
            actions = dict(zip(self.agent_ids, action_n))
            self.current_observations, rewards, dones, infos = self._step(actions)
            obs_n = []
            r_n = []
            d_n = []
            info_n = []
            for agent_id in self.agent_ids:
                obs_n.append(self.current_observations.get(agent_id, np.zeros(self.obs_dim)))
                r_n.append([rewards.get(agent_id, 0.)])
                d_n.append(dones.get(agent_id, True))
                info_n.append(infos.get(agent_id, {'scores':0.}))
            return obs_n, r_n, d_n, info_n
        else:
            obs_n = [np.zeros(self.obs_dim) for agent_id in self.agent_ids]
            r_n = [[0] for agent_id in self.agent_ids]
            d_n = [None for agent_id in self.agent_ids]
            info_n = [{} for agent_id in self.agent_ids]
            return obs_n, r_n, d_n, info_n

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for i, agent_id in enumerate(self.agent_ids):
            obs_n.append(self.current_observations.get(agent_id, np.zeros(self.obs_dim)))
        return obs_n

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_state(self):
        obs_n = []
        for i, agent_id in enumerate(self.agent_ids):
            obs_n.append(self.current_observations.get(agent_id, np.zeros(self.obs_dim)))
        return obs_n

    def get_state_dim(self):
        """ Returns the shape of the state"""
        return self.obs_dim

    def render(self, mode="human"):
        """Does nothing."""
        pass

    def close(self):
        if self._smarts is not None:
            self._smarts.destroy()
