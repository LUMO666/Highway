import numpy as np
from functools import reduce
from onpolicy.utils.multi_discrete import MultiDiscrete
from .envs.box_locking import BoxLockingEnv
from .envs.blueprint_construction import BlueprintConstructionEnv
from .envs.hide_and_seek import HideAndSeekEnv


class HNSEnv(object):

    def __init__(self, args):
        self.obs_instead_of_state = args.use_obs_instead_of_state
        if args.env_name == "BoxLocking":
            self.num_agents = args.num_agents
            self.env = BoxLockingEnv(args)
            self.order_obs = ['agent_qpos_qvel',
                              'box_obs', 'ramp_obs', 'observation_self']
            self.mask_order_obs = ['mask_aa_obs',
                                   'mask_ab_obs', 'mask_ar_obs', None]
        elif args.env_name == "BlueprintConstruction":
            self.num_agents = args.num_agents
            self.env = BlueprintConstructionEnv(args)
            self.order_obs = ['agent_qpos_qvel', 'box_obs',
                              'ramp_obs', 'construction_site_obs', 'observation_self']
            self.mask_order_obs = [None, None, None, None, None]
        elif args.env_name == "HideAndSeek":
            self.num_seekers = args.num_seekers
            self.num_hiders = args.num_hiders
            self.num_agents = self.num_seekers + self.num_hiders
            self.env = HideAndSeekEnv(args)
            self.order_obs = ['agent_qpos_qvel', 'box_obs',
                              'ramp_obs', 'foodict_obsbs', 'observation_self']
            self.mask_order_obs = [
                'mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', 'mask_af_obs', None]
        else:
            raise NotImplementedError

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.action_movement_dim = []

        for agent_id in range(self.num_agents):
            # deal with dict action space
            self.action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(self.action_movement))
            action_glueall = self.env.action_space['action_glueall'][agent_id].n
            action_vec = np.append(self.action_movement, action_glueall)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = self.env.action_space['action_pull'][agent_id].n
                action_vec = np.append(action_vec, action_pull)
            action_space = MultiDiscrete([[0, vec-1] for vec in action_vec])
            self.action_space.append(action_space)
            # deal with dict obs space
            obs_space = []
            obs_dim = 0
            for key in self.order_obs:
                if key in self.env.observation_space.spaces.keys():
                    space = list(self.env.observation_space[key].shape)
                    if len(space) < 2:
                        space.insert(0, 1)
                    obs_space.append(space)
                    obs_dim += reduce(lambda x, y: x*y, space)
            obs_space.insert(0, obs_dim)
            self.observation_space.append(obs_space)
            if self.obs_instead_of_state:
                self.share_observation_space.append([obs_space[0] * self.num_agents, [self.num_agents, obs_space[0]]])
            else:
                self.share_observation_space.append(obs_space)

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True):
        self.choose = choose
        if self.choose:
            dict_obs = self.env.reset()

            for i, key in enumerate(self.order_obs):
                if key in self.env.observation_space.spaces.keys():
                    if self.mask_order_obs[i] == None:
                        temp_share_obs = dict_obs[key].reshape(
                            self.num_agents, -1).copy()
                        temp_obs = temp_share_obs.copy()
                    else:
                        temp_share_obs = dict_obs[key].reshape(
                            self.num_agents, -1).copy()
                        temp_mask = dict_obs[self.mask_order_obs[i]].copy()
                        temp_obs = dict_obs[key].copy()
                        temp_mask = temp_mask.astype(bool)
                        mins_temp_mask = ~temp_mask
                        temp_obs[mins_temp_mask] = np.zeros(
                            (mins_temp_mask.sum(), temp_obs.shape[2]))
                        temp_obs = temp_obs.reshape(self.num_agents, -1)
                    if i == 0:
                        obs = temp_obs.copy()
                        share_obs = temp_share_obs.copy()
                    else:
                        obs = np.concatenate((obs, temp_obs), axis=1)
                        share_obs = np.concatenate(
                            (share_obs, temp_share_obs), axis=1)
            if self.obs_instead_of_state:
                concat_obs = np.concatenate(obs, axis=0)
                share_obs = np.expand_dims(concat_obs, 0).repeat(
                    self.num_agents, axis=0)
            self.obs_store = obs
            self.share_obs_store = share_obs
            return obs, share_obs, None
        else:
            return self.obs_store, self.share_obs_store, None

    def step(self, actions):
        if not np.all(actions == np.ones((self.num_agents, self.action_space[0].shape)).astype(np.int) * (-1)):
            action_movement = []
            action_pull = []
            action_glueall = []
            for agent_id in range(self.num_agents):
                action_movement.append(
                    actions[agent_id][:self.action_movement_dim[agent_id]])
                action_glueall.append(
                    int(actions[agent_id][self.action_movement_dim[agent_id]]))
                if 'action_pull' in self.env.action_space.spaces.keys():
                    action_pull.append(int(actions[agent_id][-1]))
            action_movement = np.stack(action_movement, axis=0)
            action_glueall = np.stack(action_glueall, axis=0)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = np.stack(action_pull, axis=0)
            actions_env = {'action_movement': action_movement,
                           'action_pull': action_pull, 'action_glueall': action_glueall}

            dict_obs, rewards, dones, infos = self.env.step(actions_env)

            if 'discard_episode' in infos.keys():
                if infos['discard_episode']:
                    obs = self.obs_store
                    share_obs = self.share_obs_store
                else:
                    for i, key in enumerate(self.order_obs):
                        if key in self.env.observation_space.spaces.keys():
                            if self.mask_order_obs[i] == None:
                                temp_share_obs = dict_obs[key].reshape(
                                    self.num_agents, -1).copy()
                                temp_obs = temp_share_obs.copy()
                            else:
                                temp_share_obs = dict_obs[key].reshape(
                                    self.num_agents, -1).copy()
                                temp_mask = dict_obs[self.mask_order_obs[i]].copy(
                                )
                                temp_obs = dict_obs[key].copy()
                                temp_mask = temp_mask.astype(bool)
                                mins_temp_mask = ~temp_mask
                                temp_obs[mins_temp_mask] = np.zeros(
                                    (mins_temp_mask.sum(), temp_obs.shape[2]))
                                temp_obs = temp_obs.reshape(
                                    self.num_agents, -1)
                            if i == 0:
                                obs = temp_obs.copy()
                                share_obs = temp_share_obs.copy()
                            else:
                                obs = np.concatenate((obs, temp_obs), axis=1)
                                share_obs = np.concatenate(
                                    (share_obs, temp_share_obs), axis=1)
                    if self.obs_instead_of_state:
                        concat_obs = np.concatenate(obs, axis=0)
                        share_obs = np.expand_dims(concat_obs, 0).repeat(
                            self.num_agents, axis=0)
            self.rewards_store = rewards
            self.infos_store = infos
            return obs, share_obs, rewards, dones, infos, None
        else:
            return self.obs_store, self.share_obs_store, np.zeros_like(self.rewards_store), None, self.infos_store, None

    def close(self):
        self.env.close()
