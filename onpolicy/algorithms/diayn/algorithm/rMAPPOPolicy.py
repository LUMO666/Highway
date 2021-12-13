import numpy as np
import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from .model import Discriminator


class R_MAPPOPolicy:
    def __init__(self, args, n_skills, obs_space, share_obs_space, act_space, device=torch.device("cpu"), cat_self=True):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
 
        self.obs_space.shape=(self.obs_space.shape[0]+n_skills,)
        self.share_obs_space = share_obs_space
        print(self.share_obs_space.shape[0])
        print(self.obs_space.shape[0])
        self.share_obs_space.shape=(int(self.share_obs_space.shape[0]+n_skills*(self.share_obs_space.shape[0]/(self.obs_space.shape[0]-n_skills))),)
        print(self.share_obs_space.shape[0])
        self.act_space = act_space
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device, cat_self)

        #diayn discriminator
        #print("share_obs:",self.share_obs_space.shape[0])
        #print(self.obs_space.shape[0]-n_skills)
        #for n_agent = 2 specifically
        self.discriminator = Discriminator(n_states=2*(self.obs_space.shape[0]-n_skills), n_skills=n_skills,device=self.device,n_hidden_filters=args.discriminator_n_hidden_filiters)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.discriminator_optimizer, episode, episodes, self.lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks):
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, policy_values = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, policy_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    #diayn discriminator reward
    def discriminator_calculate(self,state):
        state = torch.tensor(state, dtype=torch.float32).cuda()
        return self.discriminator(state)