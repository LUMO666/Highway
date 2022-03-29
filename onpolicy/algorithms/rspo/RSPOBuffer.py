import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio
from tensorboardX import SummaryWriter
from collections import defaultdict
import numpy as np
from onpolicy.utils.util import check, get_shape_from_obs_space, get_shape_from_act_space
from icecream import ic
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


# running average of acceptance
class SmoothingPool():
    def __init__(self, alpha=0.1, zero_init=True):
        self.value = 1 - zero_init
        self.alpha = alpha
        self.zero_init = zero_init

    def reset(self):
        self.value = 1 - self.zero_init

    def update(self, value):
        self.value = (1 - self.alpha) * self.value + self.alpha * value

    def get(self):
        return self.value


# RSPO
class RSPOCore():
    def __init__(self, all_args, episode_length, n_rollout_threads, num_agents, obs_space, action_space,
                 lambda_factor=0.1, thre_alpha=0.3,
                 smoothing_alpha=0.1):
        self.old_actor = []
        self.nll = []
        self.ran_nll = []
        self.runner_acceptance = []
        self.all_args = all_args
        self.episode_length = episode_length
        self.n_rollout_threads = n_rollout_threads
        self.num_agents = num_agents
        self.lambda_factor = lambda_factor
        self.thre_alpha = thre_alpha
        self.minus_cross_entropy = self.all_args.minus_cross_entropy
        self.thre_solid = self.all_args.thre_solid
        self.smoothing_alpha = smoothing_alpha
        self.obs_space = obs_space
        self.action_space = action_space

    def add_actor(self, actor):
        self.old_actor.append(actor)
        self.nll.append(
            np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        )
        self.ran_nll.append(
            np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        )
        for i in self.runner_acceptance:
            i.reset()
        self.runner_acceptance.append(SmoothingPool(self.smoothing_alpha, self.all_args.continuous_from_previous))

    # insert
    @torch.no_grad()
    def insert(self, step, obs, rnn_state, action, masks, available_actions, active_masks):
        rnn_state = np.zeros_like(rnn_state)
        for agent in range(self.num_agents):
            for i, actor in enumerate(self.old_actor):
                obs_i, rnn_state_i, action_i, masks_i = \
                    obs[:, agent], rnn_state[:, agent], action[:, agent], masks[:, agent]
                if available_actions is not None:
                    available_actions_i = available_actions[:, agent]
                else:
                    available_actions_i = None
                if active_masks is not None:
                    active_masks_i = active_masks[:, agent]
                else:
                    active_masks_i = None
                # action: (n_thread, num_agents, act_dim)
                log_prob, _, _ = actor.evaluate_actions(obs_i, rnn_state_i, action_i,
                                                        masks_i, available_actions_i,
                                                        active_masks_i)
                ran_action_i = np.array([
                    self.action_space.sample() for _ in range(self.n_rollout_threads)
                ])
                ran_obs_i = np.array([self.obs_space.sample() for _ in range(self.n_rollout_threads)])
                ran_log_prob, _, _ = actor.evaluate_actions(ran_obs_i, rnn_state_i, ran_action_i,
                                                            masks_i, available_actions_i,
                                                            active_masks_i)
                self._insert(i, step, agent, -log_prob, -ran_log_prob)

    # insert data for i-th actor
    def _insert(self, i, step, agent, nll, rad_nll):
        self.nll[i][step, :, agent, :] = nll.cpu().detach().numpy()  # (step, n_threads, num_agents, 1)
        self.ran_nll[i][step, :, agent, :] = rad_nll.cpu().detach().numpy()

    # decide if to accept certain episode
    def compute_acceptance(self, rewards, masks, actions_log_prob):  # (step+1, n_threads, num_agents, 1)
        threshold = []
        trajectory_masks = 1 - masks  # revert
        trajectory_masks = trajectory_masks.astype(bool)
        trajectory_masks[-1] = np.ones_like(trajectory_masks[-1], dtype=bool)
        for i in range(len(self)):
            threshold.append(np.ones_like(rewards, dtype=float))
            for j in range(self.n_rollout_threads):
                left = right = div = ran_div = cur_div = 0
                while right < rewards.shape[0]:
                    # update threshold
                    div += self.nll[i][right, j].sum()
                    ran_div += self.ran_nll[i][right, j].sum()
                    cur_div -= actions_log_prob[right, j].sum()
                    if all(trajectory_masks[right + 1, j]):
                        # reject
                        #########################################################
                        # comment this if you don't need track inter-mediate result
                        ##########################################################
                        if self.all_args.use_wandb:
                            wandb.log({f"thre_alpha{i}_{len(self)}": div / ran_div})
                            wandb.log({f"main_div{i}_{len(self)}": div / (right + 1 - left)})
                            wandb.log({f"ran_div{i}_{len(self)}": ran_div / (right + 1 - left)})
                            wandb.log({f"cur_div{i}_{len(self)}": cur_div / (right + 1 - left)})
                        #if self.all_args.tracker is not None:
                            #self.all_args.tracker.track(f"thre_alpha{i}_{len(self)}", div / ran_div)
                            #self.all_args.tracker.track(f"main_div{i}_{len(self)}", div / (right + 1 - left))
                            #self.all_args.tracker.track(f"ran_div{i}_{len(self)}", ran_div / (right + 1 - left))
                            #self.all_args.tracker.track(f"cur_div{i}_{len(self)}", cur_div / (right + 1 - left))

                        ce = self.minus_cross_entropy * cur_div
                        if (div - ce < self.thre_alpha * (ran_div - ce)
                                or div / (right + 1 - left) < self.thre_solid):
                            threshold[i][left:right + 1, j] = np.zeros_like(threshold[i][left:right + 1, j],
                                                                            dtype=float)
                        left = right + 1
                        div = ran_div = cur_div = 0
                    right += 1

        threshold = np.stack(threshold, axis=0)
        mean_thre = threshold.mean(axis=(-1, -2, -3, -4))
        for i, k in enumerate(mean_thre):
            self.runner_acceptance[i].update(k)
        acceptance = (1 - threshold).sum(axis=0) < 1  # (step, n_threads, num_agents, 1)
        #########################################################
        # comment this if you don't need track inter-mediate result
        ##########################################################
        if self.all_args.use_wandb:
            wandb.log({f"acceptance_{len(self)}": acceptance.mean()})
            for i in range(len(self)):
                wandb.log({f"mean{i}_{len(self)}": self.runner_acceptance[i].get()})
        #if self.all_args.tracker is not None:
        #    self.all_args.tracker.track(f"acceptance_{len(self)}", acceptance.mean())
        #    for i in range(len(self)):
        #        self.all_args.tracker.track(f"mean{i}_{len(self)}", self.runner_acceptance[i].get())

        return threshold, acceptance
        # (n_old, step, n_threads, num_agent, 1), (step, n_threads, num_agent, 1)

    # compute intrinsic reward
    def rspo_reward(self, rewards, masks, actions_log_prob, eps=None):  # (step, n_threads, num_agents, 1)
        if len(self) == 0:
            return rewards
        # compute external rewards
        threshold, acceptance = self.compute_acceptance(rewards, masks, actions_log_prob)
        ext_rewards = acceptance * rewards

        # compute internal rewards
        int_rewards = np.zeros_like(self.nll[0])
        for i in range(len(self)):
            norm = len(self) if self.all_args.mean_for_internal_reward else 1
            int_rewards += (1 - self.runner_acceptance[i].get()) * self.nll[i] / norm
        #########################################################
        # comment this if you don't need track inter-mediate result
        ##########################################################
        if self.all_args.use_wandb:
            wandb.log({f"average_step_reward_{len(self)}": rewards[:, 0, 0].mean()})
            wandb.log({f"external_{len(self)}": ext_rewards[:, 0, 0].mean()})
            wandb.log({f"internal_{len(self)}": self.lambda_factor * int_rewards[:, 0, 0].mean()})
        #if self.all_args.tracker is not None:
        #    self.all_args.tracker.track(f"average_step_reward_{len(self)}", rewards[:, 0, 0].mean())
        #    self.all_args.tracker.track(f"external_{len(self)}", ext_rewards[:, 0, 0].mean())
        #    self.all_args.tracker.track(f"internal_{len(self)}", self.lambda_factor * int_rewards[:, 0, 0].mean())
        rspo_rewards = ext_rewards + eps*self.lambda_factor * int_rewards
        print("ext_rewards:",ext_rewards," | int_rewards:", self.lambda_factor * int_rewards)
        return rspo_rewards

    def __len__(self):
        return len(self.old_actor)


class RSPOBuffer(SharedReplayBuffer):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        super(RSPOBuffer, self).__init__(args, num_agents, obs_space, share_obs_space, act_space)
        self.old_actor = []
        self.num_agents = num_agents
        self.lambda_factor = args.lambda_factor
        self.thre_alpha = args.thre_alpha
        self.smoothing_alpha = args.smoothing_alpha
        self.rspo_core = RSPOCore(args, self.episode_length, self.n_rollout_threads, self.num_agents, obs_space,
                                  act_space, self.lambda_factor, self.thre_alpha, self.smoothing_alpha)

    # add a new actor
    def add_actor(self, actor):
        self.rspo_core.add_actor(actor)

    # with extra insert
    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        pre_available_actions = pre_active_masks = None
        if available_actions is not None:
            pre_available_actions = self.available_actions[self.step]
        if active_masks is not None:
            pre_active_masks = self.active_masks[self.step]
        self.rspo_core.insert(self.step, self.obs[self.step], rnn_states, actions, self.masks[self.step],
                              pre_available_actions, pre_active_masks)
        super(RSPOBuffer, self).insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                                       value_preds, rewards, masks, bad_masks=None, active_masks=None,
                                       available_actions=None)

    # with rspo rewards
    def compute_returns(self, next_value, value_normalizer=None, eps=None):
        rewards = self.rspo_core.rspo_reward(self.rewards, self.masks, self.action_log_probs, eps=eps)
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + rewards[step]
