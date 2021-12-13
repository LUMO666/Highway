import time
import wandb
import os
import imageio
import torch
import numpy as np
from onpolicy.algorithms.rspo.RSPOBuffer import RSPOBuffer
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class RSPORunner(Runner):
    def __init__(self, config):
        super(RSPORunner, self).__init__(config)
        # RSPO parameters
        self.diversity_step = self.all_args.diversity_step
        self.load_diversity_step = self.all_args.load_diversity_step
        self.policy_fn = lambda: Policy(self.all_args,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0],
                                        device=self.device)
        self.trainer_fn = lambda policy: TrainAlgo(self.all_args, policy, device=self.device)

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        '''
        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = f'../gifs/{self.all_args.render_name}'
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        '''
        # policy network
        self.policy = self.policy_fn()

        if self.model_dir is not None:
            self.restore(self.load_diversity_step)

        # algorithm
        self.trainer = self.trainer_fn(self.policy)

    def resume_from_base(self, path, diversity_step):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(path) + f'/model_{diversity_step}.pt',
                                                 map_location=self.device)
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(path) + f'/actor_{diversity_step}.pt',
                                                 map_location=self.device)
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(path) + f'/critic_{0}.pt',
                                                      map_location=self.device)
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for diversity_step in range(self.diversity_step):
            for episode in range(episodes):
                if diversity_step < self.all_args.resume_from_iteration and self.all_args.resume_from_base:
                    self.resume_from_base(self.all_args.resume_from_base, diversity_step)
                    print('Resume from base model!')
                    self.save(diversity_step)
                    break

                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)

                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions)

                    obs = obs[:, np.newaxis, :]
                    rewards = rewards[:, np.newaxis, np.newaxis]
                    dones = dones[:, np.newaxis]

                    data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                    # insert data into buffer
                    self.insert(data)

                # compute return and update network
                self.compute()
                train_infos = self.train()

                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                # save model
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    self.save(diversity_step)

                # log information
                if episode % self.log_interval == 0:
                    end = time.time()
                    print("\n Diversity step {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                                     .format(diversity_step,
                                             episode,
                                             episodes,
                                             total_num_steps,
                                             self.num_env_steps,
                                             int(total_num_steps / (end - start))))

                    trajectories = max(np.sum(1 - self.buffer.masks) / self.n_rollout_threads, 1)
                    train_infos["average_episode_rewards"] = np.mean(
                        self.buffer.rewards) * self.episode_length / trajectories
                    print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                    self.log_train(train_infos, total_num_steps, diversity_step)

                # eval
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(total_num_steps)

            # post process
            self.buffer.add_actor(self.policy.actor)
            # reset policy
            if diversity_step != self.diversity_step - 1:
                if self.all_args.continuous_from_previous:
                    actor_state_dict = self.policy.actor.state_dict()
                    critic_state_dict = self.policy.critic.state_dict()
                    self.policy = self.policy_fn()
                    self.policy.actor.load_state_dict(actor_state_dict)
                    self.policy.critic.load_state_dict(critic_state_dict)
                else:
                    self.policy = self.policy_fn()
                self.trainer = self.trainer_fn(self.policy)
        print('=== FINISH ===')


    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_envs = self.eval_envs

        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " + str(np.mean(eval_env_infos['eval_average_episode_rewards'])))
        return np.mean(eval_env_infos['eval_average_episode_rewards'])

    @torch.no_grad()
    def render(self):
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)


    def save(self, divergence_step):
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(policy_model.state_dict(), str(self.save_dir) + f"/model_{divergence_step}.pt")
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{divergence_step}.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{divergence_step}.pt")

    def restore(self, diversity_step):
        # MODIFY ->
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir) + f'/model_{diversity_step}.pt',
                                                 map_location=torch.device(self.all_args.local_rank))
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + f'/actor_{diversity_step}.pt',
                                                 map_location=torch.device(self.all_args.local_rank))
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + f'/critic_{diversity_step}.pt',
                                                      map_location=torch.device(self.all_args.local_rank))
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps, diversity_step):
        # MODIFY ->
        for k, v in train_infos.items():
            d = k + f"_{diversity_step}"
            if self.use_wandb:
                wandb.log({d: v}, step = total_num_steps + self.all_args.num_env_steps * diversity_step)

