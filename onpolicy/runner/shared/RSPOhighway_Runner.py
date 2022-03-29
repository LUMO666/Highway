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
        #init Policy for lambda function
        self.algorithm_name = self.all_args.algorithm_name
        self.use_single_network = self.all_args.use_single_network
        self.use_diayn = self.all_args.use_diayn
        if "mappo" in self.algorithm_name:
            #diayn
            if self.use_diayn:
                from onpolicy.algorithms.diayn.r_mappo import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.diayn.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

            elif self.use_single_network:
                from onpolicy.algorithms.r_mappo_single.r_mappo_single import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            else:
                from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif "mappg" in self.algorithm_name:
            if self.use_single_network:
                from onpolicy.algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
                from onpolicy.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
            else:
                from onpolicy.algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
                from onpolicy.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            raise NotImplementedError
        self.use_centralized_V = self.all_args.use_centralized_V
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else \
            self.envs.observation_space[0]
        # RSPO parameters
        self.diversity_step = self.all_args.diversity_step
        self.load_diversity_step = self.all_args.load_diversity_step
        self.policy_fn = lambda: Policy(self.all_args,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0],
                                        device=self.device)
        self.trainer_fn = lambda policy: TrainAlgo(self.all_args, policy, device=self.device)

        #highway parameters
        self.use_render_vulnerability = self.all_args.use_render_vulnerability

        self.n_defenders = self.all_args.n_defenders
        self.n_attackers = self.all_args.n_attackers

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
                #print(self.policy.critic.hidden_size)
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
                self.env_infos = {"episode_rewards": [], 
                              "episode_dummy_rewards": [], 
                              "episode_other_rewards": [],
                              "speed": [], 
                              "cost": [], 
                              "crashed": [],
                              "adv_rewards":[],
                              "episode_length":[],
                              "bubble_rewards":[],}
                for i, s in enumerate(range(self.n_defenders + self.n_attackers)):
                    if i < self.n_defenders:
                        self.env_infos.update({"defender_{}_speed".format(i): []})
                        self.env_infos.update({"defender_{}_crash".format(i): []})
                        self.env_infos.update({"defender_{}_distance".format(i): []})
                    else:
                        self.env_infos.update({"attacker_{}_speed".format(i): []})
                        self.env_infos.update({"attacker_{}_crash".format(i): []})
                        self.env_infos.update({"attacker_{}_distance".format(i): []})
                        self.env_infos.update({"attacker_{}_dis_reward".format(i): []})

                self.adv_rew_for_zero=[]

                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)

                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions)
                    for n,info in enumerate(infos):
                        if info["bubble_stop"]:
                            new_done=[]
                            for _ in dones[n]:
                                new_done.append(True)
                            dones[n]=new_done
                    if self.use_render:
                        self.envs.render(mode = 'human')        
                    #obs = obs[:, np.newaxis, :]
                    #rewards = rewards[:, np.#newaxis, np.newaxis]
                    #dones = dones[:, np.newaxis]

                    data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                    # insert data into buffer
                    self.insert(data)

                # compute return and update network
                self.compute(1-(episode/episodes))
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
                                  self.all_args.scenario_name,
                                  self.experiment_name,
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
                    self.log_env(self.env_infos, total_num_steps)

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

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        #dones_env = np.all(dones[0], axis=-1)
        dones_env = np.all(dones, axis=-1).squeeze()
        #all data add for 1 empty dimension, dkwhy??
        #obs=obs.squeeze()
        #rewards=rewards.squeeze()

        for i, (done_env, info) in enumerate(zip(dones_env, infos)):
            # if env is done, we need to take episode rewards!
            if done_env:
                for key in info.keys():
                    if key in self.env_infos.keys():
                        self.env_infos[key].append(info[key])
                    if key == "frames" and self.use_render_vulnerability:
                        self.render_vulnerability(info[key], suffix = "train")

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        active_masks[dones_env == True] = np.zeros(((dones_env == True).sum(),self.num_agents, 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks, active_masks=active_masks)
    
    def render_vulnerability(self, frames, suffix = "train"):
        if self.all_args.save_gifs:
            save_dir = Path(str(self.run_dir) + '/vulner_' + suffix)
            if not save_dir.exists():
                curr_vulner = 'vulner1'
            else:
                exst_vulner_nums = [int(str(folder.name).split('vulner')[1]) for folder in save_dir.iterdir() if str(folder.name).startswith('vulner')]
                if len(exst_vulner_nums) == 0:
                    curr_vulner = 'vulner1'
                else:
                    curr_vulner = 'vulner%i' % (max(exst_vulner_nums) + 1)
            vulner_dir = save_dir / curr_vulner
            if not vulner_dir.exists():
                os.makedirs(str(vulner_dir))
            for idx, frame in enumerate(frames):
                imageio.mimsave(str(vulner_dir / str(idx)) + ".gif", frame, duration=self.all_args.ifi)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_envs = self.eval_envs

        if eval_envs.action_space[0].__class__.__name__ == 'Discrete':
            action_shape = 1
        
        eval_env_infos = {"episode_rewards": [],
                          "episode_dummy_rewards": [],
                          "episode_other_rewards": [],
                          "episode_length":[],
                          "adv_rewards":[],
                          "bubble_rewards": [],}

        for i in range(self.n_defenders + self.n_attackers):
            if i < self.n_defenders:
                eval_env_infos.update({"defender_{}_speed".format(i): []})
                eval_env_infos.update({"defender_{}_crash".format(i): []})
                eval_env_infos.update({"defender_{}_distance".format(i): []})
            else:
                eval_env_infos.update({"attacker_{}_speed".format(i): []})
                eval_env_infos.update({"attacker_{}_crash".format(i): []})
                eval_env_infos.update({"attacker_{}_distance".format(i): []})

        eval_episode_rewards = 0
        eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
        eval_obs = eval_envs.reset(eval_reset_choose)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_dones_env = np.zeros(self.n_eval_rollout_threads, dtype=bool)

        while True:
            eval_choose = eval_dones_env==False
            if ~np.any(eval_choose):
                break
            with torch.no_grad():
                eval_actions = np.ones((self.n_eval_rollout_threads, self.num_agents, action_shape)).astype(np.int) * (-1)
                self.trainer.prep_rollout()
                eval_action, eval_rnn_state = self.trainer.policy.act(np.concatenate(eval_obs[eval_choose]),
                                                np.concatenate(eval_rnn_states[eval_choose]),
                                                np.concatenate(eval_masks[eval_choose]),
                                                deterministic=True)
                
                eval_actions[eval_choose] = np.array(np.split(_t2n(eval_action), (eval_choose == True).sum()))
                eval_rnn_states[eval_choose] = np.array(np.split(_t2n(eval_rnn_state), (eval_choose == True).sum()))
               
            # Observe reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = eval_envs.step(eval_actions)

            eval_dones_env = np.all(eval_dones, axis=-1)

            eval_episode_rewards += eval_rewards

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_done, eval_info in zip(eval_dones, eval_infos):
                if np.all(eval_done == True):
                    for eval_info in eval_infos:
                        for key in eval_info.keys():
                            if key in eval_env_infos.keys():
                                eval_env_infos[key].append(eval_info[key]) 
                            if key == "frames" and self.use_render_vulnerability:
                                self.render_vulnerability(eval_info[key], suffix="eval")

            print("eval average episode rewards is {}".format(np.sum(eval_episode_rewards, axis=-1)))              

        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.render_envs

        render_env_infos = {"episode_rewards": [],
                            "episode_dummy_rewards": [],
                            "episode_other_rewards": [],
                            "episode_length": [],
                            "adv_rewards": [],
                            "bubble_rewards": [],
                            }

        for i in range(self.n_defenders + self.n_attackers):
            if i < self.n_defenders:
                render_env_infos.update({"defender_{}_speed".format(i): []})
                render_env_infos.update({"defender_{}_crash".format(i): []})
                render_env_infos.update({"defender_{}_distance".format(i): []})
            else:
                render_env_infos.update({"attacker_{}_speed".format(i): []})
                render_env_infos.update({"attacker_{}_crash".format(i): []})
                render_env_infos.update({"attacker_{}_distance".format(i): []})

        all_frames_all_episodes = []
        # episode
        for episode in range(self.all_args.render_episodes):
            all_frames = []
            render_choose = np.ones(self.n_render_rollout_threads) == 1.0
            obs = envs.reset(render_choose)

            if self.all_args.save_gifs:

                image = envs.render('rgb_array')[0]

                all_frames.append(image)
                all_frames_all_episodes.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            # step
            for step in range(self.all_args.horizon):
                calc_start = time.time()
                self.trainer.prep_rollout()

                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                            np.concatenate(rnn_states),
                                                            np.concatenate(masks),
                                                            deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_render_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_render_rollout_threads))

                # Obser reward and next obs


                obs, rewards, dones, infos = envs.step(actions)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0]
                    all_frames.append(image)
                    all_frames_all_episodes.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')
                
                dones_env = np.all(dones, axis=-1)

                if np.any(dones_env):
                    for key in infos[0].keys():
                        if key in render_env_infos.keys():
                            render_env_infos[key].append(infos[0][key])
                if np.any(dones_env):
                    break
            
            # save one episode gif
            if self.all_args.save_gifs:
                print(f"save gif of the episode as {episode}.gif")
                imageio.mimsave(str(self.run_dir) + '/' + str(episode) + '.gif', all_frames, duration=self.all_args.ifi)

            # log info
            print("render average episode rewards is: " + str(np.mean(np.array(render_env_infos["episode_rewards"]))))
            for i, s in enumerate(range(self.n_defenders + self.n_attackers)):
                if i < self.n_defenders:
                    print("render average episode defender_{}_speed is: ".format(i) + str(
                        np.mean(np.array(render_env_infos["defender_{}_speed".format(i)]))))
                    print("render average episode defender_{}_crash is: ".format(i) + str(
                        np.mean(np.array(render_env_infos["defender_{}_crash".format(i)]))))
                    print("render average episode defender_{}_distance is: ".format(i) + str(
                        np.mean(np.array(render_env_infos["defender_{}_distance".format(i)]))))

                else:
                    print("render average episode attacker_{}_speed is: ".format(i) + str(
                        np.mean(np.array(render_env_infos["attacker_{}_speed".format(i)]))))
                    print("render average episode attacker_{}_crash is: ".format(i) + str(
                        np.mean(np.array(render_env_infos["attacker_{}_crash".format(i)]))))
                    print("render average episode attacker_{}_distance is: ".format(i) + str(
                        np.mean(np.array(render_env_infos["attacker_{}_distance".format(i)]))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.run_dir) + '/full.gif', all_frames_all_episodes, duration=self.all_args.ifi)

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
                wandb.log({d: v}, step =int( total_num_steps + self.all_args.num_env_steps * diversity_step))

