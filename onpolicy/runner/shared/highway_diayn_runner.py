import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio
from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner
from pathlib import Path
from torch.nn.functional import log_softmax
import copy

def concat_state_latent(s, z_, n):
    s_concat = np.copy(s)
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    #print("s_shape:",s.shape)
    z_one_hot_expand = np.expand_dims(np.expand_dims(z_one_hot,0),0)
    z_one_hot_stack = np.tile(z_one_hot_expand,(s.shape[0],s.shape[1],1))
    return np.concatenate((s_concat,z_one_hot_stack),axis=-1)

    #for i in range(s.shape[1]-1):
    #    z_one_hot_stack = np.vstack((z_one_hot_stack,z_one_hot))
    #z_one_hot_stack = np.expand_dims(z_one_hot_stack,0)
    #return np.concatenate([s, z_one_hot_stack],2)

def _t2n(x):
    return x.detach().cpu().numpy()

class HighwayRunner(Runner):
    """
    A wrapper to start the RL agent training algorithm.
    """
    def __init__(self, config):
        super(HighwayRunner, self).__init__(config)
        # buffer

        self.use_render_vulnerability = self.all_args.use_render_vulnerability

        self.n_defenders = self.all_args.n_defenders
        self.n_attackers = self.all_args.n_attackers
        

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        #diayn parameters
        p_z = np.full(self.n_skills, 1 / self.n_skills)
        p_z_tensor = torch.tensor(p_z,dtype=torch.float32).cuda()
        #p_z_reward = np.tile(p_z,self.batch_size)
        last_logq_zs = 0

        for episode in range(episodes):
            z = np.random.choice(self.n_skills, p=p_z)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

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

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)
                origin_rewards = copy.deepcopy(rewards)
                # diayn obs
                discriminator_obs = np.copy(obs).reshape(obs.shape[0],-1)
                obs = concat_state_latent(obs,z,self.n_skills)
                for n,info in enumerate(infos):
                    if info["bubble_stop"]:
                        new_done=[]
                        for _ in dones[n]:
                            new_done.append(True)
                        dones[n]=new_done
                #diayn rewards
                #print("rewards:",rewards)
                #print("shape:",rewards.shape)
                #
                #print("dis_obs:",discriminator_obs.reshape(1,-1).shape)
                logits = self.discriminator_calculate(torch.from_numpy(discriminator_obs).detach().cuda())
                #logq_z_ns = log_softmax(logits, dim=-1)
                logq_z_ns = log_softmax(logits)
                for r_agent in rewards[0]:                    
                    #r_agent[0] = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)
                    #print(logq_z_ns.detach())
                    #print(torch.log(p_z_tensor + 1e-6))
                    r_skills = logq_z_ns.detach() - torch.log(p_z_tensor + 1e-6)
                    #print(r_skills)
                    r_agent[0] = r_skills[0][z]
                    #print("r_agent:",r_agent[0])
                    #print("r_skill:",r_skills[0][z])
                    #print("reward0:",rewards[0])
                #print("dis_rewards:",rewards)

                if self.use_render:
                    self.envs.render(mode = 'human')
                #print("whether equal",rewards==origin_rewards)
                #print("diayn_reward:",rewards)
                #print("env_reward:",origin_rewards)
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
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                print("average step rewards is {}".format(train_infos["average_step_rewards"]))
                print("average episode rewards is {}".format(np.mean(self.env_infos["episode_rewards"])))

                self.log_train(train_infos, total_num_steps)

                self.log_env(self.env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)


    def warmup(self):
        # reset env
        obs = self.envs.reset()
        #diayn obs
        obs = concat_state_latent(obs,np.random.choice(self.n_skills),self.n_skills)
        #print(obs.shape)
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        #print(share_obs.shape)
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
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

        dones_env = np.all(dones, axis=-1)
        # done_env compute the three kinds rew

        for i, (done_env, info) in enumerate(zip(dones_env, infos)):
            # if env is done, we need to take episode rewards!
            if done_env:
                for key in info.keys():
                    if key in self.env_infos.keys():
                        self.env_infos[key].append(info[key])
                    if key == "frames" and self.use_render_vulnerability:
                        self.render_vulnerability(info[key], suffix = "train")

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, active_masks=active_masks)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        return train_infos

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
        p_z = np.full(self.n_skills, 1 / self.n_skills)
        p_z_tensor = torch.tensor(p_z,dtype=torch.float32).cuda()
        #p_z_reward = np.tile(p_z,self.batch_size)
        last_logq_zs = 0

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
            z = np.random.choice(self.n_skills, p=p_z)
            print("Current policy id is ",z)
            all_frames = []
            render_choose = np.ones(self.n_render_rollout_threads) == 1.0
            obs = envs.reset(render_choose)
            obs = concat_state_latent(obs,z,self.n_skills)

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
                obs = concat_state_latent(obs,z,self.n_skills)

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
