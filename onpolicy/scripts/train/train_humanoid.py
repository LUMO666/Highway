#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import pybullet_envs
import gym
import torch
from frame.utils import tool

from all_config import get_config

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def wrapper(env):
    env.observation_space = (env.observation_space,)
    env.action_space = (env.action_space,)
    env.__setattr__('share_observation_space', env.observation_space)
    env.__setattr__('num_agents', 1)
    return env


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Humanoid":
                if all_args.mujoco:
                    env = wrapper(gym.make(all_args.scenario_name))
                else:
                    env = wrapper(pybullet_envs.make(all_args.scenario_name))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Humanoid":
                if all_args.mujoco:
                    env = wrapper(gym.make(all_args.scenario_name))
                else:
                    env = wrapper(pybullet_envs.make(all_args.scenario_name))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='HumanoidBulletEnv-v0',
                        help="[HumanoidBulletEnv-v0, HumanoidFlagrunBulletEnv-v0, HumanoidFlagrunHarderBulletEnv-v0]")
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # get entire argument (default and specific)
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.div_algo / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="PlayMujocoHumanoid",
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.div_algo) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    if all_args.div_algo == 'dvd':
        envs = [make_train_env(all_args) for _ in range(all_args.diversity_step)]
    else:
        envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = 1

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.div_algo == 'rspo':
        from algorithms.rspo.humanoid_rspo_runner import HumanoidRunner as Runner
    elif all_args.div_algo == 'dipg':
        from algorithms.dipg.humanoid_dipg_runner import HumanoidRunner as Runner
    elif all_args.div_algo == 'pg':
        from algorithms.pg.humanoid_pg_runner import HumanoidRunner as Runner
    elif all_args.div_algo == 'dvd':
        from algorithms.dvd.humanoid_dvd_runner import HumanoidRunner as Runner
    else:
        raise NotImplementedError

    # initial tool
    tool.init('Humanoid', all_args.experiment_name, '../results')
    runner = Runner(config, tool)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
