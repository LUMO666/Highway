#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ChooseSimpleSubprocVecEnv, ChooseSimpleDummyVecEnv
from onpolicy.envs.highway.Mergevd_Env import MergevdEnv

def make_train_env(all_args):
    """
    the wrapper to instantiate the Highway env with multiple vehicles controlled by trained agents, Value Iteration based RL agent, training agent and rule-based agents (Intelligent Driver Model, IDM model).
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Merge":
                env = MergevdEnv(all_args)
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
            if all_args.env_name == "Merge":
                env = MergevdEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 5000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ChooseSimpleDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSimpleSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Merge":
                env = MergevdEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 5000)
            return env
        return init_env
    return ChooseSimpleDummyVecEnv([get_env_fn(0)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='merge-v1', help="Which scenario to run on")

    parser.add_argument('--task_type', type=str,
                        default='attack', choices = ["attack","defend","all"], help="train attacker or defender")

    parser.add_argument('--n_defenders', type=int,
                        default=1, help="number of defensive vehicles, default:1")
    parser.add_argument('--n_attackers', type=int,
                        default=1, help="number of attack vehicles")
    parser.add_argument('--n_dummies', type=int,
                        default=0, help="number of dummy vehicles")

    parser.add_argument('--horizon', type=int,
                        default=40, help="the max length of one task")

    parser.add_argument("--use_same_other_policy", action='store_false', default=True, help="whether to use the same model")
    parser.add_argument("--use_render_vulnerability", action='store_true', default=False, help="whether to use the same model")
    parser.add_argument("--use_offscreen_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
        parser.add_argument('--policy_path', type=str,
                        default='../envs/highway/agents/policy_pool/actor.pt', help="load_policy_path")
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.use_render and all_args.use_render_vulnerability)==False, ("can not set render options both True, turn off one of them.")

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
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
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
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)

    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    render_envs = make_render_env(all_args) if all_args.use_render else None
    
    if all_args.task_type == "attack":
        num_agents = all_args.n_attackers
    elif all_args.task_type == "defend":
        num_agents = all_args.n_defenders
    elif all_args.task_type == "all":
        num_agents = all_args.n_defenders + all_args.n_attackers
    else:
        raise NotImplementedError

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "render_envs":render_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.merge_runner import MergeRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    
    if all_args.use_eval and eval_envs is not None:
        eval_envs.close()

    if all_args.use_render and render_envs is not None:
        render_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
