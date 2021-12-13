#!/bin/sh
env="Roundaboutvd"
scenario="roundaboutvd-v0"
task="attack"
n_defenders=1
n_attackers=2
n_dummies=3
algo="rmappo"
exp="0907_d3qndefretrain_seed3_roundabout"
seed=1
username="thumy"
ulimit -n 22222

echo "env is ${env}, scenario is ${scenario}, task is ${task}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=1 python train/train_roundabout.py --user_name ${username} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_wandb --n_eval_rollout_threads 100 --horizon 30 --episode_length 40 --log_interval 1 --use_render_vulnerability --other_agent_type "d3qn" --use_centralized_V --dummy_agent_type "vi" --other_agent_policy_path "/home/tsing89/Highway/onpolicy/envs/highway/agents/policy_pool/dqn/model/d3qn_roundabout/d3qn_random_0904/d3qn_random_roundabout1400000.tar" --vehicles_count 7 --dummy_agent_type "vi" --vehicles_count 7 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --use_offscreen_render --bubble_lenth=20 --save_interval 40
echo "training is done!"
