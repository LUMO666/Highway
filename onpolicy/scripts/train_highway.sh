#!/bin/sh
env="Highway"
scenario="highway-v0"
task="attack"
n_defenders=1
n_attackers=4
n_dummies=0

algo="rmappo"
exp="muye_master"
seed_max=1
username="thumy"
ulimit -n 22222

echo "env is ${env}, scenario is ${scenario}, task is ${task}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"

    #CUDA_VISIBLE_DEVICES=1 python train/train_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --vehicles_count 0 --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 100 --horizon 40 --episode_length 40 --log_interval 1 --use_wandb --other_agent_type "d3qn" --other_agent_policy_path "../envs/highway/agents/policy_pool/dqn/model/dueling_ddqn_obs25_act5_baseline.tar" --dummy_agent_type "rvi" --dummy_agent_policy_path "../envs/highway/agents/policy_pool/dqn/model/dueling_ddqn_obs25_act5_baseline.tar"
    CUDA_VISIBLE_DEVICES=1 python train/train_highway.py --user_name ${username} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --n_eval_rollout_threads 1 --horizon 40 --episode_length 40 --log_interval 1 --use_render_vulnerability --other_agent_type "IDM" --use_centralized_V --dummy_agent_type "vi" --other_agent_policy_path "../envs/highway/agents/policy_pool/dqn/model/dueling_ddqn_obs25_act5_baseline.tar" --vehicles_count 50 --dummy_agent_type "vi" --vehicles_count 50 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --use_offscreen_render --bubble_lenth=20

done
