#!/bin/sh
env="Highway"
scenario="highway-v0"
task="attack"
n_defenders=1

n_attackers=1
n_dummies=0
seed=3
otheragenttype="d3qn"
otheragentpolicypath="/home/tsing89/Highway/onpolicy/envs/highway/agents/policy_pool/dqn/model/dueling_ddqn_obs25_act5_baseline.tar"
algo="rmappo"

#exp="${otheragenttype}def_seed${seed}_${n_attackers}att_1029_dis006"
exp="d3qn_crash_test_highway_epilen10"

username="liuwl"
ulimit -n 22222

echo "env is ${env}, scenario is ${scenario}, task is ${task}, algo is ${algo}, exp is ${exp}"
echo "seed is ${seed}:"

CUDA_VISIBLE_DEVICES=1 python train/train_highway.py --user_name ${username} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_rollout_threads 5 --n_eval_rollout_threads 1 --horizon 10 --episode_length 10 --log_interval 1 --use_render_vulnerability --other_agent_type ${otheragenttype} --dummy_agent_type "vi" --other_agent_policy_path ${otheragentpolicypath} --dummy_agent_type "vi" --vehicles_count 7 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --use_offscreen_render --bubble_lenth=20 --save_interval 40