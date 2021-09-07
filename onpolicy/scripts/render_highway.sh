#!/bin/sh
env="Highway"
scenario="highway-v0"
task="attack"
n_defenders=1
n_attackers=2

n_dummies=0
algo="rmappo"
exp="0803_2_muye_master_2attacker_03dis_rew_intention_acc1"
seed_max=2
model_dir="/home/tsing92/Highway/onpolicy/scripts/results/Highway/highway-v0/rmappo/0803_2_muye_master_2attacker_03dis_rew_intention_acc1/wandb/run-20210802_231959-2pjuwirv/files"

echo "env is ${env}"
for seed in `seq ${seed_max}`
do

    CUDA_VISIBLE_DEVICES=0 python render/render_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --load_train_config --n_training_threads 1 --n_render_rollout_threads 1 --horizon 40 --use_render --use_wandb --other_agent_type "d3qn" --other_agent_policy_path "/home/tsing92/Highway/onpolicy/envs/highway/agents/policy_pool/dqn/model/dueling_ddqn_obs25_act5_baseline.tar" --model_dir ${model_dir}  --render_episodes 20 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --dummy_agent_type "vi" --save_gifs

done
