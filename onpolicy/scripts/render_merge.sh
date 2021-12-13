#!/bin/sh
env="Merge"
scenario="mergevd-v0"
task="attack"
n_defenders=1
n_attackers=2

n_dummies=0
algo="rmappo"
exp="d3qndef_merge_seed1_2att_0907"
seed_max=2
model_dir="/home/tsing89/Highway/onpolicy/scripts/results/Merge/mergevd-v0/rmappo/d3qndef_merge_seed1_2att_0907/wandb/run-20210907_083953-3py0y6no/files"

echo "env is ${env}"
for seed in `seq ${seed_max}`
do

    CUDA_VISIBLE_DEVICES=0 python render/render_merge.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --load_train_config --n_training_threads 1 --n_render_rollout_threads 1 --horizon 15 --use_render --use_wandb --other_agent_type "d3qn" --other_agent_policy_path "/home/tsing89/Highway/onpolicy/envs/highway/agents/policy_pool/dqn/model/d3qn_merge.tar" --model_dir ${model_dir}  --render_episodes 20 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --dummy_agent_type "vi" --save_gifs

done
