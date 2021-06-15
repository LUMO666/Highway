#!/bin/sh
env="Highway"
scenario="highway-v0"
task="attack"
n_defenders=1
n_attackers=4

n_dummies=0
algo="rmappo"
exp="lwl_cl_lowspeed"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do

    CUDA_VISIBLE_DEVICES=0 python render/render_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_render_rollout_threads 1 --horizon 40 --use_render --use_wandb --other_agent_type "IDM" --model_dir "/home/tsing89/Highway/onpolicy/scripts/results/Highway/highway-v0/rmappo/lwl_IDML_cl_lowspeed/wandb/run-20210615_091316-puydbjtv/files"  --render_episodes 20 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --dummy_agent_type "vi" --save_gifs

done
