#!/bin/sh
env="Highway"
scenario="highway-v0"
task="attack"
n_defenders=1
n_attackers=4

n_dummies=0
algo="rmappo"
exp="lwl_IDML_cl_d3qn_clreward_defspeed_15_25_fixcurriculum_seed1"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do

    CUDA_VISIBLE_DEVICES=0 python render/render_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_render_rollout_threads 1 --horizon 40 --use_render --load_train_config --use_wandb --other_agent_type "d3qn" --model_dir "/home/tsing89/Highway/onpolicy/scripts/results/Highway/highway-v0/rmappo/lwl_IDML_cl_d3qn_clreward_defspeed_15_25_fixcurriculum/wandb/run-20210804_211053-10dbrkp8/files"  --render_episodes 10 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --dummy_agent_type "vi" --other_agent_policy_path "../envs/highway/agents/policy_pool/dqn/model/dueling_ddqn_obs25_act5_baseline.tar" --save_gifs

done
