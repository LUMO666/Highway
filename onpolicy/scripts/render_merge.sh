#!/bin/sh
env="Merge"
scenario="mergevd-v0"
task="attack"
n_defenders=1
n_attackers=2
n_dummies=0
seed=1

otheragenttype="d3qn"
if [ ${otheragenttype} = "d3qn" ]; then
  otheragentpolicypath="/home/tsing92/Highway/onpolicy/envs/highway/agents/policy_pool/dqn/model/d3qn_merge.tar"
elif [ ${otheragenttype} = "ppo" ]; then
  otheragentpolicypath="/home/tsing92/Highway/onpolicy/envs/highway/agents/policy_pool/ppo/model/merge/0901_merge_fool_ppo/0901_merge_fool_actor_50.pt"
fi

algo="rmappo"
exp="d3qndef_seed6_2att_0910"

model_dir="/home/tsing92/Highway/onpolicy/scripts/results/Merge/mergevd-v0/rmappo/d3qndef_seed6_2att_0910/wandb/run-20210910_034555-3roi4qxg/files"

echo "env is ${env}, scenario is ${scenario}, task is ${task}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 python render/render_merge.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --load_train_config --n_training_threads 1 --n_render_rollout_threads 1 --horizon 20 --use_render --use_wandb --other_agent_type ${otheragenttype} --other_agent_policy_path ${otheragentpolicypath} --model_dir ${model_dir}  --render_episodes 20 --vehicles_count 7 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --dummy_agent_type "vi" --save_gifs

