#!/bin/sh
env="Roundaboutvd"
scenario="roundaboutvd-v0"
task="attack"
algo="rmappo"

n_defenders=1
n_attackers=3
n_dummies=0
otheragenttype="ppo"
if [ ${otheragenttype} = "d3qn" ]; then
  otheragentpolicypath="/home/tsing92/Highway/onpolicy/envs/highway/agents/policy_pool/dqn/model/d3qn_merge.tar"
elif [ ${otheragenttype} = "ppo" ]; then
  otheragentpolicypath="/home/tsing92/Highway/onpolicy/envs/highway/agents/policy_pool/ppo/model/roundabout/0831_actor_2M.pt"
else
  otheragentpolicypath="/home/tsing92/Highway/onpolicy/envs/highway/agents/policy_pool/dqn/model/d3qn_merge.tar"
fi
model_dir="/home/tsing92/Highway/onpolicy/scripts/results/Roundaboutvd/roundaboutvd-v0/rmappo/1018_ppodef_seed2_3att/wandb/run-20211018_210024-8k2skih2/files"

exp="1018_ppodef_seed2_3att"
seed=1

echo "env is ${env}, scenario is ${scenario}, task is ${task}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 python render/render_roundabout.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --load_train_config --n_training_threads 1 --n_render_rollout_threads 1 --horizon 20 --use_render --use_wandb --other_agent_type ${otheragenttype} --other_agent_policy_path ${otheragentpolicypath} --model_dir ${model_dir}  --render_episodes 20 --vehicles_count 7 --npc_vehicles_type "onpolicy.envs.highway.highway_env.vehicle.controller_replay.MDPVehicle_IDMVehicle" --dummy_agent_type "vi" --save_gifs



