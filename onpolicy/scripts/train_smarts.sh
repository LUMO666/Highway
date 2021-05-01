#!/bin/sh
# Specify the environment name "env" and "scenario"
# Specify agents and their borning positions 
# The borning positions are specified by the following four params.
# "born_lane_id" is index of lane that agents locate at when borning.  value Range in "straight" (0, 1, 2)
# "target_lane_id" is index of lane that agents target at. value Range in "straight" (0, 1, 2)
# "target_position_offset" is distance between agents' target position to the middle. value Range in "straight" (0~100 & "max" &"random")
# "born_position_offset" is distance between agents' borning position to the leftmost edge.  value Range in "straight" (0~100 & "max" &"random")
# The number of params should equal to the number of agents
# NOTE: agents with the same borning position will collide and die.
# Configuate the offset with respect to the leftmost edge of the straight road. 

env="SMARTS"
scenario="loop"
num_agents=3
num_lanes=3
born_lane_id="0 1 2"
born_position_offset="10 10 10"
target_lane_id="0 1 2"
target_position_offset="\"max\" \"max\" \"max\""
algo="rmappo"
exp="test"
seed_max=1

echo "building scenario ${scenario} ..."
#scl scenario build --clean --num_agents ${num_agents} --num_lanes ${num_lanes} --born_lane_id "${born_lane_id}" --born_position_offset "${born_position_offset}" --target_lane_id "${target_lane_id}" --target_position_offset "${target_position_offset}" ../envs/smarts/SMARTS/scenarios/${scenario}
echo "build scenario ${scenario} successfully!"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python -W ignore train/train_smarts.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 2 --num_mini_batch 1 --horizon 10 --episode_length 100 --num_env_steps 2000000000 --ppo_epoch 15 --gain 0.01 --lr 1e-4 --scenario_path "/home/yuchao/project/onpolicy/onpolicy/envs/smarts/SMARTS/scenarios/" --use_wandb
    echo "training is done!"
done
