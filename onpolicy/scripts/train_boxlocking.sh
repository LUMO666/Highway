#!/bin/sh
env="BoxLocking"
scenario_name="quadrant"
task_type="all" # "all" "order" "order-return" "all-return"
num_agents=2
num_boxes=4
floor_size=6.0
algo="rmappo"
exp="newlogprobs_lr5e-4"
seed_max=1

echo "env is ${env}, scenario is ${scenario_name}, num_agents is ${num_agents}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_hns.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario_name} --task_type ${task_type} --num_agents ${num_agents} --num_boxes ${num_boxes} --floor_size ${floor_size} --seed ${seed} --n_training_threads 1 --n_rollout_threads 250 --num_mini_batch 1 --episode_length 240 --num_env_steps 50000000 --lr 5e-4 --ppo_epoch 15 --gain 0.01
    echo "training is done!"
done
