#!/bin/sh
env="BlueprintConstruction"
scenario_name="empty"
num_agents=2
num_boxes=4
floor_size=4.0
algo="rmappo"
exp="bpc"
seed_max=1
ulimit -n 22222

echo "env is ${env}, scenario is ${scenario_name}, num_agents is ${num_agents}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_hns.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --floor_size ${floor_size} --seed ${seed} --n_training_threads 1 --n_rollout_threads 200 --num_mini_batch 1 --lr 5e-4 --episode_length 400 --num_env_steps 100000000 --ppo_epoch 15 --gain 0.01 --eval --eval_interval 10 --n_eval_rollout_threads 100
    echo "training is done!"
done
