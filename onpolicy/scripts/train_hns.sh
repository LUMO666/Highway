#!/bin/sh
env="HideAndSeek"
scenario_name="quadrant"
num_seekers=2
num_hiders=2
num_boxes=2
num_ramps=1
num_food=0
floor_size=6.0
algo="rmappo"
exp="debug"
seed_max=1
ulimit -n 22222

echo "env is ${env}, scenario is ${scenario_name}, num_seekers is ${num_seekers}, num_hiders is ${num_hiders}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train/train_hns.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario_name} --num_seekers ${num_seekers} --num_hiders ${num_hiders} --num_boxes ${num_boxes} --num_ramps ${num_ramps} --num_food ${num_food} --floor_size ${floor_size} --seed ${seed} --n_training_threads 1 --n_rollout_threads 250 --num_mini_batch 1 --episode_length 80 --num_env_steps 500000000 --ppo_epoch 15 --gain 0.01 --use_attn --layer_N 1 --use_feature_normlization --eval_interval 10 --n_eval_rollout_threads 32
    echo "training is done!"
done
