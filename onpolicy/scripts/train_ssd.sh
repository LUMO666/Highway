#!/bin/sh
env="Harvest"
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_ssd.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --ppo_epoch 15 --episode_length 10 --num_env_steps 10000000 --use_wandb --share_policy
    echo "training is done!"
done
