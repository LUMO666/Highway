#!/bin/sh
env="Agar"
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train/train_agar.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --num_agents 2 --episode_length 30 --num_env_steps 40000000 --hidden_size 128 --attn_N 1 --attn_size 128 --attn_heads 4 --lr 2.5e-4 --use_wandb --use_attn
    echo "training is done!"
done
