#!/bin/sh
env="MPE"
scenario="simple_speaker_listener"  # simple_speaker_listener # simple_spread
num_landmarks=3
num_agents=3
algo="mappo"
exp="mlp"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 8 --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --share_policy --use_recurrent_policy

    echo "training is done!"
done
