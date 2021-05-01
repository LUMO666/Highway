#!/bin/sh
env="hide_and_seek"
scenario_name="quadrant"
num_seekers=1
num_hiders=1
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python eval_hns.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --num_seekers ${num_seekers} --num_hiders ${num_hiders} --eval --model_dir "/home/yuchao/project/mappo-sc/results/HideAndSeek/quadrant/quadrant_seeker1_hider1_box0_ramp0_parallel200_length80/"
done
