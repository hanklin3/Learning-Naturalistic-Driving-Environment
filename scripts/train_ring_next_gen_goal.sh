#!/bin/bash
#SBATCH -n 2 --gres=gpu:volta:1 -o results/training/behavior_net/ring_fix_data_0.4s.log-%j

source /etc/profile
source activate NNDE

python3 run_training_behavior_net.py --config ./configs/ring_behavior_net_training_next_gen_goals.yml --experiment-name 0013_ring_traci_fix_gpu_leak_goal_11-18
