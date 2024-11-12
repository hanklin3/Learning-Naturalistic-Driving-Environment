#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1 -o results/training/behavior_net/ring_behavior_net_training_origPos_10-28.log-%j

source /etc/profile
source activate NNDE

python3 run_training_behavior_net.py --config ./configs/ring_behavior_net_training_goals.yml --experiment-name ring_behavior_net_training_origPos_10-28
