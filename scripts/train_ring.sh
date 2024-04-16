#!/bin/bash
#SBATCH -n 2 --gres=gpu:volta:1 -o data/training/behavior_net/ring.log-%j

source /etc/profile
source activate NNDE

python run_training_behavior_net.py --config ./configs/ring_behavior_net_training.yml --experiment-name ring_behavior_net_training