#!/bin/bash
#SBATCH -n 2 --gres=gpu:volta:1 -o results/training/behavior_net/AA_rdbt.log-%j

source /etc/profile
source activate NNDE

python run_training_behavior_net.py --config ./configs/AA_rdbt_behavior_net_training.yml --experiment-name AA_rdbt_behavior_net_training