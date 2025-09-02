#!/bin/bash
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_ailab
#SBATCH -A aim

python3 trainsilo.py --vocab_file vocab_reorganized.json