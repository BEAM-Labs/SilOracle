#!/bin/bash
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_ailab
#SBATCH -A aim

python3 make_predictions.py \
    --model_state_dict_path SilOracle_best.pth \
    --model_state_dict_folder ./out \
    --data_folder ./data \
    --vocab_file vocab_reorganized.json \
    --test_data_folder ./data \
    --test_data_csv siloracle_test.csv \
    --pred_result_save_folder ./out \
    --pred_result_save_path siloracle_test_result.csv