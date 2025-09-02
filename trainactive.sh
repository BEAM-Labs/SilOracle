#!/bin/bash
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_ailab
#SBATCH -A aim


for active_round in {1..5}; do
python3 trainactive.py \
    --approach lowest \
    --prefix siloactive_round_$active_round \
    --gene_target_name KHK \
    --num_samples_per_round 12 \
    --total_sample_rounds 50 \
    --model_folder ./out \
    --pretrained_model_name SilOracle_best.pth \
    --model_save_folder ./out/active \
    --cache_folder ./datacache \
    --result_folder ./out \
    --pred_result_save_path active_test_pred_result.csv \
    --active_model_save_name active_learning_model_KHK_round_$active_round.pth \
    --data_folder ./data \
    --vocab_file vocab_reorganized.json \
    --train_data_csv siloactive_KHK_train.csv \
    --pool_data_csv siloactive_KHK_pool.csv \
    --test_data_csv siloactive_KHK_test.csv
done