#!/bin/bash

python3 make_predictions.py \
    --model_state_dict_path SilOracle_best.pth \
    --model_state_dict_folder ./out \
    --data_folder ./data \
    --vocab_file vocab_reorganized.json \
    --test_data_folder ./data \
    --test_data_csv siloracle_test.csv \
    --pred_result_save_folder ./out \
    --pred_result_save_path siloracle_test_result.csv
