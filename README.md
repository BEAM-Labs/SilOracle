# SilOracle and SiloActive: Accurate siRNA Silencing Efficiency Prediction and New Target Discovery Powered by Language Model

SilOracle and SiloActive is a novel, accurate and fast tool for siRNA Silencing Efficiency Discovery. It utilizes the strong ability of transformer-based language models. Trained on the large siRNA dataset siRNAOD3, SilOracle demonstrates its outstanding ability to predict the silencing efficiency of siRNA sequences.

Based on the strong ability of SilOracle, we also develop an *in silico* pipeline for discovery of silencing efficiency of new targets. SiloActive is able to fit the data rapidly by only using a small amount of data, aiming to offer guidances for wet experiments.

## 0. Start your siRNA efficiency prediction by only one line of code!

For application usage, you can simply use one line of code to try the SilOracle model. Firstly, clone the repository to your local machine, ensure all the requirement packages are prepared as described in `requirements.txt`.

```bash
git clone https://github.com/Winshion/SilOracle.git
```

The test csv file is already located at `./data/siloracle_test.csv`, run following bash commands under the root path of the project:

```bash
bash make_predictions.sh
```

After a short while, the results of the prediction will save to `./out/siloracle_test_result.csv`.

If you want to perform experiments and reproduce the result, the following tutorials gives you a way.



## 1. SilOracle: Accurate siRNA silencing efficiency predictor

#### 1.1 Data Preparation

If you want to make the repreduction of the paper, make sure you download the siRNAOD3 dataset from [Shanghai Academy of AI for Science Website: Datasets](https://www.sais.com.cn/en/data-publish). After unzipping the file, move the csv file `siRNA_Open_Dataset_for_Drug_Discovery_1.0.csv` inside the downloaded folder into `./data` folder of this project. Then, run the following command at the root path of this project to make data preprocessing.

```bash
bash data_preprocessing.sh
```

After that, the preprocessed data will appear in the `./data` folder, which is comprised of:

```bash
# SilOracle dataset:
siloracle_train.csv	# data for SilOracle training
siloracle_val.csv		# data for SilOracle validation while training
siloracle_test.csv	# data for SilOracle performance evaluation

# And, SiloActive active dataset
siloactive_F5_train.csv
siloactive_F5_pool.csv
siloactive_F5_test.csv
siloactive_KHK_train.csv
siloactive_KHK_pool.csv
siloactive_KHK_test.csv
```



If you want to run SilOracle training merely on **your own data**, you should organize three csv files, the format of the csv file should look like:

```
gene_target_symbol_name,gene_target_ncbi_id,siRNA_antisense_seq,siRNA_concentration,gene_target_seq,mRNA_remaining_pct
KHK,XM_017004061.1,ACCCACGCACAGGAUCUGCUUCU,0.1,GGGGCGGGGCGGGGCCGCCGCGACCGCGGGCTTCAGGCAGGGCTGCAGATGCGAGGCCCAGCTGTACCTCGCGTGTCCCGGGT...(enter full mRNA sequence),0.9616
```

The unit of siRNA concentration is set to nM by default.

---

#### 1.2 Start SilOracle Training!

You can use only one command to enable the commence of training process:

```bash
python3 trainsilo.py \
    --model_name YourModelName \
    --vocab_file vocab_reorganized.json \       # fixed
    --data_folder ./data \
    --cache_folder ./datacache \
    --train_data_csv your_train_file.csv \
    --val_data_csv your_validation_file.csv \
    --test_data_csv your_test_file.csv \
    --model_save_folder ./out \
    --result_save_folder ./out \
    --pred_result_save_path your_name_for_test.csv
```

The training of SilOracle lasts about 2 hours on a single Nvidia Tesla A100-40G GPU for 200 epochs in siRNAOD3 dataset.

 After training, the result of inference on the test dataset will be saved at `result_save_folder/pred_result_save_path`, and state dict of the model will be saved at `model_save_folder/{YourModelName}_best.pth`.





## 2. SiloActive: A fast and accurate pipeline for new target discovery

#### 2.1 Data Preparation

Same as preparation for SilOracle, the format of data should remain exactly the same. Make sure the training set, pool set and test set contain proper amount of data. We also offer an optional command that we used in the paper for F5 and KHK targets as described in Section 1.1.

The data preprocessor can only spare the data of 2 targets in parallel. If you want to change the targets of active learning, you can manually modify the `./data/01pre_processing_siloracle.py` script by:

```python
# Change to your customized gene name;
# Make sure that the name of target do exist on the dataset.
gene1_name = 'YourGeneName1'	# on line 28
gene2_name = 'YourGeneName2'	# on line 29
```

and then modify the `./data/02pre_processing_active.py` by

```python
# Change to your customized gene name;
# Make sure that the name of target do exist on the dataset.
# And make sure gene names are exactly the same as the former code block.
gene1_name = 'YourGeneName1'	# on line 5
gene2_name = 'YourGeneName2'	# on line 6
```

It is viable to use your own data for SiloActive training, make sure they contains the same format as shown in Section 1.1.

#### 2.2 Start SiloActive Pipeline

A single command can be used for SiloActive pipeline:

```bash
python3 trainactive.py \
    --approach lowest \ # set to lowest for best performance
    --prefix a \    # prefix, can be any strings to identify your unique experiment
    --gene_target_name MyTarget \       # a prefix, your target name
    --num_samples_per_round 12 \        # default to 12
    --total_sample_rounds 10 \          # default to 10
    --model_folder ./out \
    --pretrained_model_name YourModelName_best.pth \
    --model_save_folder ./out/active \
    --cache_folder ./datacache \
    --result_folder ./out \
    --pred_result_save_path active_test_pred_result.csv \
    --active_model_save_name active_learning_model.pth \
    --data_folder ./data \
    --vocab_file vocab_reorganized.json \
    --train_data_csv your_active_train.csv \    # for pretraining before active
    --pool_data_csv your_active_pool.csv \      # pool set for selectively adding new targets
    --test_data_csv your_active_test.csv            # test set, for evaluation
```

The active process will then be conducted, all the used data samples $N_{total}$ can be calculated as:
$$
N_{total} = \text{num\_samples\_per\_round}\ * \ \text{total\_sample\_rounds} + \text{sizeof(your\_active\_train.csv)}
$$
During active learning, an extra folder under `result_folder` will be built, which records the prediction result on test sets after each active rounds.

The default sampling strategy is upper margin sampling (`--approach lowest` because we predict the remaining percentage, which is opposite to the silencing efficiency), which has already been described in the article in detail. If you want to try other sampling strategies, there's a few counterparts for you to choose:

|    Sampling Method    |    Command    |
| :-------------------: | :-----------: |
| Lower Margin Sampling |   `highest`   |
|     QBC Sampling      |     `qbc`     |
| Sample by perplexity  | `uncertainty` |
|        Random         |   `random`    |

By simply change the sampling option `--approach`, you can test the performance of other sampling methods.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

