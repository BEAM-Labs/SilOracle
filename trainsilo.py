from scripts.trainer_siloracle import Siloracle_Trainer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from scripts.serial import RNADataset, load_rna_dataset, RNATokenizer
from scripts.model import Siloracle
import tqdm
import time
import wandb  # 添加wandb库导入
import pandas as pd
import argparse
import pickle
import os



parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="SilOracle")
parser.add_argument("--vocab_file", type=str, default="vocab_reorganized.json")
parser.add_argument("--data_folder", type=str, default="./data")
parser.add_argument("--cache_folder", type=str, default="./datacache")
parser.add_argument("--train_data_csv", type=str, default="siloracle_train.csv")
parser.add_argument("--val_data_csv", type=str, default="siloracle_val.csv")
parser.add_argument("--test_data_csv", type=str, default="siloracle_test.csv")
parser.add_argument("--model_save_folder", type=str, default="./out")
parser.add_argument("--result_save_folder", type=str, default="./out")
parser.add_argument("--pred_result_save_path", type=str, default="siloracle_test_result_0901.csv")
args = parser.parse_args()


config = {
    "model_name": f"{args.model_name}",
    "batch_size": 128, # default: 128
    "embed_dim_siRNA": 256, # default: 256
    "embed_dim_mrna": 256, # default: 256
    "embed_dim": 256, # default: 256
    "dim_feedforward": 1024, # default: 1024
    "num_layers": 2,
    "nhead": 4, # default: 4
    "dropout": 0.1, # default: 0.1
    "activation": "relu", # default: "relu"
    "lamda": 1.0, # default: 1.0
    "num_epochs": 200, # default: 200
    "learning_rate": 1e-4, # default: 1e-4
    "model_save_path": args.model_save_folder, # default: ./out
    "is_save_model": True, # default: True
    "pred_save_path": f"{args.result_save_folder}/{args.pred_result_save_path}"
}

tokenizer = RNATokenizer(vocab_file=os.path.join(args.data_folder, args.vocab_file))

csv_train_path = os.path.join(args.data_folder, args.train_data_csv)
csv_val_path = os.path.join(args.data_folder, args.val_data_csv)
csv_test_path = os.path.join(args.data_folder, args.test_data_csv)

# save cache to enable faster training and loading.
train_cache_name = os.path.join(args.cache_folder, args.train_data_csv.replace(".csv", ".pkl"))
val_cache_name = os.path.join(args.cache_folder, args.val_data_csv.replace(".csv", ".pkl"))
test_cache_name = os.path.join(args.cache_folder, args.test_data_csv.replace(".csv", ".pkl"))


train_dataset, val_dataset, test_dataset = \
            load_rna_dataset(csv_train_path, csv_val_path, csv_test_path, 
                         tokenizer, sirna_max_length=32, 
                         mrna_max_length=1024,
                         cache_name={"train": train_cache_name, 
                                     "val": val_cache_name, 
                                     "test": test_cache_name})

model = Siloracle(
    config["embed_dim_siRNA"], 
    config["embed_dim_mrna"], 
    config["embed_dim"], 
    config["num_layers"], 
    config["nhead"],
    config["dim_feedforward"],
    config["dropout"], 
    config["activation"],
)

# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, 
                             pin_memory=True, num_workers=4)
val_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, 
                           pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, 
                            pin_memory=True, num_workers=4)

# initialize trainer
trainer = Siloracle_Trainer(
    model,
    config,
    device="cuda"
)

# train model
trainer.train(train_dataloader, val_dataloader)

# use test data to do one inference and save the result to csv
val_loss, precision_dict, spearman_corr, \
    pred_values, true_values = trainer.validate(
        test_dataloader, return_preds=True, 
        model_path=os.path.join(config['model_save_path'], f"{config['model_name']}_best.pth"))
    
gene_target_symbol_name = [test_dataset[i]['gene_target_symbol_name'] for i in range(len(test_dataset))]
gene_target_ncbi_id = [test_dataset[i]['gene_target_ncbi_id'] for i in range(len(test_dataset))]

# save the result to csv
df = pd.DataFrame({
    "siRNA": test_dataset.siRNA,
    "mrna": test_dataset.mrna,
    "mRNA_remaining_pct": test_dataset.mRNA_remaining_pct,
    "siRNA_concentration": test_dataset.siRNA_concentration,
    "gene_target_symbol_name": test_dataset.gene_target_symbol_name,
    "gene_target_ncbi_id": test_dataset.gene_target_ncbi_id,
    "pred_values": pred_values,
    "true_values": true_values
})

df.to_csv(config["pred_save_path"], index=False)
