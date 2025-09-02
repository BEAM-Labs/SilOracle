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
parser.add_argument("--model_state_dict_path", type=str, default="SilOracle_best.pth")
parser.add_argument("--model_state_dict_folder", type=str, default="./out")
parser.add_argument("--data_folder", type=str, default="./data")
parser.add_argument("--vocab_file", type=str, default="vocab_reorganized.json")
parser.add_argument("--test_data_folder", type=str, default="./data")
parser.add_argument("--test_data_csv", type=str, default="siloracle_test.csv")
parser.add_argument("--pred_result_save_folder", type=str, default="./out")
parser.add_argument("--pred_result_save_path", type=str, default="siloracle_test_result.csv")
args = parser.parse_args()


config = {
    "model_name": f"SilOracle_predictions",
    "batch_size": 128, # default: 128
    "embed_dim_siRNA": 256, # default: 256
    "embed_dim_mrna": 256, # default: 256
    "embed_dim": 256, # default: 256
    "dim_feedforward": 1024, # default: 1024
    "num_layers": 2,
    "nhead": 4, # default: 4
    "dropout": 0.1, # default: 0.1
    "activation": "relu", # default: "relu"
    "lamda": 0.5, # default: 0.5
    "num_epochs": 3, # default: 200
    "learning_rate": 1e-4, # default: 1e-4
    "is_save_model": False, # default: True
    "model_save_path": args.model_state_dict_folder,
    "pred_save_path": os.path.join(args.pred_result_save_folder, args.pred_result_save_path)
}

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

tokenizer = RNATokenizer(vocab_file=os.path.join(args.data_folder, args.vocab_file))
model.load_state_dict(torch.load(
    os.path.join(args.model_state_dict_folder, args.model_state_dict_path), 
    weights_only=True))

csv_path = os.path.join(args.test_data_folder, args.test_data_csv)

_, _, test_dataset = load_rna_dataset(
    csv_path, csv_path, csv_path,
    tokenizer,
    sirna_max_length=32,
    mrna_max_length=1024,
    cache_name={"train": None, 
                "val": None, 
                "test": csv_path.replace(".csv", ".pkl")}
)

test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=4)

trainer = Siloracle_Trainer(model, config, device="cuda")

# use test data to do one inference and save the result to csv
val_loss, precision_dict, spearman_corr, \
    pred_values, true_values = trainer.validate(
        test_dataloader, return_preds=True, 
        model_path=os.path.join(args.model_state_dict_folder, args.model_state_dict_path))
    
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