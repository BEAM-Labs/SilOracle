from scripts.trainer_siloactive import SiloActiveTrainer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from scripts.serial import RNADataset, load_rna_dataset, RNATokenizer
from scripts.model import Siloracle
from scripts.criteria import topk_precision, print_topk_precision, spearman_correlation
import tqdm
import time
import argparse
import os

parser = argparse.ArgumentParser()
# necessary for active process
parser.add_argument("--approach", type=str, default="lowest")
parser.add_argument("--prefix", type=str, default="active")
parser.add_argument("--gene_target_name", type=str, default="MyTarget")
parser.add_argument("--num_samples_per_round", type=int, default=12)
parser.add_argument("--total_sample_rounds", type=int, default=20)
# necessary for model loading and saving
parser.add_argument("--model_folder", type=str, default="./out")
parser.add_argument("--pretrained_model_name", type=str, default="SilOracle_best.pth")
parser.add_argument("--model_save_folder", type=str, default="./out/active  ")
parser.add_argument("--cache_folder", type=str, default="./datacache")
parser.add_argument("--result_folder", type=str, default="./out")
parser.add_argument("--pred_result_save_path", type=str, default="active_test_pred_result.csv")
parser.add_argument("--active_model_save_name", type=str, default="active_learning_model.pth")
# necessary for data loading
parser.add_argument("--data_folder", type=str, default="./data")
parser.add_argument("--vocab_file", type=str, default="vocab_reorganized.json")
parser.add_argument("--train_data_csv", type=str, default="siloactive_train.csv")
parser.add_argument("--pool_data_csv", type=str, default="siloactive_pool.csv")
parser.add_argument("--test_data_csv", type=str, default="siloactive_test.csv")
args = parser.parse_args()

active_choice = args.approach

# for all folder paths below, we should check if they exist
# if not, create them
if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)
if not os.path.exists(args.cache_folder):
    os.makedirs(args.cache_folder)
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

device = "cuda"

config = {
    "model_name": "SiloActive",
    "batch_size": 128,
    "embed_dim_siRNA": 256,
    "embed_dim_mrna": 256,
    "embed_dim": 256,
    "dim_feedforward": 1024,
    "num_layers": 2,
    "nhead": 4,
    "dropout": 0.1,
    "activation": "relu",
    "lamda": 0.5,
    "pretrain_epochs": 50,
    "active_train_epochs": 30,
    "num_samples_per_round": args.num_samples_per_round,
    "total_sample_rounds": args.total_sample_rounds,
    "learning_rate_pretrain": 1e-5,
    "learning_rate_active_start": 5e-5, # originally 5e-5
    "learning_rate_active_end": 1e-6, # originally 1e-6
    "pretrained_path": f"{args.model_folder}/{args.pretrained_model_name}",
    "save_path": args.model_save_folder,
    "save_model": False,
    "approach": active_choice,
    "prefix": args.prefix,
    "gene_target_name": args.gene_target_name
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

active_trainer = SiloActiveTrainer(model, config)

print(f"Active on {config['gene_target_name']}, {active_choice}")

csv_train_path = f"{args.data_folder}/{args.train_data_csv}"
csv_new_target_train_path = f"{args.data_folder}/{args.pool_data_csv}"
# this val path below is actually not used, for best practice,
# please use the same path as the pool path
csv_new_target_val_path = f"{args.data_folder}/{args.pool_data_csv}"
csv_new_target_test_path = f"{args.data_folder}/{args.test_data_csv}"

# save prefix for active learning
save_prefix = f"{config['gene_target_name']}_{active_choice}_{args.prefix}"

tokenizer = RNATokenizer(vocab_file=args.data_folder + "/" + args.vocab_file)

train_dataset, _, _ = \
            load_rna_dataset(csv_train_path, csv_train_path, csv_train_path, 
                         tokenizer, sirna_max_length=config["embed_dim_siRNA"], 
                         mrna_max_length=256,
                         cache_name={"train": f"siloactive_{config['gene_target_name']}_train.pkl", 
                                     "val": f"siloactive_{config['gene_target_name']}_train.pkl", 
                                     "test": f"siloactive_{config['gene_target_name']}_train.pkl"})
            
active_poolset, _, active_testset = \
    load_rna_dataset(csv_new_target_train_path, csv_new_target_val_path, 
                         csv_new_target_test_path, tokenizer,
                         sirna_max_length=config["embed_dim_siRNA"], 
                         mrna_max_length=256,
                         cache_name={"train": f"siloactive_{config['gene_target_name']}_pool.pkl",
                                     "val": f"siloactive_{config['gene_target_name']}_pool.pkl",
                                     "test": f"siloactive_{config['gene_target_name']}_test.pkl"})
    
# print dataset size
print(f"Active poolset size: {len(active_poolset)}")
print(f"Active testset size: {len(active_testset)}")

# output the number of high efficiency in the test set.
high_efficiency_samples = [i for i in range(len(active_testset)) if active_testset[i]["mRNA_remaining_pct"] <= 0.3]
print(f"High efficiency samples: {len(high_efficiency_samples)} / {len(active_testset)}")
# active_trainer.pretrain(train_dataset, val_dataset, None, pretrain_epochs=config["pretrain_epochs"])

print("Validate before training...")
val_loss, precision_dict, spearman_corr = active_trainer.validate(valset=active_testset, in_active_round=False)
print(f"Validation loss: {val_loss:.4f}, Spearman correlation: {spearman_corr:.4f}")
print_topk_precision(precision_dict)
print("Validation Finished.")

print("Start training...")
model = active_trainer.active_one_new_target(train_dataset, active_testset,
                                            active_poolset, active_testset, 
                                            pretrain_epochs=config["pretrain_epochs"], 
                                            active_train_epochs=config["active_train_epochs"], 
                                            num_samples_per_round=config["num_samples_per_round"], 
                                            total_sample_rounds=config["total_sample_rounds"], 
                                            approach=active_choice,
                                            save_prefix=save_prefix)

# save the model
with open(f"{args.model_save_folder}/{args.active_model_save_name}", "wb") as f:
    torch.save(model.state_dict(), f)