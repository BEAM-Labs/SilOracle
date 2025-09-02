# Active learning with AGT data
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import List, Dict, Tuple
from scripts.criteria import topk_precision, print_topk_precision, \
    spearman_correlation, RankNetLoss
from scripts.serial import RNADataset
import os


class SiloActiveTrainer:
    def __init__(self, model, config, device: torch.device="cuda"):
        self.model = model
        self.mse_loss = nn.MSELoss().to(device)
        self.ranknet_loss = RankNetLoss().to(device)
        self.config = config
        self.device = device
        self.pretrain_epochs = config["pretrain_epochs"]
        self.active_train_epochs = config["active_train_epochs"]
        self.num_samples_per_round = config["num_samples_per_round"]
        self.total_sample_rounds = config["total_sample_rounds"]
        self.lamda = config["lamda"]
        self.model_name = config["model_name"]
        self.pretrained_path = config["pretrained_path"]
        self.committee_models = []
        if config["approach"] == "qbc":
            for i in range(5):
                # Get model configuration
                model_config = {
                    "input_sirna_dim": config["embed_dim_siRNA"],
                    "input_mrna_dim": config["embed_dim_mrna"],
                    "embed_dim": config["embed_dim"],
                    "num_layers": config.get("num_layers", 6),
                    "nhead": config.get("nhead", 8),
                    "dim_feedforward": config.get("dim_feedforward", 2048),
                    "dropout": config.get("dropout", 0.1),
                    "activation": config.get("activation", "relu"),
                    "device": device,
                }
                
                # Create new model instance
                committee_model = type(model)(**model_config)
                committee_model.to(device)
                
                # Load pretrained weights
                try:
                    pretrained_path = self.pretrained_path
                    print(f"Loading pretrained weights: {pretrained_path}")
                    state_dict = torch.load(pretrained_path, map_location=device)
                    committee_model.load_state_dict(state_dict)
                    
                    # # Add small random perturbations to loaded weights
                    # print(f"Adding parameter perturbations for committee member {i+1}...")
                    # for param in committee_model.parameters():
                    #     # Add small Gaussian noise, standard deviation is 1% of original parameter standard deviation
                    #     noise = torch.randn_like(param) * param.std() * 1.0e-3
                    #     param.data.add_(noise)
                        
                except Exception as e:
                    print(f"Warning: Failed to load pretrained weights - {str(e)}")
                    print("Using randomly initialized parameters")
                    for param in committee_model.parameters():
                        nn.init.normal_(param, mean=0.0, std=0.02)
                
                self.committee_models.append(committee_model)
                print(f"Committee member {i+1} initialization completed")
        else:
            self.model.load_state_dict(torch.load(self.pretrained_path, map_location=self.device))
        
    def pretrain_one_epoch(self, train_loader, optimizer):
        self.model.train()
        running_loss = 0.0
        train_loops = tqdm(train_loader)
        
        for i, batch in enumerate(train_loops):
            siRNA = batch["sirna"].to(self.device)
            mrna = batch["mrna"].to(self.device)
            siRNA_concentration = batch["siRNA_concentration"].to(self.device).unsqueeze(1)
            mRNA_remaining_pct = batch["mRNA_remaining_pct"].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(siRNA, mrna, siRNA_concentration).squeeze()
            mse_loss = self.mse_loss(outputs, mRNA_remaining_pct)
            ranknet_loss = self.ranknet_loss(outputs, mRNA_remaining_pct)
            loss = self.lamda * ranknet_loss + (1 - self.lamda) * mse_loss
            
            loss.backward()
            optimizer.step()
            
            train_loops.set_postfix(loss=loss.item(), 
                                    mse_loss=mse_loss.item(), 
                                    ranknet_loss=ranknet_loss.item())
            running_loss += loss.item()
            
        return running_loss / len(train_loader)
    
    def pretrain(self, optimizer, trainset, valset, 
                 new_target_testset=None, pretrain_epochs=100):
        # pretrain processes
        train_loader = DataLoader(trainset, batch_size=self.config["batch_size"], shuffle=True, 
                                 pin_memory=True, num_workers=4)
        
        # Pre-training validation
        val_loss, precision_dict, spearman_corr = self.validate(valset)
        print("-" * 50)
        print(f"Pre-training validation loss: {val_loss:.4f}, Spearman correlation: {spearman_corr:.4f}")
        print_topk_precision(precision_dict)
        print("-" * 50)
        
        # If using QBC, also need to pretrain committee members
        if len(self.committee_models) > 0:
            print("Pretraining committee members...")
            committee_optimizers = [
                optim.Adam(model.parameters(), lr=self.config["learning_rate_pretrain"])
                for model in self.committee_models
            ]
            
        for epoch in range(pretrain_epochs):
            train_loss = self.pretrain_one_epoch(train_loader, optimizer)
            
            # Train committee members
            if len(self.committee_models) > 0:
                for i, (committee_model, committee_optimizer) in enumerate(zip(self.committee_models, committee_optimizers)):
                    committee_train_loss = self.pretrain_committee_member(
                        committee_model, train_loader, committee_optimizer)
                    print(f"Committee member {i+1} training loss: {committee_train_loss:.4f}")
            
            if new_target_testset is not None:
                new_target_val_loss, new_target_precision_dict, new_target_spearman_corr = self.validate(new_target_testset)
                print(f"Epoch {epoch+1}/{self.pretrain_epochs}, new target validation loss: {new_target_val_loss:.4f}, new target validation Spearman correlation: {new_target_spearman_corr:.4f}")
                print_topk_precision(new_target_precision_dict)
                print("-" * 50)
            else:
                val_loss, precision_dict, spearman_corr = self.validate(valset)
                print(f"Epoch {epoch+1}/{self.pretrain_epochs}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}, Spearman correlation: {spearman_corr:.4f}")
                print_topk_precision(precision_dict)
                print("-" * 50)
                
    def pretrain_committee_member(self, committee_model, train_loader, optimizer):
        """Train individual committee member"""
        committee_model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            siRNA = batch["sirna"].to(self.device)
            mrna = batch["mrna"].to(self.device)
            siRNA_concentration = batch["siRNA_concentration"].to(self.device).unsqueeze(1)
            mRNA_remaining_pct = batch["mRNA_remaining_pct"].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = committee_model(siRNA, mrna, siRNA_concentration).squeeze()
            mse_loss = self.mse_loss(outputs, mRNA_remaining_pct)
            ranknet_loss = self.ranknet_loss(outputs, mRNA_remaining_pct)
            
            loss = self.lamda * ranknet_loss + (1 - self.lamda) * mse_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, valset, in_active_round=False, active_round=0, is_save=False, save_prefix="run0"):
        val_loader = DataLoader(valset, batch_size=self.config["batch_size"], shuffle=False, 
                                pin_memory=True, num_workers=4)
        self.model.eval()
        running_loss = 0.0
        all_gene_target_symbol_names = []
        all_gene_target_ncbi_ids = []
        all_preds = []
        all_truths = [] 
        
        # Check if dataset is empty
        if len(valset) == 0:
            print("Warning: Validation set is empty!")
            if in_active_round:
                return 0.0, {}, 0.0, torch.tensor([]), torch.tensor([])
            else:
                return 0.0, {}, 0.0
        
        val_loops = tqdm(val_loader)
        
        for i, batch in enumerate(val_loops):
            siRNA = batch["sirna"].to(self.device)
            mrna = batch["mrna"].to(self.device)
            gene_target_symbol_name = batch["gene_target_symbol_name"]
            gene_target_ncbi_id = batch["gene_target_ncbi_id"]
            siRNA_concentration = batch["siRNA_concentration"].to(self.device).unsqueeze(1)
            mRNA_remaining_pct = batch["mRNA_remaining_pct"].to(self.device)
            
            outputs = self.model(siRNA, mrna, siRNA_concentration).squeeze()
            
            # Handle single sample batch case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            mse_loss = self.mse_loss(outputs, mRNA_remaining_pct)
            ranknet_loss = self.ranknet_loss(outputs, mRNA_remaining_pct)
            
            loss = self.lamda * ranknet_loss + (1 - self.lamda) * mse_loss
            
            running_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_truths.append(mRNA_remaining_pct.cpu())
            # Flatten gene names and IDs in batch
            all_gene_target_symbol_names.extend(gene_target_symbol_name)
            all_gene_target_ncbi_ids.extend(gene_target_ncbi_id)
            
            val_loops.set_postfix(loss=loss.item(), mse_loss=mse_loss.item(), ranknet_loss=ranknet_loss.item())
            
        # Check if there are prediction results
        if not all_preds:
            print("Warning: No prediction results!")
            if in_active_round:
                return 0.0, {}, 0.0, torch.tensor([]), torch.tensor([])
            else:
                return 0.0, {}, 0.0
            
        all_preds = torch.cat(all_preds)
        all_truths = torch.cat(all_truths)
        val_loss = running_loss / len(val_loader)
        precision_dict = topk_precision(all_preds, all_truths)
        spearman_corr = spearman_correlation(all_preds, all_truths)
        # Model needs to be set back to training mode due to dropout, but with gradient computation disabled
        self.model.train()
        if in_active_round:
            # Save the results
            results = pd.DataFrame({
                "gene_target_symbol_name": all_gene_target_symbol_names,
                "gene_target_ncbi_id": all_gene_target_ncbi_ids,
                "preds": all_preds.numpy(),
                "truths": all_truths.numpy()
            })
            if is_save:
                save_dir = os.path.join("out", f"active_{save_prefix}")
                os.makedirs(save_dir, exist_ok=True)
                results.to_csv(os.path.join(save_dir, f"active_round_{active_round}_test_pred.csv"), index=False)
            return val_loss, precision_dict, spearman_corr, all_preds, all_truths
        else:
            return val_loss, precision_dict, spearman_corr
        
    def get_embedding(self, sirna, mrna, siRNA_concentration):
        """Get sample embedding"""
        self.model.eval()
        with torch.no_grad():
            # Assume model has a get_embedding method or intermediate layer   
            embedding = self.model.get_embedding(sirna, mrna, siRNA_concentration)
        return embedding
    
    def qbc_sampling(self, poolset, num_samples, epoch=0):
        """Query-by-committee sampling strategy"""
        # Check if committee members exist
        if not self.committee_models:
            raise RuntimeError("QBC sampling method requires committee members, but committee_models is empty. Please check if use_qbc=True is set in configuration")
            
        print(f"Starting QBC sampling, pool size: {len(poolset)}, sampling count: {num_samples}")
        
        temp_loader = DataLoader(
            poolset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        
        all_predictions = []
        all_targets = []
        # Get predictions from each committee member
        with torch.no_grad():
            for i, model in enumerate(self.committee_models):
                model.eval()
                predictions = []
                print(f"Getting predictions from committee member {i+1}...")
                
                for batch in temp_loader:
                    siRNA = batch['sirna'].to(self.device)
                    mrna = batch['mrna'].to(self.device)
                    siRNA_concentration = batch['siRNA_concentration'].to(self.device).unsqueeze(1)
                    
                    outputs = model(siRNA, mrna, siRNA_concentration).squeeze()
                    
                    # Handle single sample batch case
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                        
                    predictions.append(outputs.cpu())
                    if i == 0:
                        all_targets.append(batch['mRNA_remaining_pct'].cpu())
                # Check if predictions is empty
                if not predictions:
                    raise RuntimeError(f"Committee member {i+1} did not produce any prediction results")
                    
                batch_predictions = torch.cat(predictions)
                batch_targets = torch.cat(all_targets)
                print(f"Committee member {i+1} prediction completed, prediction count: {len(batch_predictions)}")
                all_predictions.append(batch_predictions)
        
        # Check if all members have prediction results
        if not all_predictions:
            raise RuntimeError("No committee member produced prediction results")
            
        # Check if all prediction results have consistent lengths
        prediction_lengths = [len(pred) for pred in all_predictions]
        if len(set(prediction_lengths)) > 1:
            raise RuntimeError(f"Committee member prediction results have inconsistent lengths: {prediction_lengths}")
        
        # Calculate prediction variance
        predictions_stack = torch.stack(all_predictions)  # [n_models, n_samples]
        prediction_variance = torch.var(predictions_stack, dim=0)  # [n_samples]
        
        # Select samples with highest variance
        trainset_indices = torch.argsort(prediction_variance, descending=True)[:num_samples]
        print(f"QBC sampling completed, selected {len(trainset_indices)} samples")
        
        return trainset_indices
    
    def add_new_target(self, trainset: RNADataset, new_target_poolset: RNADataset, 
                   num_samples: int, approach: str = "random", active_round: int = 0, save_prefix: str = "run1") -> Tuple[RNADataset, RNADataset]:
        save_dir = os.path.join("out", f"active_{save_prefix}")
        os.makedirs(save_dir, exist_ok=True)
        
        if approach == "random": 
            trainset_indices = torch.randperm(len(new_target_poolset))[:num_samples]
        elif approach == "uncertainty":
            _, _, _, all_preds, all_truths = self.validate(
                new_target_poolset, in_active_round=True, active_round=active_round, save_prefix=save_prefix)
            uncertainty = abs(all_preds - 0.5)
            trainset_indices = torch.argsort(uncertainty, descending=False)[:num_samples]
        elif approach == "lowest":
            _, _, _, all_preds, all_truths = self.validate(
                new_target_poolset, in_active_round=True, active_round=active_round, save_prefix=save_prefix)
            uncertainty = all_preds
            trainset_indices = torch.argsort(uncertainty, descending=False)[:num_samples]
        elif approach == "highest":
            _, _, _, all_preds, all_truths = self.validate(
                new_target_poolset, in_active_round=True, active_round=active_round, save_prefix=save_prefix)
            uncertainty = all_preds
            trainset_indices = torch.argsort(uncertainty, descending=True)[:num_samples]
        elif approach == "qbc":
            trainset_indices = self.qbc_sampling(new_target_poolset, num_samples, epoch=active_round)
        else:
            raise ValueError(f"Unsupported approach: {approach}")
        
        partial_set = Subset(new_target_poolset, trainset_indices.tolist())
        new_trainset = ConcatDataset([trainset, partial_set])
        
        remaining_indices = list(set(range(len(new_target_poolset))) - set(trainset_indices.tolist()))
        updated_poolset = Subset(new_target_poolset, remaining_indices)
        
        # Save selected data subset
        selected_data = []
        for idx in trainset_indices.tolist():
            sample = new_target_poolset[idx]
            selected_data.append({
                'siRNA_antisense_seq_tensor': sample['sirna'],
                'gene_target_seq_tensor': sample['mrna'],
                'gene_target_symbol_name': sample['gene_target_symbol_name'],
                'gene_target_ncbi_id': sample['gene_target_ncbi_id'],
                'siRNA_concentration': sample['siRNA_concentration'].item(),
                'mRNA_remaining_pct': sample['mRNA_remaining_pct'].item()
            })
        
        df = pd.DataFrame(selected_data)
        df.to_csv(os.path.join(save_dir, f"selected_samples_active{active_round}.csv"), index=False)
        
        return new_trainset, updated_poolset
    
    def active_one_new_target(self, trainset, valset,
               new_target_poolset, new_target_testset,
               pretrain_epochs, active_train_epochs,
               num_samples_per_round, total_sample_rounds, 
               approach="uncertainty", save_prefix="run1"):
        """A script for active learning with one new target"""
        
        self.optimizer_pretrain = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate_pretrain"])
            
        # Calculate learning rate decay factor
        lr_start = self.config.get("learning_rate_active_start", 1e-4)
        lr_end = self.config.get("learning_rate_active_end", 1e-5)
        decay_factor = (lr_end / lr_start) ** (1 / total_sample_rounds)
        current_lr = lr_start
        
        # Initialize optimizer
        self.optimizer_active = optim.Adam(
            self.model.parameters(), lr=current_lr)
            
        # If using QBC, create optimizers for committee members
        if approach == "qbc" and self.committee_models:
            self.committee_optimizers_pretrain = [
                optim.Adam(model.parameters(), lr=self.config["learning_rate_pretrain"])
                for model in self.committee_models
            ]
            self.committee_optimizers_active = [
                optim.Adam(model.parameters(), lr=current_lr)
                for model in self.committee_models
            ]
        
        # Pretrain
        self.pretrain(self.optimizer_pretrain, trainset, 
                      valset, None, pretrain_epochs)
        
        print("-" * 50)
        print("Active Learning with one new target starts.")
        print("-" * 50)
        
        # Pre-active learning validation
        val_loss, precision_dict, spearman_corr, _, _ = self.validate(new_target_testset, in_active_round=True, active_round='before_active_training', is_save=True, save_prefix=save_prefix)
        print(f"Round before active, validation loss: {val_loss:.4f}, Spearman correlation: {spearman_corr:.4f}")
        print_topk_precision(precision_dict)
        print("-" * 50)
        
        best_spearman_corr = 0
        best_spearman_save_corr = 0
        best_model_path_name = ""
        best_state_dict = None
        
        # Active learning
        for each_round in range(total_sample_rounds):
            # Update learning rate
            current_lr *= decay_factor
            for param_group in self.optimizer_active.param_groups:
                param_group['lr'] = current_lr
            
            # If using QBC, also update committee members' learning rate
            if approach == "qbc" and len(self.committee_models) > 0:
                for optimizer in self.committee_optimizers_active:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
            
            print(f"Current learning rate: {current_lr:.2e}")
            
            # Select new targets
            trainset, new_target_poolset = self.add_new_target(trainset, new_target_poolset, num_samples_per_round, approach, active_round=each_round, save_prefix=save_prefix)
            print("Active Round: ", each_round, "Trainset size: ", len(trainset), "New_target_poolset size: ", len(new_target_poolset))
            
            # Train main model and committee members
            train_loader = DataLoader(trainset, batch_size=self.config["batch_size"], shuffle=True, 
                                     pin_memory=True, num_workers=4)
            
            for epoch in range(active_train_epochs):
                # Train main model
                train_loss = self.pretrain_one_epoch(train_loader, self.optimizer_active)
                
                # Train committee members
                if approach == "qbc" and len(self.committee_models) > 0:
                    for i, (committee_model, committee_optimizer) in enumerate(
                        zip(self.committee_models, self.committee_optimizers_active)):
                        committee_train_loss = self.pretrain_committee_member(
                            committee_model, train_loader, committee_optimizer)
                        print(f"Active Round {each_round}, Epoch {epoch+1}, committee member {i+1} training loss: {committee_train_loss:.4f}")
            
            # Validation
            val_loss, precision_dict, spearman_corr, _, _ = self.validate(new_target_testset, in_active_round=True, active_round=each_round, is_save=True, save_prefix=save_prefix)
            print(f"New target Active Round {each_round+1}/{total_sample_rounds}, validation loss: {val_loss:.4f}, Spearman correlation: {spearman_corr:.4f}")
            print(f"----Trainset size: {len(trainset)}----")
            print_topk_precision(precision_dict)
            print("-" * 50)
            if spearman_corr > best_spearman_corr:
                best_spearman_corr = spearman_corr
                print(f"Round {each_round+1} is the best round, Spearman Correlation: {best_spearman_corr:.4f}")
                best_model_path_name = os.path.join("./out", f"active_learning_model_{self.model_name}_best_at_round{each_round}.pth")
                if abs(spearman_corr - best_spearman_save_corr) > 0.05 and self.config["save_model"]:
                    print("Saving...")
                    # Save main model
                    torch.save(self.model.state_dict(), best_model_path_name)
                    # If using QBC, also save committee members
                    if approach == "qbc" and len(self.committee_models) > 0:
                        for i, committee_model in enumerate(self.committee_models):
                            committee_path = os.path.join("./out", f"active_learning_model_{self.model_name}_committee{i+1}_best_at_round{each_round}.pth")
                            torch.save(committee_model.state_dict(), committee_path)
                    print("Save done. model in: ", best_model_path_name)
                    best_spearman_save_corr = spearman_corr
        
        return self.model