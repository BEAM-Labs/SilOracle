import torch
from torch import nn, optim
from tqdm import tqdm
from scripts.criteria import topk_precision, print_topk_precision, \
                                spearman_correlation, RankNetLoss

class Siloracle_Trainer:
    # Trainer class for training and validating models
    """
    config example:
        config = {
            "model_name": "siRNA_model",
            "batch_size": 256,
            "embed_dim_siRNA": 32,
            "embed_dim_mrna": 256,
            "embed_dim": 128,
            "num_layers": 6,
            "nhead": 8,
            "lamda": 0.5,
            "num_epochs": 100,
            "save_path": "./out",
            "save_model": True,
        }
    """
    
    def __init__(self, model, config, device: torch.device="cuda"):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        self.mse_loss = nn.MSELoss().to(device)
        self.ranknet_loss = RankNetLoss().to(device)
        self.config = config
        # processing config
        self.lamda = config["lamda"]
        self.model_name = config["model_name"]
        self.batch_size = config["batch_size"]
        self.embed_dim_siRNA = config["embed_dim_siRNA"]
        self.embed_dim_mrna = config["embed_dim_mrna"]
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.nhead = config["nhead"]
        self.model_save_path = config["model_save_path"]
        self.is_save_model = config["is_save_model"]
        self.num_epochs = config["num_epochs"]
        self.device = device
        
        
    # Training  
    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        train_loops = tqdm(train_loader)
        
        for i, batch in enumerate(train_loops):
            siRNA = batch["sirna"].to(self.device)
            mrna = batch["mrna"].to(self.device)
            siRNA_concentration = batch["siRNA_concentration"].to(self.device).unsqueeze(1)
            mRNA_remaining_pct = batch["mRNA_remaining_pct"].to(self.device)
            
            # Optimization step
            self.optimizer.zero_grad()
            outputs = self.model(siRNA, mrna, siRNA_concentration).squeeze()
            mse_loss = self.mse_loss(outputs, mRNA_remaining_pct)
            ranknet_loss = self.ranknet_loss(outputs, mRNA_remaining_pct)
            loss = self.lamda * mse_loss + (1 - self.lamda) * ranknet_loss
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            train_loops.set_postfix(loss=loss.item(), mse_loss=mse_loss.item(), 
                                    ranknet_loss=ranknet_loss.item())
            running_loss += loss.item()
            
        return running_loss / len(train_loader)
    
    
    @torch.no_grad()
    def validate(self, val_loader, return_preds=False, model_path=None):
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        running_loss = 0.0
        
        val_loops = tqdm(val_loader)
        
        all_preds = []
        all_truths = []
        
        for i, batch in enumerate(val_loops):
            siRNA = batch["sirna"].to(self.device)
            mrna = batch["mrna"].to(self.device)
            siRNA_concentration = batch["siRNA_concentration"].to(self.device).unsqueeze(1)
            mRNA_remaining_pct = batch["mRNA_remaining_pct"].to(self.device)
            
            outputs = self.model(siRNA, mrna, siRNA_concentration).squeeze()
            mse_loss = self.mse_loss(outputs, mRNA_remaining_pct)
            ranknet_loss = self.ranknet_loss(outputs, mRNA_remaining_pct)
            
            loss = self.lamda * mse_loss + (1 - self.lamda) * ranknet_loss
            
            running_loss += loss.item()
            
            all_preds.append(outputs.cpu())
            all_truths.append(mRNA_remaining_pct.cpu())
            
            val_loops.set_postfix(loss=loss.item(), mse_loss=mse_loss.item(), 
                                  ranknet_loss=ranknet_loss.item())
            
        pred_values = torch.cat(all_preds).numpy()
        true_values = torch.cat(all_truths).numpy()
        precision_dict = topk_precision(pred_values, true_values)
        spearman_corr = spearman_correlation(torch.from_numpy(pred_values), torch.from_numpy(true_values))
        self.model.train()
        if return_preds:
            return running_loss / len(val_loader), precision_dict, spearman_corr, pred_values, true_values
        else:
            return running_loss / len(val_loader), precision_dict, spearman_corr
        
    
    def train(self, train_loader, val_loader):
        # Pre-training validation
        val_loss, precision_dict, spearman_corr = self.validate(val_loader)
        print("-" * 50)
        print(f"Pre-training validation loss: {val_loss:.4f}, Spearman correlation: {spearman_corr:.4f}")
        print_topk_precision(precision_dict)
        print("-" * 50)
        
        best_spearman_corr = 0
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, precision_dict, spearman_corr = self.validate(val_loader)
            print("-" * 50)
            print(f"Epoch {epoch+1} training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}, Spearman correlation: {spearman_corr:.4f}")
            print_topk_precision(precision_dict)
            print("-" * 50)
            
            if abs(spearman_corr) > abs(best_spearman_corr):
                best_spearman_corr = spearman_corr
                torch.save(self.model.state_dict(), f"{self.model_save_path}/{self.model_name}_best.pth")
                print(f"New best model achieved! Model saved to {self.model_save_path}/{self.model_name}_best.pth")
        