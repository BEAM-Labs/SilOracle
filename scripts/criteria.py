# Evaluation metrics
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Dict 

class RankNetLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RankNetLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred:   Tensor of shape (B,) - predicted values (raw or sigmoid)
        target: Tensor of shape (B,) - ground truth (e.g., 1 - mRNA_remaining_pct)
        """
        B = pred.shape[0]

        # Only consider target < 1.0
        valid_mask = (target < 1.0).float()
        if valid_mask.sum() <= 1:
            # Not enough valid samples to form pairs
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Compute pairwise differences
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)        # (B, B)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)  # (B, B)

        # Only use pairs where target[i] > target[j] and both > 0
        valid_pair_mask = (target_diff > 0) & (target.unsqueeze(1) < 1.0) & (target.unsqueeze(0) < 1.0)
        valid_pair_mask = valid_pair_mask.float()  # (B, B)

        # Apply mask and compute RankNet loss
        loss_matrix = -F.logsigmoid(pred_diff) * valid_pair_mask  # shape: (B, B)

        if self.reduction == 'mean':
            denom = valid_pair_mask.sum()
            if denom == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            return loss_matrix.sum() / denom

        elif self.reduction == 'sum':
            return loss_matrix.sum()

        else:  # 'none'
            return loss_matrix

def pairwise_ranking_loss(y_pred: torch.Tensor, y_true: torch.Tensor, group_ids: torch.Tensor):
    loss = 0.0
    count = 0
    # Process y_true: values less than 0.3 are positive samples
    y_true = (y_true <= 0.3).float()
    # group_ids: ID for same mRNA
    unique_groups = torch.unique(group_ids)
    for g in unique_groups:
        idx = (group_ids == g)
        y_p, y_t = y_pred[idx], y_true[idx]
        for i in range(len(y_p)):
            for j in range(len(y_p)):
                if y_t[i] > y_t[j]:
                    loss += F.binary_cross_entropy_with_logits(
                        y_p[i] - y_p[j], torch.ones_like(y_p[i]))
                    count += 1
    return loss / (count + 1e-8)

def spearman_correlation(x, y):
    """
    Calculate Spearman correlation coefficient between two one-dimensional vectors
    
    Args:
        x: Tensor of shape (batch_size)
        y: Tensor of shape (batch_size), same dimension as x
        
    Returns:
        spearman_corr: Scalar representing the Spearman correlation coefficient between the two vectors
    """
    # Ensure inputs are one-dimensional
    assert x.dim() == 1 and y.dim() == 1, "Input vectors must be one-dimensional"
    assert x.shape == y.shape, "Both input vectors must have the same length"
    
    n = x.shape[0]
    
    # Calculate ranks
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    
    # Calculate means
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()
    
    # Calculate covariance
    cov = torch.sum((x_rank - x_mean) * (y_rank - y_mean))
    
    # Calculate variance
    x_var = torch.sum((x_rank - x_mean) ** 2)
    y_var = torch.sum((y_rank - y_mean) ** 2)
    
    # Calculate correlation coefficient
    denominator = torch.sqrt(x_var * y_var)
    
    # Handle division by zero
    if denominator > 0:
        spearman_corr = cov / denominator
    else:
        spearman_corr = torch.tensor(0.0, device=x.device)
    
    return spearman_corr

def calculate_mse(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def topk_precision(pred_values: Union[np.ndarray, torch.Tensor, List[float]], 
                   true_values: Union[np.ndarray, torch.Tensor, List[float]],
                   k_values: List[int] = [10, 50, 100, 200],
                   threshold: float = 0.3) -> Dict[int, float]:
    """
    Calculate Top-K precision.
    
    Args:
        pred_values: List of predicted values
        true_values: List of true values
        k_values: List of k values to calculate
        threshold: Threshold for determining positive samples, values below this are positive samples
    
    Returns:
        Dictionary with k values as keys and corresponding precision as values
    """
    # Convert to numpy arrays for processing
    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.detach().cpu().numpy()
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()

    pred_values = np.array(pred_values)
    true_values = np.array(true_values)
    
    # Ensure input data has the same length
    assert len(pred_values) == len(true_values), "Predicted values and true values must have the same length"
    
    # Get indices sorted by predicted values from low to high
    sorted_indices = np.argsort(pred_values)  # Ascending order
    # Get true values after sorting
    sorted_true_values = true_values[sorted_indices]
    
    # Calculate precision for each k value
    precision_dict = {}
    for k in k_values:
        # Ensure k doesn't exceed data length
        if k > len(pred_values):
            print(f"Warning: k value ({k}) exceeds data length ({len(pred_values)}), will use all available data")
            # k = len(pred_values)
        
        # Get true values corresponding to top k predictions
        topk_true = sorted_true_values[:k]
        
        # Calculate how many of the top k predictions are positive samples (true values below threshold)
        num_positive = np.sum(topk_true < threshold)
        
        # Calculate precision
        precision = num_positive / k
        
        # Save result
        precision_dict[k] = precision
    
    return precision_dict

def print_topk_precision(precision_dict: Dict[int, float]) -> None:
    """
    Print top-k precision results
    
    Args:
        precision_dict: Dictionary returned by topk_precision function
    """
    print("Top-K Precision Results:")
    for k, precision in sorted(precision_dict.items()):
        print(f"Top-{k}: {precision:.4f} ({int(precision * k)}/{k})")

def example_usage():
    """Usage example"""
    # Generate some random data
    np.random.seed(42)
    num_samples = 100
    
    # Generate random predicted values and true values
    pred_values = np.random.random(num_samples)
    true_values = np.random.random(num_samples)
    
    # Calculate top-k precision
    precision_dict = topk_precision(pred_values, true_values)
    
    # Print results
    print_topk_precision(precision_dict)
