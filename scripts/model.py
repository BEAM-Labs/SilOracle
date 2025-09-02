import torch
import torch.nn as nn
import torch.nn.functional as F
import math
        
        
class PositionalEncoding(nn.Module):
    """Positional encoding module"""
    def __init__(self, d_model, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Register as non-parameter tensor
        self.register_buffer('pe', pe.to(device))
        self.device = device

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class InputProjection(nn.Module):
    """Project arbitrary dimension input to model dimension"""
    def __init__(self, input_dim, d_model, device='cuda'):
        super(InputProjection, self).__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.device = device
        self.to(device)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.to(self.device)
        return self.projection(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder"""
    def __init__(self, input_dim, embed_dim, num_layers=6, nhead=8, dim_feedforward=1024, 
                 dropout=0.1, max_seq_length=256, activation="relu", device='cuda'):
        super(TransformerEncoder, self).__init__()
        
        # Save configuration
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.d_model = embed_dim  # Transformer internal processing dimension
        self.max_seq_length = max_seq_length
        self.device = device
        
        # Input projection layer, convert arbitrary input dimension to d_model
        self.input_projection = InputProjection(input_dim, self.d_model, device=device)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=max_seq_length, device=device)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (if need to adjust output dimension)
        if self.d_model != embed_dim:
            self.output_projection = nn.Linear(self.d_model, embed_dim)
        else:
            self.output_projection = nn.Identity()
            
        # Initialize parameters
        self._init_parameters()
        
        # Move model to specified device
        self.to(device)
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input sequence [batch_size, seq_len, input_dim]
            src_mask: Sequence mask (optional)
            src_key_padding_mask: Padding mask (optional)
        Returns:
            output: Output embedding [batch_size, embed_dim]
        """
        # Ensure input is on correct device
        src = src.to(self.device)
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(self.device)
            
        # Project to model dimension
        x = self.input_projection(src)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Through transformer encoder
        memory = self.transformer_encoder(x, mask=src_mask, 
                                          src_key_padding_mask=src_key_padding_mask)
        # Get sequence representation (using average pooling)
        output = torch.mean(memory, dim=1)
        # Output projection to desired embedding dimension
        output = self.output_projection(output)
        
        return output


class TransformerVectorEncoder(nn.Module):
    """Transformer model for encoding single vectors (non-sequences)"""
    def __init__(self, input_dim, embed_dim, num_layers=6, nhead=8, 
                 dim_feedforward=2048, dropout=0.1, activation="relu", device='cuda'):
        super(TransformerVectorEncoder, self).__init__()
        
        self.device = device
        
        # Since Transformer usually processes sequences, we can treat a single vector as a sequence of length 1
        # Or expand it to a small sequence (using the first approach here)
        self.transformer = TransformerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=1,
            activation=activation,
            device=device
        )
    
    def forward(self, x):
        """
        Args:
            x: Input vector [batch_size, input_dim]
        Returns:
            output: Output embedding [batch_size, embed_dim]
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Expand vector to sequence dimension
        # Comment out for non-embedding version
        # x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Through transformer encoding
        output = self.transformer(x)  # [batch_size, embed_dim]

        return output
    
    @torch.no_grad()
    def get_embedding(self, siRNA, mrna, siRNA_concentration):
        siRNA = siRNA.to(self.device)
        mrna = mrna.to(self.device)
        siRNA_concentration = siRNA_concentration.to(self.device)
        
        siRNA_embedding = self.siRNA_encoder(siRNA)
        mrna_embedding = self.mRNA_encoder(mrna)
        
        siRNA_mrna_embedding = torch.cat((siRNA_embedding, mrna_embedding, siRNA_concentration), dim=1)
        
        embedding = self.mlp[0](siRNA_mrna_embedding)
        return embedding


class Siloracle(nn.Module):
    """Transformer model for encoding siRNA sequences"""
    def __init__(self, input_sirna_dim, input_mrna_dim, embed_dim, num_layers=6, nhead=8, 
                 dim_feedforward=2048, dropout=0.1, activation="relu", device='cuda',
                 siRNA_vocab_size=9, mrna_vocab_size=512):
        super(Siloracle, self).__init__()
        
        self.device = device
        
        self.siRNA_embedding = nn.Embedding(siRNA_vocab_size, embed_dim, device=device)
        self.mRNA_embedding = nn.Embedding(mrna_vocab_size, embed_dim, device=device)
        
        self.siRNA_encoder = TransformerVectorEncoder(
            input_sirna_dim, 
            embed_dim, 
            num_layers, 
            nhead, 
            dim_feedforward, 
            dropout, 
            activation,
            device=device
        )
        
        self.mRNA_encoder = TransformerVectorEncoder(
            input_mrna_dim, 
            embed_dim, 
            num_layers, 
            nhead, 
            dim_feedforward, 
            dropout, 
            activation,
            device=device
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Move MLP to specified device
        self.mlp.to(device)
        
    def forward(self, siRNA, mrna, siRNA_concentration):
        # Ensure inputs are on correct device
        # Note: If using nn.Embedding version, please modify the commented part in TransformerVectorEncoder function
        siRNA = self.siRNA_embedding(siRNA.to(self.device))
        mrna = self.mRNA_embedding(mrna.to(self.device))
        
        siRNA_concentration = siRNA_concentration.to(self.device)
        
        siRNA_embedding = self.siRNA_encoder(siRNA)
        mrna_embedding = self.mRNA_encoder(mrna)
        
        siRNA_mrna_embedding = torch.cat((siRNA_embedding, mrna_embedding, siRNA_concentration), dim=1)
        output = self.mlp(siRNA_mrna_embedding)
        return output
    
    @torch.no_grad()
    def get_embedding(self, siRNA, mrna, siRNA_concentration):
        siRNA = self.siRNA_embedding(siRNA.to(self.device))
        
        mrna = self.mRNA_embedding(mrna.to(self.device))
        siRNA_concentration = siRNA_concentration.to(self.device)
        
        siRNA_embedding = self.siRNA_encoder(siRNA)
        mrna_embedding = self.mRNA_encoder(mrna)
        
        siRNA_mrna_embedding = torch.cat((siRNA_embedding, mrna_embedding, siRNA_concentration), dim=1)
        
        embedding = self.mlp[0](siRNA_mrna_embedding)
        return embedding
    
    
