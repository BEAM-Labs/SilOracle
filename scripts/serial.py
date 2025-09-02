import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional
import pickle
import tqdm
import pandas as pd
import concurrent.futures
import math

class RNATokenizer:
    """
    RNA sequence tokenizer for converting RNA sequences to token IDs
    """
    
    def __init__(self, vocab_file: str):
        """
        Initialize RNA tokenizer
        
        Args:
            vocab_file: Vocabulary file path (JSON format)
        """
        # Read base vocabulary
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        
        # Add special tokens
        special_tokens = {
            "<pad>": 0,
            "<mask>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<unk>": 4
        }
        
        self.vocab.update(special_tokens)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Sort tokens by length from large to small for longest matching
        self.tokens_by_length = sorted(
            [t for t in self.vocab.keys() if t not in special_tokens],
            key=len, reverse=True
        )
        
        self.special_tokens = special_tokens
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def encode_single_base(self, rna_seq: str) -> List[int]:
        """
        Encode RNA sequence using single base tokens
        
        Args:
            rna_seq: RNA sequence
            
        Returns:
            List of token IDs
        """
        rna_seq = rna_seq.replace('U', 'T')
        return [self.vocab.get(base, self.vocab["<unk>"]) for base in rna_seq]
    
    def encode_longest_match(self, rna_seq: str) -> List[int]:
        """
        Encode RNA sequence using longest matching approach
        
        Args:
            rna_seq: RNA sequence
            
        Returns:
            List of token IDs
        """
        result = []
        i = 0
        rna_seq = rna_seq.replace('U', 'T')
        
        while i < len(rna_seq):
            matched = False
            
            # Try longest matching
            for token in self.tokens_by_length:
                if rna_seq[i:].startswith(token):
                    result.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            
            # If no match found, use single character as token
            if not matched:
                result.append(self.vocab.get(rna_seq[i], self.vocab["<unk>"]))
                i += 1
        
        return result
    
    def pad_or_truncate(self, token_ids: List[int], max_length: int, 
                         add_special_tokens: bool = False) -> List[int]:
        """
        Pad or truncate token ID list to specified length
        
        Args:
            token_ids: List of token IDs
            max_length: Target length
            add_special_tokens: Whether to add special tokens (<bos> and <eos>)
            
        Returns:
            Processed token ID list
        """
        original_max_length = max_length
        
        if add_special_tokens:
            # Reserve space for special tokens
            available_length = max_length - 2
            
            # Truncate
            if len(token_ids) > available_length:
                token_ids = token_ids[:available_length]
                
            # Add special tokens
            token_ids = [self.vocab["<bos>"]] + token_ids + [self.vocab["<eos>"]]
        else:
            # Truncate
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
        
        # Padding
        pad_length = original_max_length - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [self.vocab["<pad>"]] * pad_length
        
        return token_ids
        
    def encode_sequence_batch(self, sequences: List[str], 
                             encoder_func_name: str, 
                             max_length: int, 
                             add_special_tokens: bool = False,
                             num_threads: int = 4) -> List[torch.Tensor]:
        """
        Encode a batch of sequences using multithreading
        
        Args:
            sequences: List of RNA sequences
            encoder_func_name: Encoder function name, can be 'encode_single_base' or 'encode_longest_match'
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens
            num_threads: Number of threads
            
        Returns:
            List of encoded token ID tensors
        """
        # Select encoding function
        if encoder_func_name == 'encode_single_base':
            encoder_func = self.encode_single_base
        elif encoder_func_name == 'encode_longest_match':
            encoder_func = self.encode_longest_match
        else:
            raise ValueError(f"Unknown encoding function: {encoder_func_name}")
        
        # Calculate number of sequences per thread
        batch_size = len(sequences)
        batch_per_thread = math.ceil(batch_size / num_threads)
        
        results = [None] * batch_size
        
        # Create a shared progress bar
        pbar = tqdm.tqdm(total=batch_size, desc=f"Encoding RNA sequences ({encoder_func_name})")
        
        # Define task for each thread to execute
        def process_batch(start_idx, end_idx):
            local_results = []
            for i in range(start_idx, min(end_idx, batch_size)):
                # tokens = encoder_func(sequences[i])
                tokens = encoder_func(sequences[i].replace('U', 'T'))
                tokens = self.pad_or_truncate(tokens, max_length, add_special_tokens)
                local_results.append((i, torch.tensor(tokens, dtype=torch.int)))
                # Update progress bar
                pbar.update(1)
            return local_results
        
        try:
            # Use thread pool for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(0, batch_size, batch_per_thread):
                    futures.append(executor.submit(process_batch, i, i + batch_per_thread))
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    for idx, tensor in future.result():
                        results[idx] = tensor
        finally:
            # Ensure progress bar is properly closed
            pbar.close()
        
        return results

class RNADataset(Dataset):
    """
    RNA pair dataset, pre-encode all sequences to improve data loading speed
    Note: Corresponds to RNADatasetEmb class in original code
    """
    
    def __init__(self, 
                 siRNA: List[str], 
                 mrna: List[str],
                 siRNA_concentration: List[float],
                 gene_target_symbol_name: List[str],
                 gene_target_ncbi_id: List[str],
                 mRNA_remaining_pct: np.ndarray,
                 tokenizer: RNATokenizer,
                 sirna_max_length: int = 32,
                 mrna_max_length: int = 256,
                 precompute: bool = True,
                 cache_dir: Optional[str] = './datacache',
                 cache_name: Optional[str] = None,
                 num_threads: int = 4):
        """
        Initialize RNA dataset
        
        Args:
            siRNA: List of siRNA sequences
            mrna: List of mRNA sequences
            siRNA_concentration: List of siRNA concentrations
            gene_target_symbol_name: List of gene target symbol names
            gene_target_ncbi_id: List of gene target NCBI IDs
            mRNA_remaining_pct: Array of mRNA remaining percentages
            tokenizer: RNA tokenizer
            sirna_max_length: Maximum length of siRNA sequences
            mrna_max_length: Maximum length of mRNA sequences
            precompute: Whether to precompute all encodings, default is True
            cache_dir: Cache directory, if specified will try to load/save encoding data from this directory
            cache_name: Cache file name, default is None, will be auto-generated based on data characteristics
            num_threads: Number of threads for encoding, only effective when precompute=True
        """
        self.siRNA = siRNA
        self.mrna = mrna
        self.siRNA_concentration = siRNA_concentration
        self.gene_target_symbol_name = gene_target_symbol_name
        self.gene_target_ncbi_id = gene_target_ncbi_id
        self.tokenizer = tokenizer
        self.sirna_max_length = sirna_max_length
        self.mrna_max_length = mrna_max_length
        self.mRNA_remaining_pct = mRNA_remaining_pct
        self.num_threads = num_threads
        
        # Add mRNA encoding cache dictionary
        self.mrna_encoding_cache = {}
        
        # Encoding cache related parameters
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        # If cache_name is not specified, auto-generate based on dataset characteristics
        if cache_dir and cache_name is None:
            import hashlib
            # Generate unique hash value using dataset size and max length parameters
            hash_input = f"{len(siRNA)}_{sirna_max_length}_{mrna_max_length}"
            cache_name = f"rna_dataset_{hashlib.md5(hash_input.encode()).hexdigest()[:10]}.pkl"
        
        self.cache_path = os.path.join(cache_dir, cache_name) if cache_dir and cache_name else None
        
        # Initialize encoding data attributes
        self.encoded_sirna = None
        self.encoded_mrna = None
        
        # Try to load encoding data from cache
        if precompute:
            if self.cache_path and os.path.exists(self.cache_path):
                self._load_encodings_from_cache()
            else:
                self.precompute_encodings()
                # If cache path is specified, save encoding data
                if self.cache_path:
                    self.save_encodings_to_cache()
            
    def precompute_encodings(self):
        """Precompute encodings for all RNA sequences using multithreading, with caching mechanism for mRNA"""
        print("Starting RNA sequence encoding precomputation...")
        
        # Encode siRNA sequences using multithreading
        print(f"Using {self.num_threads} threads to encode {len(self.siRNA)} siRNA sequences...")
        self.encoded_sirna = self.tokenizer.encode_sequence_batch(
            sequences=self.siRNA, 
            encoder_func_name='encode_single_base', 
            max_length=self.sirna_max_length,
            add_special_tokens=True,
            num_threads=self.num_threads
        )
        
        # Encode mRNA sequences using caching mechanism
        print(f"Starting to encode {len(self.mrna)} mRNA sequences (using caching mechanism)...")
        self.encoded_mrna = []
        unique_mrna_count = 0
        cache_hit_count = 0
        
        with tqdm.tqdm(total=len(self.mrna), desc="Encoding mRNA sequences") as pbar:
            for mrna_seq in self.mrna:
                if mrna_seq in self.mrna_encoding_cache:
                    # Cache hit
                    self.encoded_mrna.append(self.mrna_encoding_cache[mrna_seq])
                    cache_hit_count += 1
                else:
                    # Cache miss, need to encode
                    tokens = self.tokenizer.encode_longest_match(mrna_seq)
                    tokens = self.tokenizer.pad_or_truncate(
                        tokens, self.mrna_max_length, add_special_tokens=True
                    )
                    encoded = torch.tensor(tokens, dtype=torch.long)
                    self.mrna_encoding_cache[mrna_seq] = encoded
                    self.encoded_mrna.append(encoded)
                    unique_mrna_count += 1
                pbar.update(1)
        
        print(f"✓ mRNA encoding completed!")
        print(f"  - Unique mRNA sequences: {unique_mrna_count}")
        print(f"  - Cache hits: {cache_hit_count}")
        print(f"  - Cache hit rate: {cache_hit_count/len(self.mrna)*100:.2f}%")
        
    def save_encodings_to_cache(self, cache_path: Optional[str] = None):
        """
        Save encoded data to local cache
        
        Args:
            cache_path: Cache file path, if None will use the path specified during initialization
        """
        if cache_path is None:
            cache_path = self.cache_path
            
        if cache_path is None:
            print("Cache path not specified, cannot save encoding data")
            return
            
        if self.encoded_sirna is None or self.encoded_mrna is None:
            print("Encoding data does not exist, please call precompute_encodings() method first")
            return
            
        print(f"Saving encoding data to {cache_path}...")
        
        # Create directory (if it doesn't exist)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Save encoding data and mRNA encoding cache
        cache_data = {
            'encoded_sirna': self.encoded_sirna,
            'encoded_mrna': self.encoded_mrna,
            'mrna_encoding_cache': self.mrna_encoding_cache,  # Add mRNA encoding cache
            'sirna_max_length': self.sirna_max_length,
            'mrna_max_length': self.mrna_max_length,
            'data_count': len(self.siRNA)
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Encoding data successfully saved, containing {len(self.siRNA)} RNA pairs")
            print(f"  - mRNA encoding cache size: {len(self.mrna_encoding_cache)} unique sequences")
        except Exception as e:
            print(f"Error occurred while saving encoding data: {str(e)}")
        
    def _load_encodings_from_cache(self):
        """Load encoding data from cache"""
        print(f"Loading encoding data from cache {self.cache_path}...")
        
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Verify if cache data matches current dataset
            if cache_data.get('data_count') != len(self.siRNA):
                print("Warning: Cache data does not match current dataset size, recomputing encodings...")
                self.precompute_encodings()
                return
                
            self.encoded_sirna = cache_data.get('encoded_sirna')
            self.encoded_mrna = cache_data.get('encoded_mrna')
            self.mrna_encoding_cache = cache_data.get('mrna_encoding_cache', {})  # Load mRNA encoding cache
            
            print(f"Successfully loaded {len(self.encoded_sirna)} pre-encoded data")
            print(f"  - mRNA encoding cache size: {len(self.mrna_encoding_cache)} unique sequences")
            
        except Exception as e:
            print(f"Error loading cache data: {str(e)}")
            print("Recomputing encodings...")
            self.precompute_encodings()
    
    def __len__(self) -> int:
        return len(self.siRNA)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.encoded_sirna is not None:
            # If encodings are pre-computed, return directly
            return {
                "sirna": self.encoded_sirna[idx],
                "mrna": self.encoded_mrna[idx],
                "siRNA_concentration": torch.tensor(self.siRNA_concentration[idx], dtype=torch.float),
                "mRNA_remaining_pct": torch.tensor(self.mRNA_remaining_pct[idx], dtype=torch.float),
                "gene_target_symbol_name": self.gene_target_symbol_name[idx],
                "gene_target_ncbi_id": self.gene_target_ncbi_id[idx]
            }
        else:
            # If not pre-computed, encode in real-time
            sirna, mrna, siRNA_concentration = self.siRNA[idx], self.mrna[idx], self.siRNA_concentration[idx]
            
            sirna_tokens = self.tokenizer.encode_single_base(sirna)
            sirna_tokens = self.tokenizer.pad_or_truncate(
                sirna_tokens, self.sirna_max_length, add_special_tokens=True
            )
            
            mrna_tokens = self.tokenizer.encode_longest_match(mrna)
            mrna_tokens = self.tokenizer.pad_or_truncate(
                mrna_tokens, self.mrna_max_length, add_special_tokens=True
            )
            
            return {
                "sirna": torch.tensor(sirna_tokens, dtype=torch.long),
                "mrna": torch.tensor(mrna_tokens, dtype=torch.long),
                "siRNA_concentration": torch.tensor(siRNA_concentration, dtype=torch.float),
                "mRNA_remaining_pct": torch.tensor(self.mRNA_remaining_pct[idx], dtype=torch.float),
                "gene_target_symbol_name": self.gene_target_symbol_name[idx],
                "gene_target_ncbi_id": self.gene_target_ncbi_id[idx]
            }
            
   
def load_rna_dataset(
    csv_file_train: str,
    csv_file_val: str,
    csv_file_test: str,
    tokenizer: RNATokenizer,
    sirna_max_length: int = 32,
    mrna_max_length: int = 256,
    precompute: bool = True,
    cache_name: Dict[str, str] = None,
) -> Tuple[RNADataset, RNADataset, RNADataset]:
    """
    Read data directly from CSV files, return training set, validation set and test set
    
    Args:
        csv_file_train: Training set CSV file path
        csv_file_val: Validation set CSV file path
        csv_file_test: Test set CSV file path
        tokenizer: RNA tokenizer (can be RNATokenizer or RNATokenizer_Kmer)
        sirna_max_length: Maximum length of siRNA sequences
        mrna_max_length: Maximum length of mRNA sequences
        precompute: Whether to precompute encodings
        cache_name: Cache file name dictionary, containing "train", "val", "test" keys
        use_kmer: Whether to use kmer tokenizer
        
    Returns:
        Training set, validation set and test set
    """
    # Verify if tokenizer type matches use_kmer parameter
    if not isinstance(tokenizer, RNATokenizer):
        # This may be a redundant operation, but is necessary for
        # unpredicted changes in the code.
        raise ValueError("tokenizer must be of RNATokenizer type")
    
    # Read data
    df_train = pd.read_csv(csv_file_train)
    df_val = pd.read_csv(csv_file_val)
    df_test = pd.read_csv(csv_file_test)
    
    siRNA_train = df_train["siRNA_antisense_seq"].tolist()
    mrna_train = df_train["gene_target_seq"].tolist()
    siRNA_concentration_train = df_train["siRNA_concentration"].tolist()
    gene_target_symbol_name_train = df_train["gene_target_symbol_name"].tolist()
    gene_target_ncbi_id_train = df_train["gene_target_ncbi_id"].tolist()
    
    siRNA_val = df_val["siRNA_antisense_seq"].tolist()
    mrna_val = df_val["gene_target_seq"].tolist()
    siRNA_concentration_val = df_val["siRNA_concentration"].tolist()
    gene_target_symbol_name_val = df_val["gene_target_symbol_name"].tolist()
    gene_target_ncbi_id_val = df_val["gene_target_ncbi_id"].tolist()
    
    siRNA_test = df_test["siRNA_antisense_seq"].tolist()
    mrna_test = df_test["gene_target_seq"].tolist()
    siRNA_concentration_test = df_test["siRNA_concentration"].tolist()
    gene_target_symbol_name_test = df_test["gene_target_symbol_name"].tolist()
    gene_target_ncbi_id_test = df_test["gene_target_ncbi_id"].tolist()
    
    mRNA_remaining_pct_train = df_train["mRNA_remaining_pct"].to_numpy()
    mRNA_remaining_pct_val = df_val["mRNA_remaining_pct"].to_numpy()
    mRNA_remaining_pct_test = df_test["mRNA_remaining_pct"].to_numpy()
    
    # Return datasets
    return RNADataset(siRNA_train, mrna_train, 
                       siRNA_concentration_train, 
                       gene_target_symbol_name_train, 
                       gene_target_ncbi_id_train, 
                       mRNA_remaining_pct_train, 
                       tokenizer, sirna_max_length, mrna_max_length, 
                       precompute, cache_name=cache_name["train"]), \
           RNADataset(siRNA_val, mrna_val, 
                       siRNA_concentration_val, 
                       gene_target_symbol_name_val, 
                       gene_target_ncbi_id_val, 
                       mRNA_remaining_pct_val, 
                       tokenizer, sirna_max_length, mrna_max_length, 
                       precompute, cache_name=cache_name["val"]), \
           RNADataset(siRNA_test, mrna_test, 
                       siRNA_concentration_test, 
                       gene_target_symbol_name_test, 
                       gene_target_ncbi_id_test, 
                       mRNA_remaining_pct_test, 
                       tokenizer, sirna_max_length, mrna_max_length, 
                       precompute, cache_name=cache_name["test"])