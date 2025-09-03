import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

gene_1_name = 'KHK'
gene_2_name = 'F5'

gene_1_path = f'./siloactive_{gene_1_name}.csv'
gene_2_path = f'./siloactive_{gene_2_name}.csv'

gene_1 = pd.read_csv(gene_1_path)
gene_2 = pd.read_csv(gene_2_path)

def split_dataset(df: pd.DataFrame, random_state: int = 42):
    """
    Split the input DataFrame according to the distribution of the 'mRNA_remaining_pct' field:
    select 100 samples as the trainset, and split the remaining data into poolset and testset in an 80%/20% ratio.
    
    Args:
    df (pd.DataFrame): Input DataFrame, must contain the 'mRNA_remaining_pct' field
    random_state (int): Random seed to ensure reproducibility
    
    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The split trainset, poolset, and testset
    """
    # Check if the input DataFrame contains the required field
    if 'mRNA_remaining_pct' not in df.columns:
        raise ValueError("The input DataFrame is missing the 'mRNA_remaining_pct' field")
    
    # Use stratified sampling, binning by 'mRNA_remaining_pct'
    df['bin'] = pd.qcut(df['mRNA_remaining_pct'], 10, duplicates='drop')
    
    # First, select 100 samples for the training set
    trainset, remaining = train_test_split(
        df,
        train_size=100,
        stratify=df['bin'],
        random_state=random_state
    )
    remaining['bin'] = pd.qcut(remaining['mRNA_remaining_pct'], 10, duplicates='drop')
    poolset, testset = train_test_split(
        remaining,
        test_size=0.2,
        stratify=remaining['bin'],
        random_state=random_state
    )
    
    trainset = trainset.drop(columns=['bin'])
    poolset = poolset.drop(columns=['bin'])
    testset = testset.drop(columns=['bin'])
    
    return trainset, poolset, testset

gene1_train, gene1_pool, gene1_test = split_dataset(gene_1)
gene2_train, gene2_pool, gene2_test = split_dataset(gene_2)


gene1_train.to_csv(f'./siloactive_{gene_1_name}_train.csv', index=False)
gene1_pool.to_csv(f'./siloactive_{gene_1_name}_pool.csv', index=False)
gene1_test.to_csv(f'./siloactive_{gene_1_name}_test.csv', index=False)

gene2_train.to_csv(f'./siloactive_{gene_2_name}_train.csv', index=False)
gene2_pool.to_csv(f'./siloactive_{gene_2_name}_pool.csv', index=False)
gene2_test.to_csv(f'./siloactive_{gene_2_name}_test.csv', index=False)

print(f"Gene {gene_1_name} train size: {len(gene1_train)}")
print(f"Gene {gene_1_name} pool size: {len(gene1_pool)}")
print(f"Gene {gene_1_name} test size: {len(gene1_test)}")

print(f"Gene {gene_2_name} train size: {len(gene2_train)}")
print(f"Gene {gene_2_name} pool size: {len(gene2_pool)}")
print(f"Gene {gene_2_name} test size: {len(gene2_test)}")
