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
    将输入的DataFrame按照mRNA_remaining_pct字段的分布，划分为9条训练数据的trainset，
    剩余数据按80%/20%划分为poolset和testset。
    
    参数:
    df (pd.DataFrame): 输入的DataFrame，必须包含mRNA_remaining_pct字段
    random_state (int): 随机种子，用于确保结果可重现
    
    返回:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 划分后的trainset、poolset和testset
    """
    # 检查输入DataFrame是否包含必要的字段
    if 'mRNA_remaining_pct' not in df.columns:
        raise ValueError("输入的DataFrame中缺少'mRNA_remaining_pct'字段")
    
    # 使用分层抽样，根据mRNA_remaining_pct进行分箱
    df['bin'] = pd.qcut(df['mRNA_remaining_pct'], 10, duplicates='drop')
    
    # 首先抽取9条训练数据
    trainset, remaining = train_test_split(
        df,
        train_size=100,
        stratify=df['bin'],
        random_state=random_state
    )
    
    # 对剩余数据进行80/20分层划分
    remaining['bin'] = pd.qcut(remaining['mRNA_remaining_pct'], 10, duplicates='drop')
    poolset, testset = train_test_split(
        remaining,
        test_size=0.2,
        stratify=remaining['bin'],
        random_state=random_state
    )
    
    # 移除临时添加的分箱列
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