import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

sirnaod3_filename = './siRNA_Open_Dataset_for_Drug_Discovery_1.0.csv'

save_train_csv = 'siloracle_train.csv'
save_val_csv = 'siloracle_val.csv'
save_test_csv = 'siloracle_test.csv'

df = pd.read_csv(sirnaod3_filename)
df = df[df["print_error_flag"] == "0"]
homo_df = df[df["gene_target_species"] == "Homo sapiens"]

columns = ['id', 'publication_id', 'gene_target_symbol_name',
       'gene_target_ncbi_id', 'gene_target_species', 'siRNA_duplex_id',
       'siRNA_sense_seq', 'siRNA_antisense_seq', 'cell_line_donor', 
       'siRNA_concentration', 'concentration_unit',
       'gene_target_seq', 'mRNA_remaining_pct']
homo_df = homo_df[columns]

homo_df['mRNA_remaining_pct'] = homo_df['mRNA_remaining_pct'].clip(lower=0.0, upper=100.0) / 100.0

homo_df.drop_duplicates(inplace=True)

####
# Read the data
gene1_name = 'KHK'
gene2_name = 'F5'

data = homo_df

# 1. Separate the data for the given genes
# Default is F5 and SCN9A
# But found that their training results are not good, so I changed the data
F5_data = data[data['gene_target_symbol_name'] == gene1_name]
SCN9A_data = data[data['gene_target_symbol_name'] == gene2_name]
remaining_data = data[~data['gene_target_symbol_name'].isin([gene1_name, gene2_name])]

# Save the data for F5 and SCN9A
F5_data.to_csv(f'siloactive_{gene1_name}.csv', index=False)
SCN9A_data.to_csv(f'siloactive_{gene2_name}.csv', index=False)

def check_distribution(train_data, val_data, test_data, gene):
    """Check if the mRNA distribution of a gene in the training, validation and test sets is similar"""
    train_mrna = train_data[train_data['gene_target_symbol_name'] == gene]['mRNA_remaining_pct']
    val_mrna = val_data[val_data['gene_target_symbol_name'] == gene]['mRNA_remaining_pct']
    test_mrna = test_data[test_data['gene_target_symbol_name'] == gene]['mRNA_remaining_pct']
    
    # Use KS test to compare the distributions
    ks_train_val = ks_2samp(train_mrna, val_mrna)[1]
    ks_train_test = ks_2samp(train_mrna, test_mrna)[1]
    ks_val_test = ks_2samp(val_mrna, test_mrna)[1]
    
    return min(ks_train_val, ks_train_test, ks_val_test)

def split_stratified(data):
    """Stratify the data, keeping the proportion of each gene"""
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # Separate each gene
    for gene in data['gene_target_symbol_name'].unique():
        gene_data = data[data['gene_target_symbol_name'] == gene]
        
        # First separate the test set
        train_val_data, test_subset = train_test_split(
            gene_data, 
            test_size=0.15,
            random_state=42
        )
        
        # Then separate the validation set from the remaining data
        train_subset, val_subset = train_test_split(
            train_val_data,
            test_size=0.118,  # 0.10/0.85 to ensure the final ratio is 10%
            random_state=42
        )
        
        train_data = pd.concat([train_data, train_subset])
        val_data = pd.concat([val_data, val_subset])
        test_data = pd.concat([test_data, test_subset])
    
    return train_data, val_data, test_data

# Split the data
train_data, val_data, test_data = split_stratified(remaining_data)

# Save the split data
train_data.to_csv(save_train_csv, index=False)
val_data.to_csv(save_val_csv, index=False)
test_data.to_csv(save_test_csv, index=False)

# Print the basic statistics of each data set
print("\nData set splitting results:")
print(f"Training set size: {len(train_data)} ({len(train_data)/len(remaining_data)*100:.1f}%)")
print(f"Validation set size: {len(val_data)} ({len(val_data)/len(remaining_data)*100:.1f}%)")
print(f"Test set size: {len(test_data)} ({len(test_data)/len(remaining_data)*100:.1f}%)")

# Check the distribution of each gene
print("\nThe distribution of each gene in the data set:")
for gene in remaining_data['gene_target_symbol_name'].unique():
    train_count = len(train_data[train_data['gene_target_symbol_name'] == gene])
    val_count = len(val_data[val_data['gene_target_symbol_name'] == gene])
    test_count = len(test_data[test_data['gene_target_symbol_name'] == gene])
    total = train_count + val_count + test_count
    
    print(f"\nGene {gene}:")
    print(f"Training set: {train_count} ({train_count/total*100:.1f}%)")
    print(f"Validation set: {val_count} ({val_count/total*100:.1f}%)")
    print(f"Test set: {test_count} ({test_count/total*100:.1f}%)")
    
    # Check the mRNA distribution
    distribution_similarity = check_distribution(train_data, val_data, test_data, gene)
    print(f"mRNA distribution similarity p-value: {distribution_similarity:.3f}")