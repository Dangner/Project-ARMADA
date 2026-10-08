import ember
import numpy as np
import pandas as pd

data_dir = "../data/ember2018"

print("🕵️  Diagnosing the training data...")

# Load Metadata
metadata = ember.read_metadata(data_dir)
y_train = metadata[metadata['subset'] == 'train']['label']
y_train = y_train.reset_index(drop=True)

# Replicate what we did in the broken training script
valid_rows_mask = (y_train != -1)
valid_indices = np.where(valid_rows_mask)[0]
limit = 50000
subset_indices = valid_indices[:limit]
y_subset = y_train.iloc[subset_indices]

# Count classes
num_benign = (y_subset == 0).sum()
num_malware = (y_subset == 1).sum()

print(f"\n📊 Training Subset Analysis:")
print(f"   - Benign Samples:  {num_benign}")
print(f"   - Malware Samples: {num_malware}")

if num_benign == 0 or num_malware == 0:
    print("\n❌ CRITICAL FAILURE: The training set has only ONE class.")
    print("   The model cannot learn to distinguish good from bad.")
else:
    print("\n✅ Data looks okay... something else is wrong.")
