import ember
import lightgbm as lgb
import numpy as np
import os
import pandas as pd

# 1. Setup
# Note: We use ".." because we are inside the 'src' folder
data_dir = "../data/ember2018"
print(f"🚀 Loading vectorized data from: {data_dir}")

try:
    X_train = ember.read_vectorized_features(data_dir, "train")
    if isinstance(X_train, tuple):
        X_train = X_train[0]

    metadata = ember.read_metadata(data_dir)
    y_train = metadata[metadata['subset'] == 'train']['label']
    y_train = y_train.reset_index(drop=True)

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# 2. FILTER & SHUFFLE (The Critical Fix)
print("   Filtering for valid samples...")
valid_rows_mask = (y_train != -1)
valid_indices = np.where(valid_rows_mask)[0]

# FIX: Shuffle the indices so we get a mix of classes
print("   🎲 Shuffling data to mix Benign and Malware...")
np.random.seed(42) # Ensures we get the same random mix every time
np.random.shuffle(valid_indices)

# LIMIT: Still 50,000 for RAM safety, but now it's a RANDOM 50,000
limit = 50000 
print(f"⚠️ SUPER-LITE MODE: Training on random {limit} samples.")
subset_indices = valid_indices[:limit]

# 3. Load subset
print("   Loading subset into memory...")
X_subset = X_train[subset_indices]
y_subset = y_train.iloc[subset_indices]

# CHECK: Print the balance to be sure
n_benign = (y_subset == 0).sum()
n_malware = (y_subset == 1).sum()
print(f"   📊 Class Balance: {n_benign} Benign / {n_malware} Malware")

# 4. Train
print("🧠 Training LightGBM Model... (This takes ~1 min)")
lgbm_dataset = lgb.Dataset(X_subset, y_subset)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    "verbose": 0
}

model = lgb.train(params, lgbm_dataset, num_boost_round=50)

# 5. Save
if not os.path.exists("../models"):
    os.makedirs("../models")

model.save_model("../models/ember_model_v1.txt")
print("\n✅ SUCCESS! Fixed Model Saved.")
