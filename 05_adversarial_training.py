import ember
import lightgbm as lgb
import numpy as np
import pandas as pd
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump
import warnings
import os

warnings.filterwarnings('ignore')

# 1. Setup
data_dir = "../data/ember2018"
old_model_path = "../models/ember_model_v1.txt"
new_model_path = "../models/ember_model_v2_robust.txt"

print("🚀 Phase 4: Starting Adversarial Training (The Vaccine)...")

# 2. Load Data & Old Model
try:
    print("   Loading data...")
    X_train = ember.read_vectorized_features(data_dir, "train")
    if isinstance(X_train, tuple): X_train = X_train[0]
    
    metadata = ember.read_metadata(data_dir)
    y_train = metadata[metadata['subset'] == 'train']['label']
    
    # Filter valid data (Limit to 50k for RAM, just like before)
    valid_rows = (y_train != -1)
    indices = np.where(valid_rows)[0]
    np.random.seed(42)
    np.random.shuffle(indices)
    limit = 50000
    X_train_lite = X_train[indices[:limit]]
    y_train_lite = y_train.iloc[indices[:limit]]
    
    print("   Loading old brain...")
    model = lgb.Booster(model_file=old_model_path)
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# 3. Generate "Vaccine" Data
# We will take 20 malware samples and turn them into adversarial examples.
# (In a real project, you would do this for thousands of samples, but that takes days).
print("\n🧪 Generating Adversarial Examples (This is slow, please wait)...")

# Find malware samples in our subset
malware_indices = np.where(y_train_lite == 1)[0][:20] 
X_malware_samples = X_train_lite[malware_indices]

# Setup the Attacker (Same as Phase 3)
def predict_wrapper(x):
    prob_malware = model.predict(x)
    prob_benign = 1 - prob_malware
    return np.column_stack((prob_benign, prob_malware))

art_classifier = BlackBoxClassifier(
    predict_fn=predict_wrapper,
    input_shape=(2381,),
    nb_classes=2,
    clip_values=(0, 255)
)

attack = HopSkipJump(classifier=art_classifier, targeted=False, max_iter=10, verbose=False)

# Generate the fake files
# This creates 20 "Hard to Detect" malware samples
X_adv = attack.generate(x=X_malware_samples)
y_adv = np.ones(len(X_adv)) # Label them as MALWARE (1)

print(f"   ✅ Generated {len(X_adv)} adversarial samples.")

# 4. Mix and Retrain
print("\n💉 Injecting vaccine into training data...")
# Stack old data + new tricky data
X_new_train = np.vstack((X_train_lite, X_adv))
y_new_train = np.concatenate((y_train_lite, y_adv))

print(f"   New Training Size: {len(X_new_train)} samples")

print("🧠 Retraining Model (Robust Version)...")
lgbm_dataset = lgb.Dataset(X_new_train, y_new_train)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": 0
}

model_v2 = lgb.train(params, lgbm_dataset, num_boost_round=50)

# 5. Save
model_v2.save_model(new_model_path)
print(f"\n✅ SUCCESS! Robust Model Saved: {new_model_path}")
