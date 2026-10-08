import ember
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. Setup
data_dir = "../data/ember2018"
model_path = "../models/ember_model_v1.txt"

print(f"🚀 Loading Test Data from: {data_dir}")

try:
    # Load the Test Data (The model has NEVER seen these files)
    X_test = ember.read_vectorized_features(data_dir, "test")
    if isinstance(X_test, tuple):
        X_test = X_test[0]

    metadata = ember.read_metadata(data_dir)
    y_test = metadata[metadata['subset'] == 'test']['label']
    y_test = y_test.reset_index(drop=True)

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# 2. Filter Unlabeled Data
# Just like training, we remove the "-1" rows
print("   Filtering unlabeled test samples...")
valid_rows_mask = (y_test != -1)
valid_indices = np.where(valid_rows_mask)[0]

X_test = X_test[valid_indices]
y_test = y_test.iloc[valid_indices]

print(f"   Testing on {len(y_test)} files.")

# 3. Load Your Trained Brain
print(f"🧠 Loading Model: {model_path}")
model = lgb.Booster(model_file=model_path)

# 4. Predict!
print("   Running predictions... (Finding the malware)")
y_pred = model.predict(X_test)

# 5. Calculate Score
# ROC-AUC is the standard score for security. 
# 0.5 = Guessing randomly
# 1.0 = Perfect God-mode
score = roc_auc_score(y_test, y_pred)

print("\n" + "="*40)
print(f"🏆 FINAL RESULTS")
print(f"   ROC-AUC Score: {score:.4f}")
print("="*40)

if score > 0.90:
    print("✅ Conclusion: The model is HIGHLY effective.")
elif score > 0.80:
    print("⚠️ Conclusion: Good, but needs improvement.")
else:
    print("❌ Conclusion: The model is weak. We need more data.")
