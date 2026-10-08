import ember
import lightgbm as lgb
import numpy as np
import pandas as pd
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Setup
data_dir = "../data/ember2018"
model_path = "../models/ember_model_v2_robust.txt"

print("🚀 Phase 3: Initiating Adversarial Attack Simulation (Universal Fix)...")

# 2. Load Data & Model
try:
    print("   Loading model and test data...")
    model = lgb.Booster(model_file=model_path)
    
    # Load test data
    X_test = ember.read_vectorized_features(data_dir, "test")
    if isinstance(X_test, tuple): X_test = X_test[0]
    
    metadata = ember.read_metadata(data_dir)
    y_test = metadata[metadata['subset'] == 'test']['label']
    
    # Filter for MALWARE samples only
    malware_indices = np.where(y_test == 1)[0]
    X_malware = X_test[malware_indices]
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# 3. Find a "Victim" Sample
print("   Hunting for a confident malware detection to attack...")
target_idx = -1
for i in range(100):
    prob = model.predict([X_malware[i]])[0]
    if prob > 0.90:
        target_idx = i
        print(f"   🎯 Target Found! Sample #{i} | Model Confidence: {prob:.4f}")
        break

if target_idx == -1:
    print("❌ Could not find a confident malware sample to attack.")
    exit()

victim_feature_vector = X_malware[target_idx]

# 4. Set up the Universal Attacker
print("\n⚔️  Launching HopSkipJump Attack...")

# WRAPPER FUNCTION (The Fix):
# LightGBM binary gives 1 number (Prob of Malware).
# ART expects 2 numbers (Prob of Benign, Prob of Malware).
def predict_wrapper(x):
    # Get probability of being malware
    prob_malware = model.predict(x)
    # Calculate probability of being benign
    prob_benign = 1 - prob_malware
    # Stack them: [Benign, Malware]
    return np.column_stack((prob_benign, prob_malware))

# Create the Classifier using BlackBox (Works for ANY model)
art_classifier = BlackBoxClassifier(
    predict_fn=predict_wrapper,
    input_shape=(2381,),
    nb_classes=2,
    clip_values=(0, 255) # Byte values are 0-255
)

# Define the attack
attack = HopSkipJump(classifier=art_classifier, targeted=False, max_iter=20, verbose=False)

# 5. Run the Attack
# Check original score first
original_prob = model.predict([victim_feature_vector])[0]
print(f"   Original Score (Before Attack): {original_prob:.4f} (Malware)")

# Generate adversarial example
print("   Generatng noise (This takes 30-60 seconds)...")
x_adv = attack.generate(x=np.array([victim_feature_vector]))

# 6. Verify Success
attacked_prob = model.predict(x_adv)[0]
print(f"   Adversarial Score (After Attack): {attacked_prob:.4f}")

print("\n📊 FINAL RESULT:")
if attacked_prob < 0.50:
    print("✅ SUCCESS: Evasion Complete!")
    print("   We tweaked the bytes, and now the AI thinks this Malware is SAFE.")
else:
    print("❌ FAILED: The model resisted the attack.")
