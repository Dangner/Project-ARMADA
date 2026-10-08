import ember
import os

# Define where our data lives
# We are in 'src', so data is one level up in 'data'
dataset_dir = "data/ember2018"

print(f"🚀 Starting vectorization process for: {dataset_dir}")
print("   (This converts heavy JSON text into efficient binary .dat files)")
print("   (This takes 5-10 minutes. Do not close the terminal.)")

try:
    # This function creates X_train.dat, y_train.dat, etc.
    ember.create_vectorized_features(dataset_dir, feature_version=2)
    
    print("\n✅ Success! Data is vectorized.")
    print(f"   Check {dataset_dir} for new .dat files.")

except Exception as e:
    print(f"\n❌ Error: {e}")
