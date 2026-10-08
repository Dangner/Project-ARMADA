import ember
import os

# Point to your data
data_dir = "../data/ember2018"

print(f"🚀 Generating metadata.csv for: {data_dir}")
print("   (This scans the JSON files to build an index. It takes about 1-2 minutes.)")

try:
    # This function creates the missing metadata.csv
    ember.create_metadata(data_dir)
    print("\n✅ Success! metadata.csv created.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
