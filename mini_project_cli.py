import ember
import lightgbm as lgb
import numpy as np
import os
import time
import sys
import random

# --- CONFIGURATION ---
DATA_DIR = "../data/ember2018"
MODEL_PATH = "../models/ember_model_v2_robust.txt"

# ANSI Colors for Terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def typing_effect(text, speed=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()

def load_resources():
    print(f"{CYAN}[SYSTEM]{RESET} Initializing ARMADA Core...")
    time.sleep(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}[ERROR]{RESET} Model not found at {MODEL_PATH}")
        sys.exit()
        
    print(f"{CYAN}[SYSTEM]{RESET} Loading Neural Network (LightGBM)...")
    model = lgb.Booster(model_file=MODEL_PATH)
    
    print(f"{CYAN}[SYSTEM]{RESET} Connecting to Vector Database...")
    # Lazy load to save RAM
    X_test = ember.read_vectorized_features(DATA_DIR, "test")
    if isinstance(X_test, tuple): X_test = X_test[0]
    
    metadata = ember.read_metadata(DATA_DIR)
    y_test = metadata[metadata['subset'] == 'test']['label']
    
    print(f"{GREEN}[SUCCESS]{RESET} System Online. Ready to Scan.")
    print("-" * 50)
    return model, X_test, y_test

def draw_gauge(prob):
    # Creates a text-based progress bar: [||||||||||     ] 75%
    bar_len = 30
    filled_len = int(bar_len * prob)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    
    color = GREEN
    if prob > 0.5: color = RED
    elif prob > 0.2: color = YELLOW
    
    print(f"Threat Level: {color}[{bar}] {prob:.2%}{RESET}")

def main():
    os.system('clear' if os.name == 'posix' else 'cls')
    print(f"{BOLD}{CYAN}")
    print(r"""
     _    ____  __  __    _    ____    _    
    / \  |  _ \|  \/  |  / \  |  _ \  / \   
   / _ \ | |_) | |\/| | / _ \ | | | |/ _ \  
  / ___ \|  _ <| |  | |/ ___ \| |_| / ___ \ 
 /_/   \_\_| \_\_|  |_/_/   \_\____/_/   \_\
    """)
    print(f"      Adversarially-Robust Malware Detector{RESET}")
    print("=" * 50)
    
    model, X_test, y_test = load_resources()
    
    while True:
        command = input(f"\n{BOLD}ARMADA >{RESET} Press [ENTER] to scan a file (or 'q' to quit): ")
        if command.lower() == 'q':
            print("Shutting down...")
            break
            
        print(f"\n{YELLOW}>> Selecting random file from stream...{RESET}")
        time.sleep(0.5)
        
        # Pick random file
        idx = np.random.randint(0, len(y_test))
        while y_test.iloc[idx] == -1: # Skip unlabeled
            idx = np.random.randint(0, len(y_test))
            
        # Scan
        print(f"{YELLOW}>> Analyzing Features...{RESET}")
        time.sleep(0.5)
        
        features = X_test[idx]
        prob = model.predict([features])[0]
        actual_label = y_test.iloc[idx]
        
        # Result
        draw_gauge(prob)
        
        if prob > 0.5:
            print(f"{RED}{BOLD}🚨 MALWARE DETECTED!{RESET}")
        else:
            print(f"{GREEN}{BOLD}✅ FILE IS SAFE.{RESET}")
            
        truth = "Malware" if actual_label == 1 else "Benign"
        print(f"   (Ground Truth Verification: {truth})")
        print("-" * 30)

if __name__ == "__main__":
    main()
