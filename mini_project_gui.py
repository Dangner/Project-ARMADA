import tkinter as tk
from tkinter import ttk, messagebox
import lightgbm as lgb
import ember
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
import time

# --- CONFIGURATION ---
DATA_DIR = "../data/ember2018"
MODEL_PATH = "../models/ember_model_v2_robust.txt"

class MalwareScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ARMADA Threat Intelligence Scanner")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")

        # Load Data & Model (Background)
        self.model = None
        self.X_test = None
        self.y_test = None
        
        # UI LAYOUT
        self.create_header()
        self.create_main_panel()
        self.create_status_bar()
        
        # Start loading thread
        threading.Thread(target=self.load_resources, daemon=True).start()

    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#007acc", height=80)
        header_frame.pack(fill=tk.X)
        
        title = tk.Label(header_frame, text="🛡️ ARMADA AI SCANNER", 
                         font=("Helvetica", 24, "bold"), bg="#007acc", fg="white")
        title.pack(pady=20)

    def create_main_panel(self):
        # Left Panel (Controls)
        control_frame = tk.Frame(self.root, bg="#2d2d2d", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        lbl_instr = tk.Label(control_frame, text="Scanner Controls", 
                             font=("Arial", 14), bg="#2d2d2d", fg="#00cc66")
        lbl_instr.pack(pady=20)
        
        self.btn_scan = tk.Button(control_frame, text="SCAN RANDOM FILE", 
                                  font=("Arial", 12, "bold"), bg="#444", fg="white",
                                  state=tk.DISABLED, command=self.scan_file, height=2)
        self.btn_scan.pack(fill=tk.X, padx=20, pady=10)

        # Right Panel (Visualization)
        self.viz_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.lbl_result = tk.Label(self.viz_frame, text="System Ready.", 
                                   font=("Courier", 18, "bold"), bg="#1e1e1e", fg="gray")
        self.lbl_result.pack(pady=50)

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing System...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                              bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#333", fg="white")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_resources(self):
        try:
            self.status_var.set("Loading AI Brain...")
            if os.path.exists(MODEL_PATH):
                self.model = lgb.Booster(model_file=MODEL_PATH)
            else:
                messagebox.showerror("Error", "Model file not found! Run Phase 4 first.")
                return

            self.status_var.set("Connecting to Database...")
            # Lazy load test data
            self.X_test = ember.read_vectorized_features(DATA_DIR, "test")
            if isinstance(self.X_test, tuple): self.X_test = self.X_test[0]
            
            metadata = ember.read_metadata(DATA_DIR)
            self.y_test = metadata[metadata['subset'] == 'test']['label']
            
            self.status_var.set("Ready. Database Online.")
            self.btn_scan.config(state=tk.NORMAL, bg="#007acc")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def scan_file(self):
        if self.X_test is None: return
        
        # 1. Pick a random file
        idx = np.random.randint(0, len(self.y_test))
        # Ensure it's valid (not -1)
        while self.y_test.iloc[idx] == -1:
            idx = np.random.randint(0, len(self.y_test))

        sample_x = self.X_test[idx]
        true_label = self.y_test.iloc[idx]
        
        # 2. Predict
        prob_malware = self.model.predict([sample_x])[0]
        
        # 3. Update UI
        self.update_viz(prob_malware, true_label, idx)

    def update_viz(self, prob, true_label, idx):
        # Clear old plot
        for widget in self.viz_frame.winfo_children():
            if widget != self.lbl_result: widget.destroy()

        # Text Result
        if prob > 0.5:
            res_text = "⚠️ THREAT DETECTED"
            res_color = "#ff4444" # Red
        else:
            res_text = "✅ FILE IS SAFE"
            res_color = "#00cc66" # Green
            
        truth_text = "(Actual: Malware)" if true_label == 1 else "(Actual: Benign)"
        self.lbl_result.config(text=f"{res_text}\n{truth_text}", fg=res_color)
        
        # Gauge Chart
        fig, ax = plt.subplots(figsize=(5, 2), facecolor="#1e1e1e")
        
        # Bar
        ax.barh([0], [prob], color=res_color, height=0.5)
        ax.barh([0], [1.0], color="#333", height=0.5, zorder=0) # Background bar
        
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50%", "100%"], color="white")
        ax.set_title(f"AI Confidence Score: {prob:.2%}", color="white")
        
        # Remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_visible(False)
        ax.set_facecolor("#1e1e1e")

        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)
        
        self.status_var.set(f"Scanned File ID: {idx} | Analysis Complete")

# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MalwareScannerApp(root)
    root.mainloop()
