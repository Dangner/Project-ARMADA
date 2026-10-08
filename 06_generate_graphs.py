import matplotlib.pyplot as plt
import numpy as np
import os  # <--- This is what was missing!

# Data from your actual experiments
labels = ['Clean Data', 'Under Attack']

# Model 1 (The Weak One - Before Vaccine)
# It dropped from ~94% to 18%
model_v1_scores = [0.9387, 0.1800] 

# Model 2 (ARMADA - The Strong One - After Vaccine)
# It stayed at ~94% even when attacked
model_v2_scores = [0.9394, 0.9394]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# Plot the bars
rects1 = ax.bar(x - width/2, model_v1_scores, width, label='Standard Model', color='#ff9999')
rects2 = ax.bar(x + width/2, model_v2_scores, width, label='ARMADA (Ours)', color='#66b3ff')

# Add text labels and titles
ax.set_ylabel('Detection Confidence (Probability)')
ax.set_title('Impact of Adversarial Attack: Standard vs. ARMADA')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Function to write the percentage on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Save the graph
if not os.path.exists("../output"):
    os.makedirs("../output")

save_path = "../output/result_graph.png"
plt.savefig(save_path)
print(f"✅ Graph saved to: {save_path}")
