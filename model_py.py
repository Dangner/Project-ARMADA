import torch
# 1. Define or import the EXACT network architecture class used to train the file
# Note: Replace 'ArmadaModelClass' with your actual model class name
from your_project_module import ArmadaModelClass  

model = ArmadaModelClass()

# 2. Load the checkpoint file (pointing to the correct subfolder)
checkpoint = torch.load('armada_output/armada_checkpoint.pt', map_location='cpu')

# 3. Extract the clean model state dictionary from the container
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    # If the file contains only the raw state dict
    model.load_state_dict(checkpoint)

# 4. Ready the model for evaluation/testing
model.eval()

# 5. Pass dummy input data through the model
dummy_input = torch.randn(1, 3, 224, 224) 

with torch.no_grad():
    predictions = model(dummy_input)

print("Model output shape:", predictions.shape)