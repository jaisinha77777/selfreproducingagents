import torch

# Load the pickle file
state_dict = torch.load('model.pkl')

# Get all parameters as a list
params_list = [param for param in state_dict.values()]

# Or if you want just the tensors flattened
params_flat = [param.flatten() for param in state_dict.values()]

# Print shapes
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")