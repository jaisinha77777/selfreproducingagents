import torch

# Load both model state dicts
model_3layer = torch.load('model_3layer.pkl')
model_4layer = torch.load('model_4layer.pkl')

print("="*50)
print("3-Layer Model Parameters")
print("="*50)

params_3layer = []
for name, param in model_3layer.items():
    print(f"{name}: shape {param.shape}, {param.numel()} parameters")
    params_3layer.append(param.flatten().tolist())

# Flatten all into single list
all_params_3layer = []
for param_list in params_3layer:
    all_params_3layer.extend(param_list)

print(f"\nTotal parameters in 3-layer model: {len(all_params_3layer)}")
print(f"First 10 parameters: {all_params_3layer[:10]}")

print("\n" + "="*50)
print("4-Layer Model Parameters")
print("="*50)

params_4layer = []
for name, param in model_4layer.items():
    print(f"{name}: shape {param.shape}, {param.numel()} parameters")
    params_4layer.append(param.flatten().tolist())

# Flatten all into single list
all_params_4layer = []
for param_list in params_4layer:
    all_params_4layer.extend(param_list)

print(f"\nTotal parameters in 4-layer model: {len(all_params_4layer)}")
print(f"First 10 parameters: {all_params_4layer[:10]}")

print("\n" + "="*50)
print("Summary")
print("="*50)
print(f"3-layer model has {len(all_params_3layer)} total parameters")
print(f"4-layer model has {len(all_params_4layer)} total parameters")
#test
for i in all_params_3layer:
    print(i)