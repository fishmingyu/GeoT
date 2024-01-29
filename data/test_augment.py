import torch
import torch.nn.functional as F

# Example 1D tensor
tensor_1d = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

# Reshape to [1, 1, length] - batch size = 1, channels = 1
tensor_3d = tensor_1d.unsqueeze(0).unsqueeze(0)

# Define new length for interpolation
new_length = 10

# Interpolate - mode can be 'linear', 'nearest', etc.
# Since we are dealing with 1D data, 'linear' is used here
interpolated = F.interpolate(tensor_3d, size=new_length, mode='linear')

# Reshape back to 1D
interpolated_1d = interpolated.squeeze()

print(interpolated_1d)
