import torch

# Create an example voxel grid
voxel_grid = torch.rand(8, 8, 8)  # Random values in an 8x8x8 grid

# Flatten the voxel grid
flat_voxel_grid = voxel_grid.flatten()

# Compute downsampled indices for each voxel
# Here we map each voxel to its new position in a 4x4x4 downsampled grid
indices = torch.arange(0, voxel_grid.numel())
downsampled_indices = indices // 2 % 4 + (indices // 2 // 4 % 4) * 4 + (indices // 2 // 4 // 4 % 4) * 16

# Prepare the output tensor for the downsampled grid
downsampled_grid = torch.zeros(4 * 4 * 4, dtype=torch.float)

# Use scatter_reduce to downsample by averaging
downsampled_grid = torch.scatter_reduce(downsampled_grid, 0, downsampled_indices, flat_voxel_grid, reduce="mean")

# Reshape back to 3D
downsampled_grid = downsampled_grid.view(4, 4, 4)
