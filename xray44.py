import numpy as np

# Load one of your processed 8k files
data = np.load('data/train_44/028c6f6a-f4f6-4795-84f2-a81222b38e7e.npz')['arr_0']

print(f"Final Shape: {data.shape}")
print(f"Data Type:   {data.dtype}")
