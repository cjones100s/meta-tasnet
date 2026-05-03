import numpy as np

# Load one of your processed 8k files
data = np.load('data/train_8/BigTroubles_Phantom.npz')['arr_0']

print(f"Final Shape: {data.shape}")
print(f"Data Type:   {data.dtype}")