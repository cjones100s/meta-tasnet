import os
import numpy as np
from tqdm import tqdm

def restack_folder(folder_path):
    if not os.path.exists(folder_path):
        return
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    for f in tqdm(files, desc=f"Restacking {os.path.basename(folder_path)}"):
        path = os.path.join(folder_path, f)
        data = np.load(path)['arr_0']
        
        # data is likely (2, Length) for Stereo or (1, Length) for Mono
        # Model wants [Left, Right, Mono] stacked vertically
        if data.shape[0] == 2:
            left = data[0:1, :]
            right = data[1:2, :]
            mono = np.mean(data, axis=0, keepdims=True)
            # Stack into (3, 1, Length) or (3, Length) based on model expectation
            # The error 'array is 2-dimensional' suggests it's currently (Instrument, Time)
            # We need to make it (3, Time) or similar.
            stacked = np.concatenate([left, right, mono], axis=0)
        else:
            # If already mono, just triple it to satisfy the 3-index requirement
            stacked = np.concatenate([data, data, data], axis=0)
            
        np.savez_compressed(path, arr_0=stacked.astype('float32'))

# Run for all stage folders
folders = ['data/train_44', 'data/test_44']
for folder in folders:
    restack_folder(folder)
print("Done! Try training now.")