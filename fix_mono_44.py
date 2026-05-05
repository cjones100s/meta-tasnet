import os
import numpy as np
from tqdm import tqdm

def fix_mono_with_backup(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"Created new directory: {dst_folder}")

    files = [f for f in os.listdir(src_folder) if f.endswith('.npz')]
    print(f"Processing {len(files)} files from {src_folder} into {dst_folder}...")

    for f in tqdm(files):
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f)
        
        # Load the original (2, 2, T)
        data = np.load(src_path)['arr_0']
        
        if data.ndim == 3 and data.shape[1] == 2:
            # Average channels (axis 1): (2, 2, T) -> (2, 1, T)
            data = np.mean(data, axis=1, keepdims=True)
            
            # Save to the NEW directory
            np.savez_compressed(dst_path, arr_0=data.astype('float32'))
        else:
            print(f"Skipping {f}: Unexpected shape {data.shape}")

if __name__ == "__main__":
    # Run for train and test separately
    fix_mono_with_backup("data/train_44_backup_stereo", "data/train_44")
    fix_mono_with_backup("data/test_44_backup_stereo", "data/test_44")
    print("\nMono conversion complete! Original files are safe.")
