import os
import numpy as np
from tqdm import tqdm
import shutil

def merge_folder(folder):
    if not os.path.exists(folder): return
    mix_files = [f for f in os.listdir(folder) if f.endswith('_mix.npz')]
    print(f"Merging {len(mix_files)} songs in {folder}...")
    
    for f in tqdm(mix_files):
        base_name = f.replace('_mix.npz', '')
        mix_path = os.path.join(folder, f)
        guitar_path = os.path.join(folder, base_name + '_guitar.npz')
        
        if not os.path.exists(guitar_path): continue
            
        try:
            mix = np.load(mix_path)['arr_0']
            guitar = np.load(guitar_path)['arr_0']
            
            # Ensure 2D (Channels, Samples)
            if mix.ndim == 1: mix = mix[np.newaxis, :]
            if guitar.ndim == 1: guitar = guitar[np.newaxis, :]
            
            # --- FIX: FORCE EQUAL LENGTH ---
            min_len = min(mix.shape[1], guitar.shape[1])
            mix = mix[:, :min_len]
            guitar = guitar[:, :min_len]
            
            # Stack into (2, Channels, Samples)
            combined = np.stack([mix, guitar], axis=0)
            
            new_path = os.path.join(folder, base_name + '.npz')
            np.savez_compressed(new_path, arr_0=combined.astype('float32'))
            
            # Cleanup
            os.remove(mix_path)
            os.remove(guitar_path)
            for stem in ['_vocals.npz', '_drums.npz', '_bass.npz']:
                dummy = os.path.join(folder, base_name + stem)
                if os.path.exists(dummy): os.remove(dummy)
        except Exception as e:
            print(f"Error merging {base_name}: {e}")

# 1. Run Merge
merge_folder('data/train_8')
merge_folder('data/test_8')

print("\nAll files merged and stages recreated. Ready to train!")