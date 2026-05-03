import torch
import torchaudio
import numpy as np

def check_pipeline(mix_wav, guitar_wav, npz_file):
    # 1. Load the original WAVs
    mix_orig, sr_m = torchaudio.load(mix_wav)
    gui_orig, sr_g = torchaudio.load(guitar_wav)
    
    # 2. Load the transformed NPZ
    npz_data = np.load(npz_file)['arr_0'] # Expected shape: (2, 1, T)
    
    print(f"--- Pipeline Audit ---")
    print(f"Original SR: {sr_m}Hz")
    print(f"NPZ Shape:   {npz_data.shape}")
    
    # Check 1: Resampling (for 8kHz run)
    expected_samples = int(mix_orig.shape[1] * (8000 / sr_m))
    print(f"Expected Samples at 8k: ~{expected_samples}")
    print(f"Actual NPZ Samples:      {npz_data.shape[2]}")
    
    # Check 2: Mono Conversion
    if npz_data.shape[1] == 1:
        print("[SUCCESS] NPZ is Mono.")
    else:
        print(f"[ERROR] NPZ is still Stereo ({npz_data.shape[1]} channels)!")

    # Check 3: Content Match (Basic energy check)
    # If normalized correctly, max should be near 1.0 or std-based
    print(f"NPZ Mix Max:    {np.abs(npz_data[0]).max():.4f}")
    print(f"NPZ Guitar Max: {np.abs(npz_data[1]).max():.4f}")

# Example usage:
check_pipeline('../../Guitar_Separation/train/Cayetana_MissThing/mixture.wav', '../../Guitar_Separation/train/Cayetana_MissThing/guitar.wav', 'data/train_8/Cayetana_MissThing.npz')