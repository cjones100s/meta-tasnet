import numpy as np
import librosa

def confirm_npz_mapping(npz_path, mix_wav_path, guitar_wav_path, sr=44100):
    # 1. Load the Stacked NPZ
    # Current shape is (2, 2, T) according to your xray result
    npz_data = np.load(npz_path)['arr_0']
    
    # 2. Load the original WAVs exactly as the model would
    # Forced mono to match the intended model input
    mix_wav, _ = librosa.load(mix_wav_path, sr=sr, mono=True)
    guitar_wav, _ = librosa.load(guitar_wav_path, sr=sr, mono=True)

    # 3. Handle Stereo-to-Mono averaging in the verification step
    # Your NPZ is (2, 2, T), so we average the second dimension for checking
    npz_mix_mono = np.mean(npz_data[0, :, :], axis=0)
    npz_guitar_mono = np.mean(npz_data[1, :, :], axis=0)

    # 4. Trim to match the shortest length
    min_len = min(len(npz_mix_mono), len(mix_wav), len(guitar_wav))
    
    # 5. Correlation Check (using np.allclose for float precision)
    mix_match = np.allclose(npz_mix_mono[:min_len], mix_wav[:min_len], atol=1e-3)
    guitar_match = np.allclose(npz_guitar_mono[:min_len], guitar_wav[:min_len], atol=1e-3)

    print(f"--- Verification Results ---")
    print(f"File: {npz_path}")
    print(f"Mix (Index 0) matches Mixture.wav: {'✅ YES' if mix_match else '❌ NO'}")
    print(f"Guitar (Index 1) matches Guitar.wav: {'✅ YES' if guitar_match else '❌ NO'}")

# Use the paths from your Blue Hive directory
confirm_npz_mapping(
    'data/train_44/028c6f6a-f4f6-4795-84f2-a81222b38e7e.npz', 
    'mixture.wav', 
    'guitar.wav'
)
