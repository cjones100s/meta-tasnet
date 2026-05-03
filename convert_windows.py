import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

def convert_audio(input_root, output_root, sample_rate=8000):
    """
    Converts mix/guitar and creates dummy stems (vocals, drums, bass)
    to satisfy the Meta-TasNet requirements.
    """
    os.makedirs(output_root, exist_ok=True)
    song_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
    
    print(f"Found {len(song_folders)} songs in {input_root}")

    for song in tqdm(song_folders, desc="Processing"):
        song_path = os.path.join(input_root, song)
        
        # The script expects 5 specific components
        # We use 'guitar.wav' as the source for all target stems
        target_files = {
            "mix": "mixture.wav",
            "guitar": "guitar.wav",
            "vocals": "guitar.wav", 
            "drums": "guitar.wav",
            "bass": "guitar.wav"
        }

        # Verify essential files exist
        if not (os.path.exists(os.path.join(song_path, "mixture.wav")) and 
                os.path.exists(os.path.join(song_path, "guitar.wav"))):
            continue

        for key, filename in target_files.items():
            filepath = os.path.join(song_path, filename)
            try:
                # Load audio
                audio, sr = torchaudio.load(filepath)
                
                # Resample if not 8kHz
                if sr != sample_rate:
                    audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
                
                # Meta-TasNet REQUIRES .npz format with 'arr_0' key
                output_path = os.path.join(output_root, f"{song}_{key}.npz")
                np.savez_compressed(output_path, arr_0=audio.numpy())
            except Exception as e:
                print(f"Error processing {song}/{key}: {e}")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    # Use 'r' and the folder names you renamed (train_8 / test_8)
    TRAIN_WAVS_PATH = r"C:\Users\cjone\Guitar_Separation\train" 
    TEST_WAVS_PATH = r"C:\Users\cjone\Guitar_Separation\test"
    # ---------------------------

    # We save to data/train_44 and data/test_44 as required by the trainer
    convert_audio(TRAIN_WAVS_PATH, os.path.join("data", "train_8"))
    convert_audio(TEST_WAVS_PATH, os.path.join("data", "test_8"))
    
    print("\nConversion Complete! You can now run the training command.")