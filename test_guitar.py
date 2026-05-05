import torch
import librosa
import soundfile as sf
import numpy as np
import argparse
import os
from model.tasnet import MultiTasNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Name of folder in ./checkpoints")
    parser.add_argument("--input_wav", required=True, type=str, help="Path to raw wav file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LOAD THE MODEL
    # We use the 'MultiTasNet' class as per your baseline script
    checkpoint_path = r"C:\Users\cjone\Projects\meta-tasnet\checkpoint8\best_checkpoint"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["args"]
    
    network = MultiTasNet(model_args).to(device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    # 2. LOAD AND PREP AUDIO (Mono 8kHz)
    print(f"Loading {args.input_wav}...")
    # Pull the sampling rate directly from the model's own training arguments
    target_sr = model_args.sampling_rate 
    audio, _ = librosa.load(args.input_wav, sr=target_sr, mono=True)
    
    # 3. MATCH MULTI-STAGE FORMAT
    # Ensure we use the exact same normalization as dataset.py
    mix_tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0).unsqueeze(1)
    std = mix_tensor.std() if mix_tensor.std() != 0 else 1.0
    mix = [mix_tensor / std]

    print("Separating guitar...")
    with torch.no_grad():
        # n_chunks=1 is safe for 8kHz on a 4070
        # [-1] grabs the final high-res output stage
        separation = network.inference(mix, n_chunks=1)[-1].cpu().squeeze()

    # 4. SAVE RESULTS
    # The model still outputs 4 slots. Slot 0 or 1 will be your guitar.
    song_name = os.path.basename(os.path.dirname(args.input_wav))
    
    output_dir = os.path.join("inference_results", song_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
     # Mapping based on your Slot 1 Hijack
    slot_labels = {
        0: "drums",
        1: "guitar",
        2: "other",
        3: "vocals"
    }

    for i in range(4):
        stem = separation[i].numpy()
        label = slot_labels.get(i, f"slot_{i}")
        output_file = os.path.join(output_dir, f"{song_name}_{label}.wav")
        
        sf.write(output_file, stem, target_sr)
        print(f"Saved: {output_file}")

    print(f"\nInference Complete! Results saved to: {output_dir}")