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
    audio, _ = librosa.load(args.input_wav, sr=8000, mono=True)
    
    # 3. MATCH MULTI-STAGE FORMAT
    # Your baseline MultiTasNet expects a LIST of stages.
    # Since we are doing 8kHz local training, we give it a list with 1 item.
    mix = [torch.from_numpy(audio).float().to(device).unsqueeze(0).unsqueeze(1)]
    
    # Standardize/Normalize (Matching the 'mix.std' logic in your baseline)
    mix = [s / (s.std(dim=-1, keepdim=True) + 1e-8) for s in mix]

    print("Separating guitar...")
    with torch.no_grad():
        # n_chunks=1 is safe for 8kHz on a 4070
        # [-1] grabs the final high-res output stage
        separation = network.inference(mix, n_chunks=1)[-1].cpu().squeeze()

    # 4. SAVE RESULTS
    # The model still outputs 4 slots. Slot 0 or 1 will be your guitar.
    output_dir = "inference_results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(4):
        stem = separation[i].numpy()
        output_file = os.path.join(output_dir, f"guitar_slot_{i}.wav")
        sf.write(output_file, stem, 8000)
        print(f"Saved: {output_file}")

    print("\nInference Complete! Check the 'inference_results' folder.")