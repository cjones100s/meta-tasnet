import torch
import librosa
import soundfile as sf
import numpy as np
import os
from model.tasnet import MultiTasNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD THE HERO MODEL
# Use the best_checkpoint from your Blue Hive run
model_path = r"C:\Users\cjone\Projects\meta-tasnet\checkpoint44\best_checkpoint"
checkpoint = torch.load(model_path, map_location=device)
network = MultiTasNet(checkpoint["args"]).to(device)
network.load_state_dict(checkpoint["state_dict"])
network.eval()

# 2. MULTI-STAGE PREP (Theoretical Requirement)
# High-fi inference requires 8k, 16k, and 32k versions of the mix
raw_input = r"C:\Users\cjone\Guitar_Separation\train\BigTroubles_Phantom\mixture.wav"
audio, _ = librosa.load(raw_input, sr=44100, mono=True)

# Generate the 3-stage mixture list
mix_stages = []
for s in [8000, 16000, 32000]:
    resampled = librosa.resample(audio, orig_sr=44100, target_sr=s, res_type='kaiser_best')
    # Standardize length for the architecture
    resampled = torch.from_numpy(resampled).float().to(device).view(1, 1, -1)
    # Normalize by standard deviation (Matches Blue Hive training logic)
    resampled = resampled / (resampled.std(dim=-1, keepdim=True) + 1e-8)
    mix_stages.append(resampled)

# 3. RUN INFERENCE 
with torch.no_grad():
    # If stages_num is 1, it expects a list with 1 tensor.
    # We'll take the highest res one from your stages (the 32kHz one)
    mix_input = [mix_stages[-1]] 
    
    # Run the AI
    # Since stages_num is 1, we don't use [-1] on the output
    output = network.inference(mix_input, n_chunks=8)
    
    # If the output is a list, grab the tensor. If it's a tensor, keep it.
    separation = output[0] if isinstance(output, list) else output
    separation = separation.cpu().squeeze()

# 4. SAVE THE RESULTS (Final resample back to 44.1kHz)
for i in range(4):
    stem = separation[i].numpy()
    # Resample to final 44.1k for maximum clarity
    high_res_stem = librosa.resample(stem, orig_sr=32000, target_sr=44100)
    sf.write(f"hero_result_slot_{i}.wav", high_res_stem, 44100)