import torch
import torch.nn.functional as F

def sdr_objective(estimation, origin, mask=None):
    # 1. Flatten to 1D to stop all shape/channel mismatches
    estimation = estimation.flatten()
    origin = origin.flatten()

    # 2. Trim to the shortest length to handle padding differences
    min_len = min(len(estimation), len(origin))
    if mask is not None:
        mask = mask.flatten()
        min_len = min(min_len, len(mask))
        mask = mask[:min_len]

    estimation = estimation[:min_len]
    origin = origin[:min_len]

    # 3. Apply mask ONLY if it is provided (for guitar stems)
    if mask is not None:
        estimation = estimation * mask
        origin = origin * mask

    # 4. SI-SNR Math
    origin_power = torch.pow(origin, 2).sum()
    dot_product = (origin * estimation).sum()
    projection = dot_product * origin / (origin_power + 1e-8)
    
    noise = estimation - projection
    ratio = torch.pow(projection, 2).sum() / (torch.pow(noise, 2).sum() + 1e-8)
    
    return 10 * torch.log10(ratio + 1e-8)

def calculate_loss(estimated_separation, true_separation, mask, true_latents, estimated_mix, true_mix, args):
    """ 
    Simplified loss for single-instrument (Guitar) training 
    optimized for Slot 1 (Bass-lane) hijack.
    """
    stats = torch.zeros(7).to(mask.device)

    # 1. ISOLATION SURGERY: Pick ONLY Slot 1 [Batch, 1, Time]
    # estimated_separation is (B, 4, 1, T) -> guitar_est is (B, 1, T)
    guitar_est = estimated_separation[:, 1, :, :]
    
    # 2. ALIGN GROUND TRUTH: Ensure shapes match
    # true_separation is often [B, 1, 1, T], we squeeze to get [B, 1, T]
    guitar_true = true_separation.squeeze(1) if true_separation.dim() == 4 else true_separation

    # --- ONE-TIME TELEMETRY AUDIT ---
    if not hasattr(calculate_loss, "_verified"):
        print(f"\n" + "="*50)
        print(f"[AUDIT] Model Raw Output Shape: {estimated_separation.shape}")
        print(f"[AUDIT] Sliced Guitar Est:     {guitar_est.shape}")
        print(f"[AUDIT] Sliced Guitar True:    {guitar_true.shape}")
        
        # This is the "Static Noise" check: lengths MUST be identical
        est_len = len(guitar_est.flatten())
        true_len = len(guitar_true.flatten())
        print(f"[AUDIT] Est Flattened Len:     {est_len}")
        print(f"[AUDIT] True Flattened Len:    {true_len}")
        
        if est_len != true_len:
            print("[WARNING] Length mismatch detected! Check slicing.")
        else:
            print("[SUCCESS] Dimensions aligned for SI-SNR calculation.")
        print("="*50 + "\n")
        
        calculate_loss._verified = True

    # 3. SDR MATH: Use the matched 1D arrays
    # Flattening here is now safe because we aren't mixing different channels
    sdr = sdr_objective(guitar_est.flatten(), guitar_true.flatten())
    
    # 4. TOTAL LOSS: Negate SDR so the optimizer minimizes the error
    total_loss = -sdr
    stats[1] = sdr

    # 5. RECONSTRUCTION LOSS: Mix vs. Sum of Parts
    # This forces the other 3 lanes to hold the leftover non-guitar energy
    if args.reconstruction_loss_weight > 0:
        # Sum all 4 channels to check against original mix
        summed_est = estimated_separation.sum(dim=1) 
        reconstruction_sdr = sdr_objective(summed_est.flatten(), true_mix.flatten())
        stats[4] = reconstruction_sdr
        total_loss += -args.reconstruction_loss_weight * reconstruction_sdr

    return total_loss, stats