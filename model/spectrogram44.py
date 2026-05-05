import torch
import torch.nn as nn
import torch.nn.functional as F

class Spectrogram(nn.Module):
    def __init__(self, n_fft, hop, mels, sr):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.mels = mels
        self.sr = sr
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        stft_size = n_fft // 2 + 1
        self.mel_transform = nn.Conv1d(stft_size, mels, kernel_size=1, stride=1, padding=0, bias=True)
        self.mean = nn.Parameter(torch.empty(1, stft_size, 1), requires_grad=False)
        self.std = nn.Parameter(torch.empty(1, stft_size, 1), requires_grad=False)
        self.affine_bias = nn.Parameter(torch.zeros(1, stft_size, 1), requires_grad=True)
        self.affine_scale = nn.Parameter(torch.ones(1, stft_size, 1), requires_grad=True)

    def forward(self, audio_signal, target_length=None):
        # ULTIMATE FAIL-SAFE: If any stat parameter is on CPU, move everything to GPU
        if self.mean.device != audio_signal.device:
            self.mean.data = self.mean.data.to(audio_signal.device)
            self.std.data = self.std.data.to(audio_signal.device)
            self.affine_bias.data = self.affine_bias.data.to(audio_signal.device)
            self.affine_scale.data = self.affine_scale.data.to(audio_signal.device)

        mag = self.calculate_mag(audio_signal, db_conversion=True)
        mag = (mag - self.mean) / self.std
        mag = mag * self.affine_scale + self.affine_bias
        mag = self.mel_transform(mag)
        if target_length is not None:
            mag = F.interpolate(mag, size=target_length, mode='linear', align_corners=True)
        return mag

    def calculate_mag(self, signal, db_conversion=True):
        if self.window.device != signal.device:
            self.window.data = self.window.data.to(signal.device)
        
        signal = signal.view(-1, signal.shape[-1])
        stft = torch.stft(
            signal, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
            center=True, normalized=False, onesided=True, pad_mode='reflect', return_complex=True
        )
        mag = torch.abs(stft).pow(2)
        if db_conversion:
            mag = torch.log10(mag + 1e-8)
        return mag

    def compute_stats(self, dataset, stage_i):
        with torch.no_grad():
            sum_x = None
            sum_x2 = None
            total_elements = 0
            samples_to_process = 50
            count = 0
            
            # Keep track of where we should be
            final_target_device = self.window.device
            self.window.data = self.window.data.cpu()

            print(f"Computing stats for Stage {stage_i} (44.1kHz) using CPU accumulation...")
            for i_batch, (mix, _, _) in enumerate(dataset):
                mix = mix[stage_i].cpu()
                spec = self.calculate_mag(mix, db_conversion=True)

                if sum_x is None:
                    sum_x = torch.zeros(1, spec.shape[1], 1)
                    sum_x2 = torch.zeros(1, spec.shape[1], 1)

                sum_x += spec.sum(dim=(0, 2), keepdim=True)
                sum_x2 += (spec**2).sum(dim=(0, 2), keepdim=True)
                total_elements += (spec.shape[0] * spec.shape[2])
                count += mix.shape[0]
                if count >= samples_to_process:
                    break

            # Move window back immediately
            self.window.data = self.window.data.to(final_target_device)

            mean = sum_x / total_elements
            var = (sum_x2 / total_elements) - (mean**2)
            std = torch.sqrt(var + 1e-8)

            # Move stats to the proper device
            self.mean.data = mean.to(final_target_device)
            self.std.data = std.to(final_target_device)
            print(f"Stats computed successfully. Moved parameters to {final_target_device}")
            