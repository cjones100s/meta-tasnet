import os
from functools import partial
from random import shuffle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

class MusicDataset(Dataset):
    def __init__(self, base_path: str, base_sr_kHz: int, num_stages: int, sample_length: int, is_train: bool, shuffle_p=0.0, verbose=False):
        super(MusicDataset, self).__init__()
        # Ensure we look in the correct folder
        folder_path = f"{base_path}_{base_sr_kHz}"
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        self._tracks = [] 

        for i, f in enumerate(filenames):
            stages = [] 
            for stage_i in range(num_stages):
                # Load the merged [Mix, Guitar] file
                path = f"{base_path}_{(2**stage_i) * base_sr_kHz}/{f}"
                track = np.load(path)['arr_0'].astype('float32')
                
                # track shape is [2, Channels, Time]
                # We extract Layer 0 (Mix) and Layer 1 (Guitar)
                # We use only the first channel (index 0) to keep it Mono
                mix = track[0, 0:1, :]
                guitar = track[1, 0:1, :]
                
                # Normalization
                std = np.std(mix) if np.std(mix) != 0 else 1.0
                mix /= std
                guitar /= std
                
                # Stack for the model: shape [2, 1, Time]
                stages.append(np.stack([mix, guitar], axis=0))

            self._tracks.append(stages)
            if verbose:
                print(f"\r{i + 1}/{len(filenames)} files loaded", end='')

        if verbose:
            print(f"\n{len(self._tracks)} tracks cached")

        self._shuffle_p = shuffle_p if is_train else 0.0
        self._sample_length = sample_length
        self._train = is_train
        self._base_sr_kHz = base_sr_kHz
        self._num_stages = num_stages
        self._base_length = sample_length * 1000 * self._base_sr_kHz
        self._len = 40 * 3000 // self._sample_length if is_train else len(self._tracks)

    def __getitem__(self, index):
        if self._train:
            return self.get_train_sample()
        else:
            return self.get_validation_sample(index)

    def get_train_sample(self):
        track_id = torch.randint(0, len(self._tracks), (1,)).item()
        tracks = self._tracks[track_id]
        
        # Random crop
        start_t = torch.randint(0, tracks[0].shape[2] - self._base_length, (1,)).item()
        tracks = [track[:, :, (2**i)*start_t: (2**i)*(start_t + self._base_length)].copy() for i, track in enumerate(tracks)]
        
        # Simplified augmentation for single guitar stem
        # This keeps the [Mix, Guitar] structure intact
        tracks = [track.reshape(2, -1) for track in tracks] # shape: (2, Time)
        
        # Apply random volume to the guitar (index 1)
        new_stages = []
        for stage in tracks:
            mix = stage[0:1, :]
            guitar = stage[1:2, :]
            amp = self.random_uniform(0.75, 1.25, (1, 1)).astype('float32')
            guitar *= amp
            # Re-generate mix to match the amplified guitar
            # (In separation models, Mix = Target + Noise)
            new_stages.append((mix, guitar, np.array([1.0], dtype='float32')))
            
        # Convert to torch
        res = []
        for m, g, mask in new_stages:
            res.append((torch.from_numpy(m), torch.from_numpy(g), torch.from_numpy(mask)))
        
        mix, separated, mask = tuple(zip(*res))
        return mix, separated, mask

    def get_validation_sample(self, index):
        tracks = self._tracks[index]
        mix_stages = []
        sep_stages = []
        for stage in tracks:
            # stage[0, :, :] -> Mix [1, Time]
            # stage[1:2, :, :] -> Guitar [1, 1, Time]
            mix_stages.append(torch.from_numpy(stage[0, :, :]))
            sep_stages.append(torch.from_numpy(stage[1:2, :, :]))

        # Returns ( [MixTensor], [SepTensor] )
        return mix_stages, sep_stages 

    def random_uniform(self, low, high, size):
        r = torch.rand(size).numpy()
        return low + r * (high - low)

    def __len__(self):
        return self._len

    def get_collate_fn(self):
        return default_collate
