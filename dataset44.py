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

        # Surgery: Save metadata only, do not load data into RAM
        self._base_path = base_path
        self._base_sr_kHz = base_sr_kHz
        self._num_stages = num_stages
        
        folder_path = f"{base_path}_{base_sr_kHz}"
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        self._filenames = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

        if verbose:
            print(f"Dataset initialized with {len(self._filenames)} tracks (Lazy Loading for 44kHz)")

        self._shuffle_p = shuffle_p if is_train else 0.0
        self._sample_length = sample_length
        self._train = is_train
        self._base_length = sample_length * 1000 * self._base_sr_kHz
        
        # Use filename count for validation length
        self._len = 40 * 3000 // self._sample_length if is_train else len(self._filenames)

    def _load_track(self, filename):
        """ The exact loading and normalization logic from your 8k __init__ loop """
        stages = []
        for stage_i in range(self._num_stages):
            path = f"{self._base_path}_{(2**stage_i) * self._base_sr_kHz}/{filename}"
            track = np.load(path)['arr_0'].astype('float32')
            
            mix = track[0, 0:1, :]
            guitar = track[1, 0:1, :]
            
            std = np.std(mix) if np.std(mix) != 0 else 1.0
            mix /= std
            guitar /= std
            
            stages.append(np.stack([mix, guitar], axis=0))
        return stages

    def __getitem__(self, index):
        if self._train:
            return self.get_train_sample()
        else:
            return self.get_validation_sample(index)

    def get_train_sample(self):
        # Pick a random filename
        track_id = torch.randint(0, len(self._filenames), (1,)).item()
        filename = self._filenames[track_id]
        
        # LOAD ON THE FLY
        tracks = self._load_track(filename)

        start_t = torch.randint(0, tracks[0].shape[2] - self._base_length, (1,)).item()
        tracks = [track[:, :, (2**i)*start_t: (2**i)*(start_t + self._base_length)].copy() for i, track in enumerate(tracks)]

        tracks = [track.reshape(2, -1) for track in tracks]

        new_stages = []
        for stage in tracks:
            mix = stage[0:1, :]
            guitar = stage[1:2, :]
            amp = self.random_uniform(0.75, 1.25, (1, 1)).astype('float32')
            guitar *= amp
            new_stages.append((mix, guitar, np.array([1.0], dtype='float32')))

        res = []
        for m, g, mask in new_stages:
            res.append((torch.from_numpy(m), torch.from_numpy(g), torch.from_numpy(mask)))
        mix, separated, mask = tuple(zip(*res))
        return mix, separated, mask

    def get_validation_sample(self, index):
        filename = self._filenames[index]
        
        # LOAD ON THE FLY
        tracks = self._load_track(filename)
        
        mix_stages = []
        sep_stages = []
        for stage in tracks:
            mix_stages.append(torch.from_numpy(stage[0, :, :]))
            sep_stages.append(torch.from_numpy(stage[1:2, :, :]))
        return mix_stages, sep_stages

    def random_uniform(self, low, high, size):
        r = torch.rand(size).numpy()
        return low + r * (high - low)

    def __len__(self):
        return self._len

    def get_collate_fn(self):
        return default_collate