import os
import numpy as np
from tqdm.notebook import tqdm
from deepnote import MusicRepr
from joblib import delayed, Parallel

import torch
from torch.utils.data import random_split, Dataset, DataLoader

def get_dataloaders(dataset,
                    n_jobs=2,
                    batch_size=64, 
                    val_frac=0.2):
    
    n = len(dataset)
    v = int(n*val_frac)
    train_dataset, val_dataset = random_split(dataset, [n - v, v])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_jobs, collate_fn=dataset.fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_jobs, collate_fn=dataset.fn)
    print('train dataset has {} samples and val dataset has {} samples.'.format(n-v, v))
    return train_loader, val_loader


def load_midi(file, instruments=None):
    seq = MusicRepr.from_file(file)
    if instruments is None:
        return seq
    if len(set(instruments).intersection(set(seq.get_instruments()))) == 0:
        return None

    tracks = seq.separate_tracks()
    res = {}
    for inst in instruments:
        if inst in tracks:
            res[inst] = tracks[inst]
    return np.concatenate([MusicRepr.merge_tracks(res).to_cp(), np.array([[2] + [0]*7])], axis=0)


class LMDataset(Dataset):
    
    def __init__(self, data_dir, max_files=100, instruments:list=None, max_len=256, masked=False, p_mask=0.2):
        super().__init__()

        ## load samples
        files = list(filter(lambda x: x.endswith('.mid'), os.listdir(data_dir)))[:max_files]
        self.samples = list(
            filter(
                lambda x: x is not None, 
                Parallel(n_jobs=12)(delayed(load_midi)(data_dir + file, instruments) for file in tqdm(files))
            )
        )
        if instruments is None:
            instruments = set()
            for samp in self.samples:
                instruments.update(samp.get_instrument())
            self.instruments = list(instruments)
        else:
            self.instruments = instruments
        
        self.max_len = max_len
        self.masked = masked
        self.p_mask = p_mask
        
        self.lens = [max(1, len(samp) - max_len) for samp in self.samples]
        self.cum_lens = [0] + [sum(self.lens[:i+1]) for i in range(len(self.samples))]
        

    def __len__(self):
        return self.cum_lens[-1]
    
    def get_idx(self, idx):
        for i, cl in enumerate(self.cum_lens):
            if idx < cl:
                return i-1, idx - self.cum_lens[i-1]
        return -1, -1

    def __getitem__(self, idx):
        samp_idx, offset = self.get_idx(idx)
        if samp_idx > -1:
            x = np.array(self.samples[samp_idx][offset : offset + self.max_len])

            y = np.array(self.samples[samp_idx][offset + 1 : offset + self.max_len + 1])
            return x, y
        raise Exception('Wrong index for the dataset.')

    def mask(self, x):
        if self.masked:
            raise NotImplementedError
        return x

    def fn(self, batch):
        X = []
        Y = []
        for b in batch:
            x, y = b
            X += [x] 
            Y += [y] 

        x_len = torch.tensor([x.shape[0] for x in X])
        M = max(x_len)
        res = {
            'X': torch.tensor([np.pad(x, ((0, M - x.shape[0]), (0,0))) for x in X]),
            'X_len': x_len,
            'labels': torch.tensor([np.pad(x, ((0, M - y.shape[0]), (0,0))) for y in Y])
        }
        return res