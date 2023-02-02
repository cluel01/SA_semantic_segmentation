import torch
from torch.utils.data import Dataset
import numpy as np


class ConcatDataset(Dataset):
    def __init__(self,dataset_list,shuffle=True,seed=42):
        np.random.seed(seed)
        datasets = []
        
        N = 0
        for d in dataset_list:
            if isinstance(d,Dataset):
                n = len(d)
                
                datasets.append(d)
                N += n
        mapping = np.empty((N,2),dtype=np.int64) # N x (dataset idx, row idx)
        
        start_idx = 0
        for d_idx,d in enumerate(datasets):
            n = len(d)
            end_idx = start_idx + n
            r_idxs = np.arange(n)
            d_idxs = np.repeat(d_idx,n)
            d_map = np.stack([d_idxs,r_idxs],axis=1)
            mapping[start_idx:end_idx] = d_map
            start_idx += n

        if shuffle:
            np.random.shuffle(mapping)

        self.datasets = datasets
        self.mapping = mapping
        self.shuffle = shuffle
        self.seed  = seed

    
    def __len__(self):
        return len(self.mapping)

    def __getitem__(self,idx):
        d_idx,r_idx = self.mapping[idx]
        return self.datasets[d_idx][r_idx]






































