import numpy as np
import torch
from ..data.concat_dataset import ConcatDataset

def create_weighted_dataset(datasets,pixel_weighted=False,weight=None):
    n = 0 
    for i in datasets:
        n += len(i)

    weights = np.zeros(n,dtype=np.float64)

    start = 0
    total = 0
    for j in range(len(datasets)):
        d = datasets[j]
        summed = np.sum(d.y,axis=(1,2))
        weights[start:start+len(d)] = summed > 0
        start += len(d)
        total += np.sum(summed)

    if weight is None:
        if pixel_weighted:
            w = total / (n * np.prod(d.y[0].shape))
        else:
            c = np.sum(weights)   
            w = c / n
    else:
        w = weight

    weights[weights == 0] = w
    weights[weights == 1] = 1 - w
    #concat_dataset = torch.utils.data.ConcatDataset(datasets)
    concat_dataset = ConcatDataset(datasets)
    return concat_dataset,weights
    