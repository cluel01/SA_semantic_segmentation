from concurrent.futures import thread
import math
from mmap import mmap
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import numpy as np
import os 

class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset,start_idx=0, end_idx = None,num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank

        if rank >= num_replicas:
            raise ValueError(f"Not enough replicas for rank {rank}")

        self.start_idx = start_idx
        if end_idx is None:
            self.end_idx = len(dataset)
        else:
            self.end_idx = end_idx
        self.total_size = self.end_idx-self.start_idx
        self.num_samples = self._get_num_samples()          # true value without extra samples


    def __iter__(self):
        # subsample
        indices = self._get_indices()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def _get_num_samples(self):
        threshold = self.total_size %  self.num_replicas
        if self.rank < threshold:
            return self.total_size // self.num_replicas + 1 
        else:
            return self.total_size // self.num_replicas

    def _get_indices(self):
        threshold = self.total_size %  self.num_replicas
        idx = 0
        start_idx = self.start_idx
        while idx < self.rank:
            if idx < threshold:
                start_idx += self.total_size //  self.num_replicas+1
            else:
                start_idx += self.total_size //  self.num_replicas
            idx += 1
        if self.rank < threshold:
            len_idx = self.total_size //  self.num_replicas+1
        else:
            len_idx = self.total_size //  self.num_replicas
        return list(range(start_idx,start_idx+len_idx))
        
