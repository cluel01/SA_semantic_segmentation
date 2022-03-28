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

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.total_size = len(self.dataset)    
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
        start_idx = 0
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
        



class VirtualMMAP():
    def __init__(self,num_replicas,total_size,mmap_name,dtype,patch_size) -> None:
        self.num_replicas = num_replicas
        self.total_size = total_size
        self.mmap_name = mmap_name
        self.patch_size = patch_size

        indices  = np.array_split(np.arange(self.total_size),self.num_replicas)
        self.mmap_handler = [np.memmap(mmap_name+"_"+str(i), dtype=dtype, mode='r', shape=(len(indices[i]),patch_size,patch_size)) for i in range(num_replicas)]
        
        start_idx = 0
        mapping = [] #todo improve search structure
        for ind in indices:
            end_idx = start_idx+len(ind)#-1
            mapping.append((start_idx,end_idx))
            start_idx = end_idx#+1
        self.mapping = mapping

    def __getitem__(self, index):
        for i,val in enumerate(self.mapping):
            min, max = val
            if min <= index < max:
                m_idx  = index-min
                mmap = self.mmap_handler[i]
                return mmap[m_idx]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.clean()

    def clean(self):
        for i in range(self.num_replicas):
            os.remove(self.mmap_name+"_"+str(i))
