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
        self.epoch = 0

        self.total_size = len(self.dataset)    
        indices = np.array_split(list(range(self.total_size)),self.num_replicas)[self.rank]
        self.num_samples = len(indices)             # true value without extra samples
        self.start_idx = indices[0]
        self.end_idx = indices[-1]


    def __iter__(self):
        indices = list(range(len(self.dataset)))
 
        # subsample
        indices = np.array_split(indices,self.num_replicas)[self.rank]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

class VirtualMMAP():
    def __init__(self,num_replicas,total_size,mmap_name,dtype,patch_size) -> None:
        self.num_replicas = num_replicas
        self.total_size = total_size
        self.mmap_name = mmap_name
        self.patch_size = patch_size

        indices  = np.array_split(list(range(self.total_size)),self.num_replicas)
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

    def clean(self):
        for i in range(self.num_replicas):
            os.remove(self.mmap_name+"_"+str(i))
