from queue import Empty
import torch
from torch.utils.data import Dataset

class QueueDataset(Dataset):
    def __init__(self,queue,n_batches,timeout = 1):
        self.queue = queue
        self.size = n_batches
        self.timeout = timeout

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            batch =  self.queue.get(timeout=self.timeout)
        except Empty:
            return torch.empty(0)
        self.queue.task_done()
        return batch