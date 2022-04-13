import multiprocessing
import torch
from torch.utils.data import DataLoader,Dataset
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
import numpy as np
import os
import torch.multiprocessing as mp
import pickle
import pandas as pd 
import torch
from torch.utils.data import Dataset
import pickle
import gc

def custom_collate_fn(data):
            x,idx = zip(*data)
            x = np.stack(x)
            idx = np.stack(idx)
            del data
            return x,idx

class DataIter(Dataset):
    def __init__(self):
        with open("test_all.pkl", 'rb') as inp:
            data = pickle.load(inp)
        self.data_np = data[0]
        self.shapes = data[1]
        #self.data_np = np.load("test.npy")
        #self.shapes = pd.read_csv("test.csv")
        #self.data = [x for x in range(24000000)]

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, idx):
        data = self.data_np[idx]
        data = np.array([data], dtype=np.int64)
        return data,idx


def run(rank,d_path,s_path,queue,event):
    try:

        #with open("test.pkl", 'rb') as inp:
        print("LOAD")
        #dataset = DataIter()
        
        #dataset.save("test.pkl")
        
        #dataset = pickle.load(inp)
        dataset = SatInferenceDataset(dataset_path=d_path)
        print("LOADED")
        torch.set_num_threads(1)
        dl = DataLoader(dataset,batch_size=50,num_workers = 4,pin_memory=True,multiprocessing_context="fork")

        print("START ",str(rank))
        for i,batch in enumerate(dl):
            if i % 10 == 0:
                #print(i+rank)
                queue.put(batch[0])
        queue.put(rank)
        event.wait()
        print("Done")
    except Exception as e:
        print(f"Error: GPU {rank} - {e}")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    data_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/" 
    shape_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/shapes/"
    model_path = "/cloud/wwu1/d_satdat/christian_development/rapidearth/notebooks/satellite_data/saved_models/"
    m_path = os.path.join(model_path,"unet_07_04_2022_094905.pth")
    out_path = "/scratch/tmp/c_luel01/satellite_data/inference/"
    patch_size = [256,256,3] # [x,y,bands]
    overlap = 128
    padding = 64
    nworkers = 4
    area = "24"
    year = "2012"
    d_path = os.path.join(data_path,area,year+"_cog.tif")
    s_path = os.path.join(shape_path,area,year+".shp")

    #dataset = SatInferenceDataset(data_file_path=d_path,shape_path=s_path,overlap=128,padding=64)
    #dataset.save("testV2.pkl")
    # np.save("test.npy",dataset.patches)
    # dataset.shapes.to_csv("test.csv")

    # all = [dataset.patches,dataset.shapes]
    # with open("test_all.pkl", 'wb') as outp:  
    #             pickle.dump(all, outp, pickle.HIGHEST_PROTOCOL)

    queue = mp.JoinableQueue(1)#mp.Queue()
    event = mp.Event()

    context = mp.spawn(run,
        args=("testV2.pkl",s_path,queue,event),
        nprocs=nworkers,join=False)

    active = list(range(nworkers))
    while (len(active) > 0):
        d = queue.get()
        if type(d) == int:
            print("DONE ",str(d))
            active.remove(d)
        else:
           pass #print(d[1][0][0][0][0])
        del d
        queue.task_done()
        #print(queue.qsize())
        gc.collect()
    event.set()
    context.join()



