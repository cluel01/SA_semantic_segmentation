import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.windows import from_bounds
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from patchify import patchify
import torch.multiprocessing as mp
import torch
from torch.functional import F
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import fiona
from shapely.geometry import shape
import cv2
from pytorch_segmentation.utils.postprocessing import mosaic_to_raster
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
import pickle

def custom_collate_fn(data):
            x,idx = zip(*data)
            x = np.stack(x)
            idx = np.stack(idx)
            del data
            return x,idx

def run(rank,d_path,s_path):
    try:

        #with open("test.pkl", 'rb') as inp:
        dataset = SatInferenceDataset(d_path,s_path,overlap=128,padding=64)
        #print(len(dataset))
        #dataset.save("test.pkl")
        #print("LOAD")
        #dataset = pickle.load(inp)
        print("LOADED")

        dl = DataLoader(dataset,collate_fn=custom_collate_fn,batch_size=50,num_workers = 4,pin_memory=True)

        print("START")
        for i,batch in enumerate(dl):
            if i % 400 == 0:
                print(i+rank)
        print("Done")
    except Exception as e:
        print(f"Error: GPU {rank} - {e}")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


    data_path = "/home/jovyan/work/satellite_data/tmp" 
    shape_path = "/home/jovyan/work/satellite_data/tmp/shapes/"
    model_path = "saved_models/"
    m_path = os.path.join(model_path,"unet_07_04_2022_094905.pth")
    out_path = "data/out"
    patch_size = [256,256,3] # [x,y,bands]
    overlap = 128
    padding = 64
    nworkers = 4
    area = "24"
    year = "2012"
    d_path = os.path.join(data_path,area,year+"_cog.tif")
    s_path = os.path.join(shape_path,area,year+".shp")

    mp.spawn(run,
            args=(d_path,s_path,),
            nprocs=nworkers)