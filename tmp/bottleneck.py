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

import torch
from torch.functional import F
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import fiona
from shapely.geometry import shape
import cv2
from pytorch_segmentation.utils.postprocessing import mosaic_to_raster
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
#from pytorch_segmentation.utils.preprocessing import unpatchify,pad_image_topleft
from pytorch_segmentation.models import UNet
from pytorch_segmentation.utils.postprocessing import mosaic_to_raster_mp_queue

if __name__ == '__main__':
    seed = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    patch_size = [256,256,3] # [x,y,bands]
    overlap = 128
    padding = 64

    bval = (255,255,255)
    nworkers = 4
    bs = 16

    dataset_path = "data/datasets/inference_data.pkl"
    data_path = "/home/jovyan/work/satellite_data/tmp/2018.vrt"
    #data_path = "/home/jovyan/work/satellite_data/tmp/22/2018_cog.tif"
    shape_path = "data/areas/test_small"
    model_path = "saved_models/unet_07_04_2022_094905.pth" #unet_15_03_2022_071331.pth" #unet_24_03_2022_064749.pth
    
    
    dataset = SatInferenceDataset(data_file_path=data_path,shape_path=shape_path,overlap=128,padding=64)
    dataset.patches = np.tile(dataset.patches,(10,1))
    df = dataset.shapes
    df = df.loc[df.index.repeat(10)]
    df = df.reset_index(drop=True)
    df["shape_id"] = list(range(10))
    dataset.shapes = df
    dataset.save(dataset_path)
    len_dataset = len(dataset)
    del dataset
    
    
    net = UNet(n_channels=patch_size[2], n_classes=2, bilinear=False)
    net.load_state_dict(torch.load(model_path))
    #net = net.to(device)
    net.eval();
    
    
    mosaic_to_raster_mp_queue(dataset_path,net,"data/out/",mmap_shape=(len_dataset,256,256),device_ids=[0,1,2,3,4],
                              bs=150,pin_memory=True,num_workers=10)


