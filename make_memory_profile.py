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
from pytorch_segmentation.inference import mosaic_to_raster
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
#from pytorch_segmentation.utils.preprocessing import unpatchify,pad_image_topleft


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
#data_path = "/home/jovyan/work/satellite_data/tmp/27/2018_cog.tif"
shape_path = "data/areas/test"
model_path = "saved_models/unet_07_04_2022_094905.pth" #unet_15_03_2022_071331.pth" #unet_24_03_2022_064749.pth


dataset = SatInferenceDataset(data_file_path=data_path,shape_path=shape_path,overlap=128,padding=64)
shapes = dataset.shapes.copy()
dataset.save(dataset_path)

del dataset

from pytorch_segmentation.models import UNet
net = UNet(n_channels=patch_size[2], n_classes=2, bilinear=False)
net.load_state_dict(torch.load(model_path))
#net = net.to(device)
net.eval();

from pytorch_segmentation.inference import mosaic_to_raster_mp_queue_memory
mosaic_to_raster_mp_queue_memory(dataset_path,shapes,net,"data/out/",device_ids=[1,2,3,4],
                          bs=150,pin_memory=True,num_workers=5)