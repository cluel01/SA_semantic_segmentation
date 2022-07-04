import torch

import os
from pytorch_segmentation.models import UNet
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
from pytorch_segmentation.utils.helper import shapes_intersecting_with_raster
from inference_tmp import mosaic_to_raster

# from .inference_tmp import mosaic_to_raster

if __name__ == '__main__':
    seed = 42

    device = torch.device('cuda')
    d_ids = [0,1]
    #device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


    patch_size = [256,256,3] # [x,y,bands]
    overlap = 128
    padding = 64

    bval = (255,255,255)
    nworkers = 25
    bs = 100

    dataset_path = "inference_data.pkl"
    #data_path = "2018_cog.tif"
    data_path = "/home/jovyan/work/satellite_data/tmp/2018.vrt"
    #shape_path = "test_area_20.shp"
    shape_path = "../../data/areas/test/boundary.shp"

    data_path = "/home/jovyan/work/satellite_data/tmp/24/2018_cog.tif"
    #data_path = "/home/jovyan/work/satellite_data/tmp/27/2018_cog.tif"
    shape_path = "/home/jovyan/work/satellite_data/tmp/shapes/24/2018.shp"


    model_name = "../../saved_models/unet_02_06_2022_125601_new"
    model_path =  model_name +  ".pth" #unet_15_03_2022_071331.pth" #unet_24_03_2022_064749.pth
    out_path = "out/"

    #shape_idxs = shapes_intersecting_with_raster(shape_path,data_path)
    dataset = SatInferenceDataset(data_file_path=data_path,shape_file=shape_path,shape_idx=None,overlap=128,padding=64)
    shapes = dataset.shapes.copy()
    dataset.save(dataset_path)

    print("t")

    net = UNet(n_channels=patch_size[2], n_classes=2, bilinear=False)
    net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net = net.to(device)
    net.eval()

    mosaic_to_raster(dataset_path,shapes,net,out_path,device_ids=d_ids,
                    bs=bs,pin_memory=True,num_workers=nworkers)

    os.remove(dataset_path)