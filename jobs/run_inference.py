import sys 
sys.path.append('/home/c/c_luel01/satellite_data/SA_semantic_segmentation/')

import torch
import torch.multiprocessing as mp

import os
from pathlib import Path

from pytorch_segmentation.utils.postprocessing import mosaic_to_raster_mp_queue,mosaic_to_raster
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
from pytorch_segmentation.models import UNet


if __name__ == '__main__':
    area = str(sys.argv[1])
    year = str(sys.argv[2])
    n_gpus = int(sys.argv[3])
    device_ids = list(range(n_gpus))

    data_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/" 
    shape_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/shapes/"
    model_path = "/cloud/wwu1/d_satdat/christian_development/rapidearth/notebooks/satellite_data/saved_models/"
    m_path = os.path.join(model_path,"unet_07_04_2022_094905.pth")
    out_path = "/scratch/tmp/c_luel01/satellite_data/inference/"
    patch_size = [256,256,3] # [x,y,bands]
    overlap = 128
    padding = 64
    nworkers = 4
    if n_gpus == 1:
        bs = 150
    else:
        bs = 130 

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(sys.argv) != 4:
        sys.exit("Missing arguments!")

    if int(area) not in range(22,35):
        sys.exit("Not existing area!")
    if int(year) not in range(2008,2019):
        sys.exit()

    d_path = os.path.join(data_path,area,year+"_cog.tif")
    s_path = os.path.join(shape_path,area,year+".shp")
    o_path = os.path.join(out_path,area)
    Path(o_path).mkdir(parents=True, exist_ok=True)
    tmp_d_path = os.path.join(o_path,"tmp_"+year+".pkl")

    dataset = SatInferenceDataset(d_path,s_path,overlap=overlap,padding=padding)
    dataset.save(tmp_d_path)
    mmap_shape = (len(dataset),patch_size[0],patch_size[1])
    del dataset

    net = UNet(n_channels=patch_size[2], n_classes=2, bilinear=False)
    net.load_state_dict(torch.load(m_path))

    mosaic_to_raster_mp_queue(tmp_d_path,net,o_path,mmap_shape=mmap_shape,
                        device_ids=device_ids,bs=bs,pin_memory=True,num_workers=nworkers)
    os.remove(tmp_d_path)