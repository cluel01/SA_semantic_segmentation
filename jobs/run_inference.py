import sys 
sys.path.append('/home/c/c_luel01/satellite_data/SA_semantic_segmentation/')

import torch
import torch.multiprocessing as mp

import os
from pathlib import Path

from pytorch_segmentation.inference import mosaic_to_raster
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
from pytorch_segmentation.models import UNet


if __name__ == '__main__':
    #torch.set_num_threads(1)

    #area = str(sys.argv[1])
    year = str(sys.argv[1])
    n_gpus = int(sys.argv[2])
    device_ids = list(range(n_gpus))
    gpu_type = str(sys.argv[3])
    model_name = str(sys.argv[4])
    rescale_factor = int(sys.argv[5])
    nworkers = int(sys.argv[6])
    if len(sys.argv) == 8:
        start_idx = int(sys.argv[7])
        end_idx = int(sys.argv[8])
        shape_idx = list(range(start_idx,end_idx))
    else:
        shape_idx = None

    data_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/" 
    #shape_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/shapes/"
    shape_path = "/cloud/wwu1/d_satdat/shared_satellite_data/ku_sync/South_Africa/merged_v0/cutlines/"
    model_path = "/cloud/wwu1/d_satdat/christian_development/rapidearth/notebooks/satellite_data/saved_models/"
     #"unet_25_05_2022_174303.pth" #"unet_17_05_2022_085640.pth" #unet_12_05_2022_145256#m_path = os.path.join(model_path,"unet_05_05_2022_113034.pth")
    m_path = os.path.join(model_path,model_name) 
    out_path = "/scratch/tmp/c_luel01/satellite_data/inference/"
    patch_size = [256,256]
    n_channels = 3 

    overlap = 128 / rescale_factor
    padding = 64 / rescale_factor

    if n_gpus == 1:
        bs = 150
    else:
        if gpu_type == "gpua100":
            bs = 230
        elif gpu_type == "gpu2080":
            bs = 60
        else:
            bs = 130 


    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("######################")
    print("Config:\nYear: {}\nNGPUs: {}\nGPU: {}\nModel: {}\nRescale: {}\nNWorkers: {}\nShape idx: {}\n\
           Overlap: {}\nPadding: {}\nBS: {}\nData Path: {}\nShape Path: {}\nOut Path:{}".format([year,n_gpus,
           gpu_type,model_name,rescale_factor,nworkers,str(shape_idx),overlap,padding,bs,data_path,shape_path,out_path]))
    print("######################")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if int(area) not in range(22,35):
    #     sys.exit("Not existing area!")
    if int(year) not in range(2008,2019):
        sys.exit()

    d_path = os.path.join(data_path,year+".vrt")
    #s_path = os.path.join(shape_path,area,year+".shp")
    s_path = os.path.join(shape_path,year+".geojson")
    o_path = os.path.join(out_path,model_name)
    Path(o_path).mkdir(parents=True, exist_ok=True)
    tmp_d_path = os.path.join(o_path,"tmp_"+year+".pkl")

    dataset = SatInferenceDataset(data_file_path=d_path,shape_file=s_path,overlap=overlap,padding=padding,shape_idx=shape_idx,
                                    rescale_factor=rescale_factor,n_channels=n_channels,patch_size=patch_size)
    shapes = dataset.shapes.copy()
    dataset.save(tmp_d_path)
    del dataset

    net = UNet(n_channels=n_channels, n_classes=2, bilinear=False)
    net.load_state_dict(torch.load(m_path))

    mosaic_to_raster(tmp_d_path,shapes,net,o_path,device_ids=device_ids,bs=bs,pin_memory=True,num_workers=nworkers)
    os.remove(tmp_d_path)

