import sys 
sys.path.append('/home/c/c_luel01/satellite_data/SA_semantic_segmentation/')

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import os
from pathlib import Path
from pytorch_segmentation.data.inference_dataset import SatInferenceDataset

area = "27"
year  = "2018"

data_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/" 
shape_path = "/cloud/wwu1/d_satdat/shared_satellite_data/tmp/shapes/"

out_path = "/scratch/tmp/c_luel01/satellite_data/inference/"
patch_size = [256,256,3] # [x,y,bands]
overlap = 128
padding = 64
nworkers = 10
bs = 150

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_path = os.path.join(data_path,area,year+"_cog.tif")
s_path = os.path.join(shape_path,area,year+".shp")
o_path = os.path.join(out_path,"test",area)
Path(o_path).mkdir(parents=True, exist_ok=True)
tmp_d_path = os.path.join(o_path,"tmp_"+year+".pkl")

dataset = SatInferenceDataset(data_file_path=d_path,shape_file=s_path,overlap=overlap,padding=padding)
shapes = dataset.shapes.copy()
dataset.save(tmp_d_path)

queue = mp.JoinableQueue(20)

dl =  DataLoader(dataset,batch_size=bs,num_workers = nworkers,pin_memory=True,multiprocessing_context="fork")

for batch in dl:
    queue.put(batch)
    print(queue.qsize())