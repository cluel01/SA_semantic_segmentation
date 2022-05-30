
import psutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
import rasterio
from rasterio.rio.overview import get_maximum_overview_level
import os
from tqdm import tqdm
import time
import torch.multiprocessing as mp

import signal

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

out_path = "/home/c/c_luel01/satellite_data/SA_semantic_segmentation/jobs"
f = "/scratch/tmp/c_luel01/satellite_data/inference/27/2018_1.tif"

with rasterio.open(f) as file:
    out_meta = file.meta.copy()
    print(out_meta)
    out_file = os.path.join(out_path,"_rio.tif")
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**out_meta) as dataset: 
            for _, window in memfile.block_windows():
                    r = file.read(window=window)
                    dataset.write(r,window=window)
    
    

    dst_profile = cog_profiles.get("deflate")
    print(dst_profile)
    cog_translate(
        dataset,
        out_file,
        out_meta,
        in_memory=True,
        quiet=True,
    )