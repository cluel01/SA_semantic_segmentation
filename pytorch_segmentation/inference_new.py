from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
# from rasterio.enums import Resampling
# from rasterio.vrt import WarpedVRT
import fiona
import os
from tqdm import tqdm
import time
import torch.multiprocessing as mp
from torchvision.transforms.functional import resize

from .data.sampler import DistributedEvalSampler
from .data.inference_dataset import SatInferenceDataset
from .utils.unpatchify import unpatchify_window_memfile
from .utils.cog_translate import cog_translate


def segment_trees(data_path,net,out_path,device_ids,shape_path=None,shape_idxs=None,bs=16,num_workers=4,pin_memory=True,
                  compress="deflate",blocksize=512,nodata=0,model_class="unet",patch_size=[256,256],overlap=128,padding=64,rescale_factor=1):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Runtime error!")
        pass

    if not os.path.isdir(out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)
    
    if shape_path is None:
        n_shapes = 1
        shape_idxs = [0]
    else:
        if shape_idxs is None:
            with fiona.open(shape_path) as shps:
                n_shapes = len(shps)
                shape_idxs = list(range(n_shapes))


    #Split dataset into smaller chunks per number of devices
    if type(device_ids) == int:
        n_devices = 1
    elif type(device_ids) == list:
        n_devices = len(device_ids)

    #split the shape_idxs into n_devices chunks
    shape_idxs = np.array_split(shape_idxs,n_devices)   

    processes = []
    for i in range(n_devices):
        s_idxs = shape_idxs[i].tolist()
        if len(s_idxs) > 0:
            if n_devices > 1:
                device_id = device_ids[i]
            else:
                device_id = device_ids
            print("Device: ",device_id)

            d = SatInferenceDataset(data_file_path=data_path,shape_file=shape_path,patch_size=patch_size,overlap=overlap,padding=padding,shape_idx=s_idxs,rescale_factor=rescale_factor)

            p = mp.Process(target=run_inference_gpu, args=(d,net,device_id,i,out_path,bs,num_workers,pin_memory,model_class,compress,blocksize,nodata),
                            daemon=False)
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
        

def run_inference_gpu(dataset,net,device_id,rank,out_path,bs,num_workers,pin_memory,model_class,compress,blocksize,nodata):
    with rasterio.Env(GDAL_CACHEMAX=32000000): 
        try:
            mp_context = "fork"
            #torch.set_num_threads(1) #prevent memory leakage
            # if num_workers == 0:
            #     mp_context = None

            print("Start GPU:",device_id)
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
            torch.cuda.set_device(device_id)
            #device_id = torch.device("cuda",device_id)
            torch.cuda.empty_cache()
            net = net.to(device_id)
            net.eval()

            shapes = dataset.shapes

            processes = []
            for _,s in shapes.iterrows():
                queue = mp.Queue(10)
                batch_queue = mp.Queue(10)
                
                p = mp.Process(target=batch_loader, args=(dataset,bs,s,batch_queue))
                p.start()
                processes.append(p)
                p = mp.Process(target=queue_consumer_gpu, args=(queue,s,out_path,compress,blocksize,nodata))
                p.start()
                processes.append(p)
                start_idx = s["start_idx"]
                # end_idx  = start_idx + np.prod(s["grid_shape"])
                # subset = Subset(dataset,range(start_idx,end_idx))
                # dl = DataLoader(subset,batch_size=bs,num_workers = num_workers,pin_memory=pin_memory,multiprocessing_context=mp_context,shuffle=False)
                run = True
                n = int(np.ceil(np.prod(s["grid_shape"])) / bs)
                with tqdm(total=n,position=rank) as pbar:
                    while run:
                        while not batch_queue.empty():
                            batch = batch_queue.get()
                            if batch == "DONE":
                                run = False
                            else:
                                with torch.no_grad():
                                    x,idx = batch
                                    #x = torch.from_numpy(x).float().to(rank,non_blocking=True)#[0].to(device)
                                    x = x.float().to(device_id,non_blocking=True)
                                    b_idx = idx - start_idx
                                    if model_class == "unet":
                                        with torch.cuda.amp.autocast():
                                            out = net(x)
                                            out = F.softmax(out,dim=1)
                                            out = torch.argmax(out,dim=1)
                                            out = out.byte().cpu()
                                    elif model_class == "smp":
                                        with torch.cuda.amp.autocast():
                                            out = net(x)
                                            out = out.squeeze().sigmoid()
                                            out = (out > 0.5).byte().cpu()
                                    queue.put([b_idx,out])
                                    pbar.update(1)
                                    print(batch_queue.qsize())
                queue.put(["DONE"])
                for p in processes:
                    p.join()
                queue.close()
                batch_queue.close()

        except Exception as e:
            print(f"Error: Job {device_id} - {e}")
            queue.put(["ERROR"])

def queue_consumer_gpu(queue,shape,out_path,compress,blocksize,nodata):
    complete = True
    memfile,mem_meta = create_memfile(shape,compress,blocksize,nodata) #create_files(shapes,out_path,compress,blocksize)
    while complete == True:
        d = queue.get()
        if type(d[0]) == str:
            if d[0] == "DONE":
                complete = False
            else:
                print("Error in queue")
                return
        else:
            unpatchify_window_memfile(shape,d[1],d[0],memfile)
        del d
    start = time.time()
    out_meta = mem_meta.copy() #shape["sat_meta"]
    out_file = os.path.join(out_path,shape["name"]+".tif")
    cog_translate(
        memfile,
        out_file,
        out_meta
    )
    end = time.time()
    print(f"INFO: Written {out_file} in {end-start:.3f} seconds")
    memfile.close()
    return
                

def create_memfile(shape,compress,blocksize,nodata):
    #with rasterio.Env(GDAL_CACHEMAX=32000000000,GDAL_TIFF_INTERNAL_MASK=True,GDAL_TIFF_OVR_BLOCKSIZE=128): #TODO change to variable
    memory = rasterio.MemoryFile()
    out_transform = shape["transform"]
    out_meta = shape["sat_meta"]
    
    height = shape["height"]
    width = shape["width"]
    out_meta.update({"driver": "GTiff",
                    "overwrite":True,
                    "count":1,
                    "height": height,
                    "width": width,
                    "transform": out_transform,
                    "compress":compress,
                    "tiled":True,
                    "blockxsize":blocksize, 
                    "blockysize":blocksize,
                    "BIGTIFF":'YES',
                    "nodata":nodata,
                    "PREDICTOR":2,
                    "NUM_THREADS":"ALL_CPUS"})
    mfile = memory.open(**out_meta)
    return mfile,out_meta

def batch_loader(dataset,batch_size,shape,queue):
    start_idx = shape["start_idx"]
    end_idx  = start_idx + np.prod(shape["grid_shape"])
    target_shape = dataset.img_shape
    batch = torch.empty((batch_size,target_shape[0],target_shape[1],target_shape[2]),dtype=torch.float32)
    c = 0
    s_idx = -1
    with rasterio.open(dataset.data_file_path) as src_sat:
        for i in range(start_idx,end_idx):
            if s_idx == -1:
                s_idx = i

            win = Window(*dataset.patches[i])
            img = src_sat.read(window=win,fill_value=0)
            # if img.shape != dataset.img_shape:
            #     vertical = target_shape[1] - img.shape[1]
            #     horizontal = target_shape[2] - img.shape[2]

            #     bottom = vertical // 2
            #     top = vertical - bottom
            #     right = horizontal // 2
            #     left = horizontal - right
            #     img = np.pad(img,[(0,0),(top,bottom),(left,right)],"reflect")
            tensor = torch.from_numpy(img).float()
            # if dataset.rescale_factor != 1:
            #     img = resize(img,dataset.patch_size)

            # if dataset.transform:
            #     sample = dataset.transform(image=img,target=None)
            #     img,_=  sample

            batch[c] = tensor
            c += 1
            if c == batch_size:
                batch = batch // 255
                queue.put([batch,s_idx])
                print(f"Batch {s_idx} - {i} loaded")
                c = 0
                s_idx = -1
    #for last batch
    if c != 0:
        batch = batch[:c] // 255
        queue.put([batch,s_idx])
    queue.put(["DONE"])


