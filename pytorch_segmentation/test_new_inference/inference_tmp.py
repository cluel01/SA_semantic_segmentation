#from rio_cogeo.cogeo import cog_translate
import psutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import rasterio
# from rasterio.enums import Resampling
# from rasterio.vrt import WarpedVRT
from osgeo import gdal
import os
from tqdm import tqdm
import time
import torch.multiprocessing as mp

from pytorch_segmentation.data.inference_dataset import SatInferenceDataset
from pytorch_segmentation.data.queue_dataset import QueueDataset
from pytorch_segmentation.utils.unpatchify import unpatchify_window_queue
from pytorch_segmentation.utils.cog_translate import cog_translate

import signal

#Wrapper function that can call all different versions
def mosaic_to_raster(dataset_path,shapes,net,out_path,device_ids,bs=16,
                    num_workers=4,pin_memory=True,compress="deflate",blocksize=512):
    files = []
    print("Total number of shapes: ",len(shapes))

    if not os.path.isdir(out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)

    for idx,s in shapes.iterrows():
        print("Shape: ",idx)
        ofile = mosaic_to_raster_mp_queue_memory_multi(dataset_path,s,net,out_path,device_ids,bs,
                        num_workers,pin_memory,compress,blocksize)

        if ofile is not None:
            files.append(ofile)
        else:
            print("Error for ",idx)
    
    if len(shapes) > 1:
        vrt_file = os.path.join(out_path,"tmp_vrt.vrt")
        out_file = os.path.join(out_path,"mask_"+time.strftime("%d_%m_%Y_%H%M%S")+".tif")
        start = time.time()
        gdal.SetConfigOption("GDAL_CACHEMAX","1024")
        gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE","128")
        gdal.SetConfigOption("GDAL_TIFF_INTERNAL_MASK","True")
        
        vrt = gdal.BuildVRT(vrt_file,files)
        vrt = None
        
        #options = "-of COG -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co COMPRESS=DEFLATE -co BLOCKSIZE="+str(blocksize)

        options = "-of Gtiff -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co COMPRESS=DEFLATE -co TILED=YES \
                     -co BLOCKXSIZE="+str(blocksize)+" -co BLOCKYSIZE="+str(blocksize)

        ds = gdal.Translate(out_file,vrt_file,options=options)
        ds = None
        os.remove(vrt_file)
        end = time.time()
        #os.system("gdal_translate -of GTiff -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co COMPRESS=DEFLATE -co TILED=YES -co COPY_SRC_OVERVIEWS=YES " + vrt_file + " " + out_file)
        
        for i in files:
            os.remove(i)
        print(f"Created Tif file in {end-start} seconds: {out_file}")
   
    


def mosaic_to_raster_mp_queue_memory_multi(dataset_path,shape,net,out_path,device_ids,bs=16,
                                    num_workers=4,pin_memory=True,compress="deflate",blocksize=512,nodata=0):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    if not torch.cuda.is_available():
        raise Exception("No Cuda device available!")

    if device_ids == "all":
        world_size = torch.cuda.device_count()
        device_ids = list(range(world_size))
    elif type(device_ids) == list:
        world_size = len(device_ids)
    elif type(device_ids) == int:
        device_ids = [device_ids]
        world_size = 1
    elif (device_ids is None) or device_ids == "cpu":
        device_ids = [torch.device("cpu")]
        world_size = 1
        

    n = int(np.prod(shape["grid_shape"]))
    n_batches = int(np.ceil(n / bs))

    memfile = create_memfile(shape,compress,blocksize,nodata) #create_files(shapes,out_path,compress,blocksize) #

    data_queue = mp.JoinableQueue(500)
    in_queue = mp.JoinableQueue(500)#mp.JoinableQueue(100)
    out_queue = mp.JoinableQueue(10000)# mp.JoinableQueue(10000)
    #queue = mp.Queue(500)#mp.Queue(500)#mp.JoinableQueue(1000)
    event = mp.Event()

    processes = []

    #Start data loading process
    p = mp.Process(target=load_data_queue, args=(dataset_path,data_queue,bs,num_workers,event))
    p.start()     
    processes.append(p)

    for rank in range(world_size):
        p = mp.Process(target=run_inference_queue, args=(rank,device_ids,net,num_workers,pin_memory,data_queue,in_queue,event,n_batches,shape["start_idx"]))
        p.start()     
        processes.append(p)

    consumers = [mp.Process(target=queue_consumer, args=(in_queue,out_queue,shape), daemon=False)
                 for _ in range(world_size)]
    
    for p in consumers:
         p.start()

    complete = True
    c = 0
    pid = os.getpid()
    print("Queue PID: ",pid)

    with tqdm(total=n,position=0) as pbar:
        while (c < n) and (complete == True):
                d = out_queue.get()
                if type(d[1]) == str:
                    if d[1] == "ERROR":
                        complete = False
                else:
                    memfile.write(d[1],window=d[0],indexes=1)
                    c += 1
                    pbar.update(1)
                    if (c % 10000) == 0:
                        print("In-Queue length: ",in_queue.qsize())
                        print("Out-Queue length: ",out_queue.qsize())
                        print("Data-Queue length: ",data_queue.qsize())
                        print("Memory allocated (MB): ",psutil.Process().memory_info().rss / (1024 * 1024))
                del d
                out_queue.task_done()

    for _ in range(world_size): #stop consumer 
        in_queue.put([None,"QUIT"])

    time.sleep(1)
    in_queue.close()
    out_queue.close()
    event.set()
    #context.join()
    for p in processes:
        p.terminate()#p.join()

    stop_child_pids(pid) # to clean up memory 
    time.sleep(1)

    out_file = None
    if complete:
        start = time.time()
        out_meta = shape["sat_meta"]
        out_file = os.path.join(out_path,shape["name"]+".tif")
        cog_translate(
            memfile,
            out_file,
            out_meta
        )


        memfile.close()
        end = time.time()
        print(f"INFO: Written {out_file} in {end-start:.3f} seconds")
    return out_file


def run_inference_queue(rank,device_ids,net,num_workers,pin_memory,data_queue,out_queue,event,n_batches,start_idx):
    try:
        mp_context = "fork"
        #torch.set_num_threads(1) #prevent memory leakage
        if num_workers == 0:
            mp_context = None

        device_id = device_ids[rank]
        if device_id != torch.device("cpu"):
            print("Start GPU:",device_id)
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
            torch.cuda.set_device(device_id)
        else:
            print("Start on CPU!")
            pin_memory = False

        dataset = QueueDataset(data_queue,n_batches)
        dl = DataLoader(dataset,batch_size=1,num_workers = 4,pin_memory=pin_memory,multiprocessing_context=mp_context) # TODO hardcoded num workers -> more add two different params for workers -> one for loading queue and this 

        net = net.to(device_id)
        net.eval()
        for batch in dl:
            with torch.no_grad():
                if not torch.is_tensor(batch):
                    x,idx = batch
                    x = x.squeeze(0)
                    
                    x = x.float().to(device_id,non_blocking=True)
                    b_idx = int(idx[0]) - start_idx
                    out = net(x)
                    out = F.softmax(out,dim=1)
                    out = torch.argmax(out,dim=1)
                    out = out.byte().cpu()
                    out_queue.put([b_idx,out])
        event.wait()
    except Exception as e:
        print(f"Error: Job {device_id} - {e}")
        out_queue.put([device_id,"ERROR"])
        event.wait()

def load_data_queue(dataset_path,queue,bs,num_workers,event):
    dataset =  SatInferenceDataset(dataset_path=dataset_path)
    dl = DataLoader(dataset,batch_size=bs,num_workers = num_workers,pin_memory=False,multiprocessing_context="fork")
    
    for batch in dl:
        x,idx = batch
        queue.put([x,idx[0]])
    event.wait()

def queue_consumer(in_queue,out_queue,shape):
    complete = True
    while complete == True:
        d = in_queue.get()
        if type(d[1]) == str:
            if d[1] =="QUIT":
                in_queue.task_done()
                return
            out_queue.put(d)
        else:
            unpatchify_window_queue(shape,d[1].numpy(),d[0],out_queue)
        del d
        in_queue.task_done()

def custom_collate_fn(data):
    x,idx = zip(*data)
    x = np.stack(x)
    idx = np.stack(idx)
    del data
    return x,idx

def create_memfile(shape,compress,blocksize,nodata):
    with rasterio.Env(GDAL_CACHEMAX=1024,GDAL_TIFF_INTERNAL_MASK=True,GDAL_TIFF_OVR_BLOCKSIZE=128): #TODO change to variable
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
                        #"predictor":2,
                        "NUM_THREADS":"ALL_CPUS"})
        mfile = memory.open(**out_meta)
        return mfile

def create_files(shapes,out_path,compress,blocksize):
    memfiles = []
    for _,shp in shapes.iterrows():
        out_transform = shp["transform"]
        out_meta = shp["sat_meta"]
        out_file = os.path.join(out_path,shp["name"]+".tif")
        height = shp["height"]
        width = shp["width"]
        out_meta.update({"driver": "GTiff",
                        "count":1,
                        "overwrite":True,
                        "height": height,
                        "width": width,
                        "transform": out_transform,
                        "compress":compress,
                        "tiled":True,
                        "blockxsize":blocksize, 
                        "blockysize":blocksize,
                        "BIGTIFF":'YES',
                        "NUM_THREADS":"ALL_CPUS"})
        mfile = rasterio.open(out_file,"w",**out_meta)
        memfiles.append(mfile)
    return memfiles

def stop_child_pids(pid):
    try:
      parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
      return
    children = parent.children(recursive=True)
    for process in children:
      process.send_signal(signal.SIGTERM)

