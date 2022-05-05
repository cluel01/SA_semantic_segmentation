#from rio_cogeo.cogeo import cog_translate
import psutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
import rasterio
# from rasterio.enums import Resampling
# from rasterio.vrt import WarpedVRT
from osgeo import gdal
import os
from tqdm import tqdm
import time
import torch.multiprocessing as mp
from .data.sampler import DistributedEvalSampler
from .data.inference_dataset import SatInferenceDataset
from .utils.unpatchify import unpatchify,unpatchify_window,unpatchify_window_batch,unpatchify_window_queue
from .utils.cog_translate import cog_translate

import signal

#Wrapper function that can call all different versions
def mosaic_to_raster(dataset_path,shapes,net,out_path,device_ids,bs=16,
                    num_workers=4,pin_memory=True,compress="deflate",blocksize=512):
    files = []
    print("Total number of shapes: ",len(shapes))
    for idx,s in shapes.iterrows():
        print("Shape: ",idx)
        ofile = mosaic_to_raster_mp_queue_memory_multi(dataset_path,s,idx,net,out_path,device_ids,bs,
                        num_workers,pin_memory,compress,blocksize)
        files.append(ofile)
    
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
   
        # vrt = gdal.BuildVRT(vrt_file,files)
        # vrt = None
        # with rasterio.open(vrt_file) as src:
        #     with WarpedVRT(src,
        #                 resampling=Resampling.nearest) as vrt:
        #         out_meta = src.meta.copy()
        #         out_meta.update({"num_threads":"all_cpus","bigtiff":"yes","compress":compress,"blocksize":blocksize,"tiled":True})
        #         cog_translate(vrt,out_file,out_meta)
    


def mosaic_to_raster_dp(dataset,net,out_path,device_ids,bs=16,out_size=256,num_workers=4,pin_memory=True,dtype="uint8",compress="deflate"):
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        raise Exception("No Cuda device available!")

    mmap_file = os.path.join(out_path,"tmp_mmap_"+time.strftime("%d_%m_%Y_%H%M%S"))
    mmap = np.memmap(mmap_file, dtype=dtype, mode='w+', shape=(len(dataset),out_size,out_size))

    if device_ids == "all":
        world_size = torch.cuda.device_count()
        device_ids = list(range(world_size))
    elif type(device_ids) == list:
        world_size = len(device_ids)
    elif type(device_ids) == int:
        device_ids = [device_ids]
        world_size = 1

    net = DataParallel(net, device_ids=device_ids)
    net = net.to(device)
    net.eval()

    sampler = torch.utils.data.SequentialSampler(dataset)
    dl = DataLoader(dataset,sampler=sampler,batch_size=bs,
                    num_workers=num_workers,pin_memory=pin_memory)

    pointer = 0
    for batch in tqdm(dl):
        with torch.no_grad():
            x = batch.to(device)#[0].to(device)
            out = net(x)
            out = F.softmax(out,dim=1)
            out = torch.argmax(out,dim=1)
            out = out.cpu().numpy().astype(dtype)
            end_idx = pointer+len(out)
            mmap[pointer:end_idx] = out
            pointer = end_idx

    for _,s in dataset.shapes.iterrows():
        unpatchify_window(dataset,s,mmap,out_path,compress)
    os.remove(mmap_file)

def mosaic_to_raster_mp_queue(dataset_path,net,out_path,device_ids,mmap_shape,bs=16,num_workers=4,pin_memory=True,dtype="uint8",compress="deflate"):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    #torch.set_num_threads(1)

    if not torch.cuda.is_available():
        raise Exception("No Cuda device available!")

    mmap_file = os.path.join(out_path,"tmp_mmap_"+time.strftime("%d_%m_%Y_%H%M%S"))
    mmap = np.memmap(mmap_file, dtype=dtype, mode='w+', shape=mmap_shape)

    if device_ids == "all":
        world_size = torch.cuda.device_count()
        device_ids = list(range(world_size))
    elif type(device_ids) == list:
        world_size = len(device_ids)
    elif type(device_ids) == int:
        device_ids = [device_ids]
        world_size = 1

    #manager = mp.Manager()
    #queue = manager.Queue() #mp.Queue(1000)
    queue  = mp.JoinableQueue(100)
    event = mp.Event()
    context = mp.spawn(run_inference_queue,
        args=(device_ids,world_size,dataset_path,net,mmap_file,mmap_shape[1],bs,num_workers,pin_memory,dtype,queue,event),
        nprocs=world_size,
        join=False)

    complete = True
    active = list(range(world_size))
    while (len(active) > 0) and (complete == True):
        while (not queue.empty()) and (complete == True):
            d = queue.get()
            if type(d[1]) == str:
                print("DONE ",str(d[0]))
                active.remove(d[0])
                if d[1] == "ERROR":
                    complete = False
            else:
                start_idx = d[0]
                end_idx = start_idx + len(d[1])
                mmap[start_idx:end_idx] = d[1].numpy()
            del d
            queue.task_done()
    event.set()
    context.join()
    
    if complete:
        with open(dataset_path, 'rb') as inp:
            dataset = SatInferenceDataset(dataset_path=dataset_path)
            for _,s in dataset.shapes.iterrows():
                unpatchify_window(dataset,s,mmap,out_path,compress)
    os.remove(mmap_file)

def mosaic_to_raster_mp_queue_memory(dataset_path,shape,shape_idx,net,out_path,device_ids,bs=16,
                                    num_workers=4,pin_memory=True,compress="deflate",blocksize=512):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    #torch.set_num_threads(1)

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

    n = np.prod(shape["grid_shape"])
    nbatches = int(np.ceil(n/bs))

    memfile = create_memfile(shape,compress,blocksize)

    queue = mp.JoinableQueue(500)
    #queue = mp.Queue(500)#mp.Queue(500)#mp.JoinableQueue(1000)
    event = mp.Event()
    context = mp.spawn(run_inference_queue,
        args=(device_ids,world_size,dataset_path,shape_idx,net,bs,num_workers,pin_memory,queue,event),
        nprocs=world_size,
        join=False)

    complete = True
    active = list(range(world_size))
    c = 0
    print("Queue PID: ",os.getpid())
    with tqdm(total=nbatches,position=0) as pbar:
        while (len(active) > 0) and (complete == True):
            while (not queue.empty()) and (complete == True):
                d = queue.get()
                if type(d[1]) == str:
                    print("DONE ",str(d[0]))
                    active.remove(d[0])
                    if d[1] == "ERROR":
                        complete = False
                else:
                    unpatchify_window_batch(shape,memfile,d[1].numpy(),d[0])
                    c += 1
                    pbar.update(1)
                    if (c % 500) == 0:
                        #pbar.update(100)
                        print("Queue length: ",queue.qsize())
                        print("Memory allocated (MB): ",psutil.Process().memory_info().rss / (1024 * 1024))
                del d
                queue.task_done()
    queue.close()
    event.set()
    context.join()

    if complete:
        
        start = time.time()
        out_file = os.path.join(out_path,shape["name"]+".tif")
        out_meta = shape["sat_meta"]
        out_meta.update({"driver":"COG"})
        with rasterio.open(out_file, "w", **out_meta) as dest:
            for _, window in memfile.block_windows():
                r = memfile.read(window=window)
                dest.write(r,window=window)
        memfile.close()
        end = time.time()
        print(f"INFO: Written {out_file} in {end-start:.3f} seconds")

def mosaic_to_raster_mp_queue_memory_multi(dataset_path,shape,shape_idx,net,out_path,device_ids,bs=16,
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

    n = int(np.prod(shape["grid_shape"]))

    memfile = create_memfile(shape,compress,blocksize,nodata) #create_files(shapes,out_path,compress,blocksize) #

    in_queue = mp.JoinableQueue(100)
    out_queue = mp.JoinableQueue(1000)
    #queue = mp.Queue(500)#mp.Queue(500)#mp.JoinableQueue(1000)
    event = mp.Event()
    context = mp.spawn(run_inference_queue,
        args=(device_ids,world_size,dataset_path,shape_idx,net,bs,num_workers,pin_memory,in_queue,event),
        nprocs=world_size,
        join=False)

    consumers = [mp.Process(target=queue_consumer, args=(in_queue,out_queue,shape), daemon=False)
                 for _ in range(world_size)]
    
    for p in consumers:
         p.start()

    complete = True
    active = list(range(world_size))
    c = 0
    pid = os.getpid()
    print("Queue PID: ",pid)

    with tqdm(total=n,position=0) as pbar:
        while (len(active) > 0) and (complete == True):
            while (not out_queue.empty()) and (complete == True):
                d = out_queue.get()
                if type(d[1]) == str:
                    print("DONE ",str(d[0]))
                    active.remove(d[0])
                    if d[1] == "ERROR":
                        complete = False
                else:
                    memfile.write(d[1],window=d[0],indexes=1)
                    c += 1
                    pbar.update(1)
                    if (c % 50000) == 0:
                        print("In-Queue length: ",in_queue.qsize())
                        print("Out-Queue length: ",out_queue.qsize())
                        print("Memory allocated (MB): ",psutil.Process().memory_info().rss / (1024 * 1024))
                del d
                out_queue.task_done()

    for _ in range(world_size): #stop consumer 
        in_queue.put([None,"QUIT"])

    time.sleep(1)
    in_queue.close()
    out_queue.close()
    event.set()
    context.join()

    stop_child_pids(pid) # to clean up memory 
    time.sleep(1)

    out_file = None
    # if complete:
    #     start = time.time()
    #     out_file = os.path.join(out_path,shape["name"]+".tif")
    #     out_meta = shape["sat_meta"]
    #     out_meta.update({"driver":"COG"})
    #     print(out_meta)
    #     with rasterio.open(out_file, "w", **out_meta) as dest:
    #         for _, window in memfile.block_windows():
    #             r = memfile.read(window=window)
    #             dest.write(r,window=window)
    #     memfile.close()
    #     end = time.time()
    #     print(f"INFO: Written {out_file} in {end-start:.3f} seconds")
    if complete:
        start = time.time()
        out_meta = shape["sat_meta"]
        out_file = os.path.join(out_path,shape["name"]+".tif")
        #out_meta.update({"driver":"COG"})
        #TODO do this without rio-cogeo package
        cog_translate(
            memfile,
            out_file,
            out_meta
            # in_memory=True,
            # quiet=True,
            # allow_intermediate_compression=True
        )


        memfile.close()
        end = time.time()
        print(f"INFO: Written {out_file} in {end-start:.3f} seconds")
    return out_file


def run_inference_queue(rank,device_ids,world_size,dataset_path,shape_idx,net,bs,num_workers,pin_memory,queue,event):
    try:
        mp_context = "fork"
        #torch.set_num_threads(1) #prevent memory leakage
        if num_workers == 0:
            mp_context = None

        device_id = device_ids[rank]
        print("Start GPU:",device_id)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        torch.cuda.set_device(device_id)
        dataset =  SatInferenceDataset(dataset_path=dataset_path)
        shape = dataset.shapes.iloc[shape_idx]
        start_idx = shape["start_idx"]
        end_idx  = start_idx + np.prod(shape["grid_shape"])
        sampler = DistributedEvalSampler(dataset,start_idx=start_idx,end_idx=end_idx,num_replicas=world_size,rank=rank)
        dl = DataLoader(dataset,batch_size=bs,num_workers = num_workers,pin_memory=pin_memory,sampler=sampler,multiprocessing_context=mp_context)
        #dl = DataLoader(dataset,batch_size=bs,collate_fn=custom_collate_fn,num_workers = num_workers,pin_memory=pin_memory,sampler=sampler,multiprocessing_context=mp_context)

        net = net.to(device_id)
        net.eval()
        for batch in dl:
            with torch.no_grad():
                x,idx = batch
                #x = torch.from_numpy(x).float().to(rank,non_blocking=True)#[0].to(device)
                x = x.float().to(device_id,non_blocking=True)
                b_idx = int(idx[0]) - start_idx
                out = net(x)
                out = F.softmax(out,dim=1)
                out = torch.argmax(out,dim=1)
                out = out.byte().cpu()
                queue.put([b_idx,out])
        queue.put([rank,"DONE"])
        event.wait()
    except Exception as e:
        print(f"Error: GPU {device_id} - {e}")
        queue.put([rank,"ERROR"])
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

