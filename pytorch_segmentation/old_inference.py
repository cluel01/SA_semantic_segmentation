
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
import rasterio
import os
from tqdm import tqdm
import time
import torch.multiprocessing as mp
from .data.sampler import DistributedEvalSampler
from .data.inference_dataset import SatInferenceDataset
from .utils.unpatchify import unpatchify,unpatchify_window,unpatchify_window_batch

def mosaic_to_raster(dataset,net,out_path,device_ids,bs=16,out_size=256,num_workers=4,pin_memory=True,dtype="uint8",compress="deflate"):
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


def mosaic_to_raster_mp_queue_memory(dataset_path,shapes,net,out_path,device_ids,bs=16,num_workers=4,pin_memory=True,compress="deflate"):
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

    memfiles = create_memfiles(shapes,compress)

    queue = mp.Queue(500)#mp.JoinableQueue(1000)
    event = mp.Event()
    context = mp.spawn(run_inference_queue,
        args=(device_ids,world_size,dataset_path,net,bs,num_workers,pin_memory,queue,event),
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
                unpatchify_window_batch(shapes,memfiles,d[1].numpy(),d[0])
            del d
            #queue.task_done()
    queue.close()
    event.set()
    context.join()
    
    if complete:
        for i,s in shapes.iterrows():
            out_file = os.path.join(out_path,s["name"]+".tif")
            out_meta = s["sat_meta"]
            with rasterio.open(out_file, "w", **out_meta,BIGTIFF='YES') as dest:
                for _, window in memfiles[i].block_windows():
                    r = memfiles[i].read(window=window)
                    dest.write(r,window=window)
            memfiles[i].close()


def run_inference_queue(rank,device_ids,world_size,dataset_path,net,bs,num_workers,pin_memory,queue,event):
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
        sampler = DistributedEvalSampler(dataset,num_replicas=world_size,rank=rank)
        #dl = DataLoader(dataset,batch_size=bs,num_workers = num_workers,pin_memory=pin_memory,sampler=sampler,multiprocessing_context=mp_context)
        dl = DataLoader(dataset,batch_size=bs,collate_fn=custom_collate_fn,num_workers = num_workers,pin_memory=pin_memory,sampler=sampler,multiprocessing_context=mp_context)

        net = net.to(rank)
        net.eval()
        if rank == 0:
            dl = tqdm(dl,position=0)
        for batch in dl:
            with torch.no_grad():
                x,idx = batch
                x = torch.from_numpy(x).float().to(rank,non_blocking=True)#[0].to(device)
                #x = x.float().to(rank,non_blocking=True)
                out = net(x)
                out = F.softmax(out,dim=1)
                out = torch.argmax(out,dim=1)
                out = out.byte().cpu()
                queue.put([int(idx[0]),out])
                del out
                del batch
                del x
                del idx
            #torch.cuda.empty_cache()
        queue.put([rank,"DONE"])
        event.wait()
    except Exception as e:
        print(f"Error: GPU {device_id} - {e}")
        queue.put([rank,"ERROR"])
        event.wait()



def custom_collate_fn(data):
    x,idx = zip(*data)
    x = np.stack(x)
    idx = np.stack(idx)
    del data
    return x,idx

def create_memfiles(shapes,compress):
    memfiles = []
    memory = rasterio.MemoryFile()
    for _,shp in shapes.iterrows():
        out_transform = shp["transform"]
        out_meta = shp["sat_meta"]
        
        height = shp["height"]
        width = shp["width"]
        out_meta.update({"driver": "GTiff",
                        "count":1,
                        "height": height,
                        "width": width,
                        "transform": out_transform,
                        "compress":compress,
                        "tiled":True,
                        "blockxsize":128,
                        "blockysize":128})
        mfile = memory.open(**out_meta,BIGTIFF='YES',NUM_THREADS="ALL_CPUS")
        memfiles.append(mfile)
    return memfiles

