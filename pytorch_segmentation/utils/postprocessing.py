import queue
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
#from torch.nn.parallel import DistributedDataParallel as DDP
import rasterio
from rasterio.windows import Window
import os
from tqdm import tqdm
import time
from shapely.geometry import box
import geopandas as gpd
import pyproj
from shapely.ops import transform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import pickle

from .sampler import DistributedEvalSampler,VirtualMMAP
from ..models.unet_model import UNet

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



    

def unpatchify_window(dataset,shape,patches,out_path,compress):
    out_file = os.path.join(out_path,shape["name"]+".tif")
    out_transform = shape["transform"]
    start_idx = shape["start_idx"]
    ny,nx = shape["grid_shape"]
    ypad,ypad_extra,xpad,xpad_extra = shape["padding"]

    print("Write file: ",out_file)

    out_meta = dataset.sat_meta
    height = shape["height"]
    width = shape["width"]
    out_meta.update({"driver": "GTiff",
                    "count":1,
                    "height": height,
                    "width": width,
                    "transform": out_transform,
                    "compress":compress})

    i = start_idx
    col_off,row_off = 0,0
    with rasterio.open(out_file, "w", **out_meta) as dest:
        for y in range(ny):
            for x in range(nx):
                img = patches[i]

                if img is None:
                    print(i)
                co,ro = 0,0
                if (x != nx-1) and (y != ny-1):
                    cropped_img = img[ypad:img.shape[0]-ypad,xpad:img.shape[1]-xpad]
                    co = cropped_img.shape[1]
                if (x == nx-1) and (y != ny-1):
                    cropped_img = img[ypad:img.shape[0]-ypad,xpad:img.shape[1]-xpad_extra]
                    ro = cropped_img.shape[0]
                    co = -col_off
                if (x != nx-1) and (y == ny-1):
                    cropped_img = img[ypad:img.shape[0]-ypad_extra,xpad:img.shape[1]-xpad]
                    co = cropped_img.shape[1]
                if (x == nx-1) and (y == ny-1):
                    cropped_img = img[ypad:img.shape[0]-ypad_extra,xpad:img.shape[1]-xpad_extra]
                win = Window(row_off=row_off,col_off=col_off,
                            width=cropped_img.shape[1],height=cropped_img.shape[0])
                dest.write(cropped_img,window=win,indexes=1)
                i += 1
                col_off += co
                row_off += ro

    

#w: npatches width, h: npatches height, c: nchannel, x: patch_size x, y: patch_size y
#patches format: (npatches,(c),x,y)
#grid_shape format: (h,w)
#output format: (x,y)
def unpatchify(patches,grid_shape,padding):
    i = 0
    ypad,ypad_extra,xpad,xpad_extra = padding
    ny,nx = grid_shape

    if len(patches.shape) == 3:
        #only one channel
        patches = np.expand_dims(patches,1)

    out = []
    for y in range(ny):
        o = []
        for x in range(nx):
            img = patches[i]
            
            if (x != nx-1) and (y != ny-1):
                cropped_img = img[:,ypad:img.shape[1]-ypad,xpad:img.shape[2]-xpad]
            if (x == nx-1) and (y != ny-1):
                cropped_img = img[:,ypad:img.shape[1]-ypad,xpad:img.shape[2]-xpad_extra]
            if (x != nx-1) and (y == ny-1):
                cropped_img = img[:,ypad:img.shape[1]-ypad_extra,xpad:img.shape[2]-xpad]
            if (x == nx-1) and (y == ny-1):
                cropped_img = img[:,ypad:img.shape[1]-ypad_extra,xpad:img.shape[2]-xpad_extra]

            o.append(cropped_img.transpose(2,1,0))
            i += 1
        tmp = np.vstack(o).transpose(1,0,2)
        del o
        out.append(tmp)
        del tmp
    output_arr = np.vstack(out)
    del out
    return output_arr.transpose(2,0,1)



def raster_bounds_to_shape(raster_path,shape_path,crs="EPSG:4326"):
    ra = rasterio.open(raster_path)
    bounds  = ra.bounds

    old_crs = pyproj.CRS(ra.crs)
    new_crs = pyproj.CRS(crs)

    geom = box(*bounds)
    if old_crs != new_crs:
        project = pyproj.Transformer.from_crs(ra.crs, new_crs, always_xy=True).transform
        geom = transform(project, geom)

    df = gpd.GeoDataFrame({"id":1,"geometry":[geom]},crs=new_crs)
    df.to_file(shape_path)


def mosaic_to_raster_mp(dataset_path,net,out_path,device_ids,bs=16,out_size=256,num_workers=4,pin_memory=True,dtype="uint8",compress="deflate"):
    

    if not torch.cuda.is_available():
        raise Exception("No Cuda device available!")

    mmap_file = os.path.join(out_path,"tmp_mmap_"+time.strftime("%d_%m_%Y_%H%M%S"))
    #mmap = np.memmap(mmap_file, dtype=dtype, mode='w+', shape=mmap_shape)

    if device_ids == "all":
        world_size = torch.cuda.device_count()
        device_ids = list(range(world_size))
    elif type(device_ids) == list:
        world_size = len(device_ids)
    elif type(device_ids) == int:
        device_ids = [device_ids]
        world_size = 1


    mp.spawn(run_inference,
        args=(device_ids,world_size,dataset_path,net,mmap_file,out_size,bs,num_workers,pin_memory,dtype),
        nprocs=world_size,
        join=True)

    with open(dataset_path, 'rb') as inp:
        dataset = pickle.load(inp)
        with VirtualMMAP(world_size,len(dataset),mmap_file,dtype,out_size) as virt_mmap:
            for _,s in dataset.shapes.iterrows():
                unpatchify_window(dataset,s,virt_mmap,out_path,compress)

def run_inference(rank,device_ids,world_size,dataset_path,net,mmap_path,patch_size,bs,num_workers,pin_memory,dtype):
    device_id = device_ids[rank]
    print("Start GPU:",device_id)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    torch.cuda.set_device(device_id)
    with open(dataset_path, 'rb') as inp:
        dataset = pickle.load(inp)
        sampler = DistributedEvalSampler(dataset,num_replicas=world_size,rank=rank)
        #sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank)
        mmap_file = os.path.join(mmap_path+"_"+str(rank))
        mmap = np.memmap(mmap_file, dtype=dtype, mode='w+', shape=(len(sampler),patch_size,patch_size))
        #print(num_workers)

        dl = DataLoader(dataset,batch_size=bs,num_workers = num_workers,pin_memory=pin_memory,sampler=sampler,multiprocessing_context="fork")
        
        net = net.to(rank)
        net.eval()
        if rank == 0:
            dl = tqdm(dl,position=0)
        pointer = 0
        for batch in dl:
            with torch.no_grad():
                x = batch.to(rank)#[0].to(device)
                out = net(x)
                out = F.softmax(out,dim=1)
                out = torch.argmax(out,dim=1)
                out = out.cpu().numpy().astype(dtype)
                end_idx = pointer+len(out)
                mmap[pointer:end_idx] = out
                pointer = end_idx


def mosaic_to_raster_mp_queue(dataset_path,net,out_path,device_ids,mmap_shape,bs=16,num_workers=4,pin_memory=True,dtype="uint8",compress="deflate"):
    mp.set_start_method('spawn')

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

    queue = mp.Queue()
    event = mp.Event()
    context = mp.spawn(run_inference_queue,
        args=(device_ids,world_size,dataset_path,net,mmap_file,mmap_shape[1],bs,num_workers,pin_memory,dtype,queue,event),
        nprocs=world_size,
        join=False)

    active = list(range(world_size))
    while len(active) > 0:
        d = queue.get()
        if d[1] == "DONE":
            active.remove(d[0])
        else:
            start_idx = d[0]
            end_idx = start_idx + len(d[1])
            mmap[start_idx:end_idx] = d[1]
    event.set()
    context.join()
    

    with open(dataset_path, 'rb') as inp:
        dataset = pickle.load(inp)
        for _,s in dataset.shapes.iterrows():
            print(s)
            unpatchify_window(dataset,s,mmap,out_path,compress)
    os.remove(mmap_file)

def run_inference_queue(rank,device_ids,world_size,dataset_path,net,mmap_path,patch_size,bs,num_workers,pin_memory,dtype,queue,event):
    device_id = device_ids[rank]
    print("Start GPU:",device_id)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    torch.cuda.set_device(device_id)
    with open(dataset_path, 'rb') as inp:
        dataset = pickle.load(inp)
        sampler = DistributedEvalSampler(dataset,num_replicas=world_size,rank=rank)
        dl = DataLoader(dataset,batch_size=bs,num_workers = num_workers,pin_memory=pin_memory,sampler=sampler,multiprocessing_context="fork")

        net = net.to(rank)
        net.eval()
        if rank == 0:
            dl = tqdm(dl,position=0)
        for batch in dl:
            with torch.no_grad():
                x = batch["img"].to(rank)#[0].to(device)
                out = net(x)
                out = F.softmax(out,dim=1)
                out = torch.argmax(out,dim=1)
                out = out.cpu().numpy().astype(dtype)
                queue.put([batch["idx"][0],out])
        queue.put([rank,"DONE"])
        event.wait()