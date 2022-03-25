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

def mosaic_to_raster(dataset,net,out_path,device,device_ids=None,bs=16,out_size=256,num_workers=4,pin_memory=True,dtype="uint8",compress="deflate"):
    dl = DataLoader(dataset,batch_size=bs,num_workers = num_workers,pin_memory=pin_memory)

    if (device_ids == "all") and (device.type == "cuda"):
        net = DataParallel(net)
        #net = DDP(net)
    elif (type(device_ids) == list) and (device.type == "cuda"):
        net = DataParallel(net,device_ids=device_ids)
        #net = DDP(net,device_ids=device_ids)

    net = net.to(device)
    net.eval()

    mmap_file = os.path.join(out_path,"tmp_mmap_"+time.strftime("%d_%m_%Y_%H%M%S"))
    mmap = np.memmap(mmap_file, dtype=dtype, mode='w+', shape=(len(dataset),out_size,out_size))

    pointer = 0
    for batch in tqdm(dl):
        with torch.no_grad():
            x = batch.to(device)#[0].to(device)
            out = net(x)
            out = F.softmax(out,dim=1)
            out = torch.argmax(out,dim=1)
            out = out.cpu().numpy().astype(dtype)

            end_idx = pointer + len(out)
            mmap[pointer:end_idx] = out
            pointer += len(out)

    for s in dataset.shapes:
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