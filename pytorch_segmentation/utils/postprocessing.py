import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import rasterio
import os

def mosaic_to_raster(dataset,net,out_path,device,bs=16,num_workers=4,dtype="uint8"):
    output = []
    dl = DataLoader(dataset,batch_size=bs,num_workers = num_workers)
    
    net.eval()
    for batch in dl:
        x = batch.to(device)#[0].to(device)
        out = net(x)
        out = F.softmax(out,dim=1)
        out = torch.argmax(out,dim=1)
        out = out.cpu().numpy().astype(dtype)
        output.append(out)
    imgs = np.vstack(output)

    for s in dataset.shapes:
        out_file = os.path.join(out_path,s["name"]+".tif")
        out_meta = dataset.sat_meta
        out_transform = s["transform"]
        start_idx = s["start_idx"]
        end_idx = start_idx+s["n"]

        out_image = unpatchify(imgs[start_idx:end_idx],s["grid_shape"],s["padding"])
        out_meta.update({"driver": "GTiff",
                    "count":1,
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
        with rasterio.open(out_file, "w", **out_meta) as dest:
            print("Write file ",out_file)
            dest.write(out_image)



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

