

import numpy as np
import rasterio
from rasterio.windows import Window
import os


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


def unpatchify_window_batch(shape,memfile,patches,start_idx):
    #So far only for one shape at once possible!
    shape_start_idx = shape["start_idx"]
    ny,nx = shape["grid_shape"]
    ypad,ypad_extra,xpad,xpad_extra = shape["padding"] 


    patch_size = patches[0].shape
    grid = np.arange(ny*nx).reshape(ny,nx)
    y,x = np.where(grid == start_idx)
    y = int(y)
    x = int(x)

    col_off = (patch_size[0] - xpad*2) * x  
    row_off = (patch_size[1] - ypad*2) * y
    for i in range(len(patches)):
        img = patches[i]

        co,ro = 0,0
        if (x != nx-1) and (y != ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad,xpad:img.shape[1]-xpad]
            co = cropped_img.shape[1]
            x += 1
        elif (x == nx-1) and (y != ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad,xpad:img.shape[1]-xpad_extra]
            ro = cropped_img.shape[0]
            co = -col_off
            x = 0
            y += 1
        elif (x != nx-1) and (y == ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad_extra,xpad:img.shape[1]-xpad]
            co = cropped_img.shape[1]
            x += 1
        elif (x == nx-1) and (y == ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad_extra,xpad:img.shape[1]-xpad_extra]
        win = Window(row_off=row_off,col_off=col_off,
                        width=cropped_img.shape[1],height=cropped_img.shape[0])

        memfile.write(cropped_img,window=win,indexes=1)
        col_off += co
        row_off += ro

def unpatchify_window_queue(shape,patches,start_idx,out_queue):
    #So far only for one shape at once possible!
    shape_start_idx = shape["start_idx"]
    ny,nx = shape["grid_shape"]
    ypad,ypad_extra,xpad,xpad_extra = shape["padding"] 

    patch_size = patches[0].shape
    grid = np.arange(ny*nx).reshape(ny,nx)
    y,x = np.where(grid == start_idx)
    y = int(y)
    x = int(x)

    col_off = (patch_size[0] - xpad*2) * x  
    row_off = (patch_size[1] - ypad*2) * y
    for i in range(len(patches)):
        img = patches[i]

        co,ro = 0,0
        if (x != nx-1) and (y != ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad,xpad:img.shape[1]-xpad]
            co = cropped_img.shape[1]
            x += 1
        elif (x == nx-1) and (y != ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad,xpad:img.shape[1]-xpad_extra]
            ro = cropped_img.shape[0]
            co = -col_off
            x = 0
            y += 1
        elif (x != nx-1) and (y == ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad_extra,xpad:img.shape[1]-xpad]
            co = cropped_img.shape[1]
            x += 1
        elif (x == nx-1) and (y == ny-1):
            cropped_img = img[ypad:img.shape[0]-ypad_extra,xpad:img.shape[1]-xpad_extra]
        win = Window(row_off=row_off,col_off=col_off,
                        width=cropped_img.shape[1],height=cropped_img.shape[0])

        out_queue.put([win,cropped_img])
        col_off += co
        row_off += ro

