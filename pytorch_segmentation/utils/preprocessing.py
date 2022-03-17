import cv2
import numpy as np

# pad so that the sliding window can work fine based on the image -> evenly padding to all sites
def pad_image_even(img,patch_size,overlap,dim=3,border_val=255):
    if dim==2:
        img = img[np.newaxis,...]

    step = patch_size[0]-overlap
    x_rest = (img.shape[2]-patch_size[0]) % step
    y_rest = (img.shape[1]-patch_size[1]) % step
    x_pad,y_pad = step-x_rest, step-y_rest
    left,right = int(np.ceil(x_pad/2)),int(np.floor(x_pad/2))
    top,bottom = int(np.ceil(y_pad/2)),int(np.floor(y_pad/2))
    image = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_CONSTANT,value=border_val)
    if dim == 2:
        return image,[top,bottom,left,right]
    return image.transpose(2,0,1),[top,bottom,left,right]

# With given padding: add padding to top and left edges so that it works for sliding window
def pad_image_topleft(img,patch_size,overlap,pad_size,border_val=255):
    step = patch_size[0]-overlap
    x_rest = (img.shape[2]-patch_size[0]+pad_size) % step 
    y_rest = (img.shape[1]-patch_size[1]+pad_size) % step

    x_pad,y_pad = step-x_rest, step-y_rest
    left,right = pad_size,x_pad
    top,bottom = pad_size,y_pad
    image = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_REPLICATE,value=border_val)
    return image.transpose(2,0,1),[top,bottom,left,right]

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