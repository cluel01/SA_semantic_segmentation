import cv2
import numpy as np

from rasterio.enums import Resampling
from rasterio.io import MemoryFile

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

def resample_raster(sat_arr,transform,meta,resampling_factor,resampling_method):
    # rescale the metadata
    width = sat_arr.shape[-1]
    height = sat_arr.shape[-2]
    org_arr = sat_arr.copy()
    if resampling_method == "bilinear":
        resampling_method = Resampling.bilinear
    elif resampling_method == "cubic":
        resampling_method = Resampling.cubic
    elif resampling_method == "nearest":
        resampling_method = Resampling.nearest

    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(sat_arr)
            data = dataset.read(
            out_shape=(
                    dataset.count,
                    int(dataset.height * resampling_factor),
                    int(dataset.width * resampling_factor)
                ),
                resampling=Resampling.bilinear
            )

    transform = transform * transform.scale(
        (width / data.shape[-1]),
        (height / data.shape[-2])
    )

    return data,transform,org_arr