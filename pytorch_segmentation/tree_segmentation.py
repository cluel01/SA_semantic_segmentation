import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import multiprocessing as mp
from time import time

from .utils.window_generator import window_generator

def segment_trees(raster_file,out_file,footprint=(3,3),min_distance=10,min_size=10,cachesize=10,n_cpus=1,
                patch_size=[512,512],driver=None):
    start_time = time()
    with rasterio.open(raster_file) as src:
        n_windows = np.ceil((np.array(src.shape) / np.squeeze(np.array(patch_size)))).astype(int)
        n = int(np.prod(n_windows))
        n_distributed = n // n_cpus
        rest = n % n_cpus

    with mp.Manager() as mgr:
        processes = []
        shps = mgr.dict()
        start = 0

        for rank in range(n_cpus):
            if rank < rest:
                end = n_distributed + 1 + start
            else:
                end = n_distributed + start
            #shps[rank] = mgr.list()
            pbar_id = int(np.median(list(range(n_cpus)))) #middle id 
            p = mp.Process(target=run_segmentation, args=(rank,pbar_id,raster_file,start,end,n_windows,patch_size,footprint,min_distance,min_size,cachesize,shps))
            p.start()     
            processes.append(p)
            start += end-start
        
        for p in processes:
            p.join()

        all_shps = []
        for shp in shps.values():
            all_shps.extend(shp)
        shps.clear()
        #del shps
    df = gpd.GeoDataFrame(crs=src.crs,geometry=all_shps)
    df.to_file(out_file,driver=driver)
    end_time = time()
    print(f"Finished in: {end_time-start_time} seconds")
    print("Number of segmented trees: ",len(df))
    return df

def run_segmentation(rank,pbar_id,raster_file,start,end,n_windows,patch_size,footprint,min_distance,min_size,cachesize,shps):
    if rank == pbar_id:
        pbar = tqdm(total=end-start,position=0)
    
    shp_list = []
    with rasterio.Env(GDAL_CACHEMAX=cachesize):
        with rasterio.open(raster_file) as src:
            h,w = src.shape
            window_gen = window_generator(w,h,patch_size,step_size=patch_size[0],start_end=(start,end))
            for window in window_gen:
            #for x,y in window_gen:
                #window = src.block_window(1,x,y)
                image = src.read(1, window=window)
                if np.all(image == 0):
                    if rank == pbar_id:
                        pbar.update(1)
                    continue
                labels = seg_watershed(image,footprint,min_distance)
                ids, counts = np.unique(labels,return_counts=True)
                filter_objects = []
                for i in ids[1:]:
                    if counts[i] >= min_size:
                        filter_objects.append(i)

                win_transform = src.window_transform(window)
                for i, (s, v) in enumerate(shapes(labels, mask=None, transform=win_transform)):
                    if v in filter_objects:
                        shp_list.append(shape(s))
                if rank == pbar_id:
                    pbar.update(1)
    if rank == pbar_id:
        pbar.close()
    shps[rank] = shp_list
    del shp_list
    return

def seg_watershed(image,footprint,min_distance):
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones(footprint), labels=image,min_distance=min_distance)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    return labels

# def window_generator(start,end,n_windows):   
#     total = int(np.prod(n_windows))
#     ny,nx = n_windows
#     grid = np.arange(total).reshape(ny,nx)
#     y,x = np.where(grid == start)
#     y = int(y)
#     x = int(x)

#     for _ in range(end-start):
#         yield([y,x])
#         if (x != nx-1) and (y != ny-1):
#             x += 1
#         elif (x == nx-1) and (y != ny-1):
#             x = 0
#             y += 1
#         elif (x != nx-1) and (y == ny-1):
#             x += 1
