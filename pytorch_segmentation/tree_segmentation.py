from cv2 import threshold
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import rasterio
from rasterio.features import shapes
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import multiprocessing as mp

def segment_trees(raster_file,out_file,footprint=(3,3),min_distance=10,min_size=10,cachesize=100,n_cpus=1):
    with rasterio.open(raster_file) as src:
        n_windows = np.ceil((np.array(src.shape) / np.squeeze(np.array(src.block_shapes)))).astype(int)
        n = int(np.prod(n_windows))
        n_distributed = n // n_cpus
        rest = n % n_cpus

    with mp.Manager() as mgr:
        processes = []
        shapes = mgr.dict()
        start = 0

        for rank in range(n_cpus):
            if rank < rest:
                end = n_distributed + 1 + start
            else:
                end = n_distributed + start
            shapes[rank] = mgr.list()
            p = mp.Process(target=run_segmentation, args=(rank,raster_file,start,end,n_windows,footprint,min_distance,min_size,cachesize,shapes))
            p.start()     
            processes.append(p)
            start += end-start
        
        for p in processes:
            p.join()

        ntrees = 0
        all_df = None
        for k,shp in shapes.items():
            if len(shp) == 0 :
                continue  
            df = gpd.GeoDataFrame.from_features(shp)
            df.crs = src.crs
            if len(df) == 0:
                continue

            if all_df is None:
                all_df = df
            else:
                df["raster_val"] = df["raster_val"] + ntrees
                all_df = all_df.append(df,ignore_index=True)
            ntrees += len(df)
        all_df.to_file(out_file)
        print("Number of segmented trees: ",ntrees)
        return all_df

def run_segmentation(rank,raster_file,start,end,n_windows,footprint,min_distance,min_size,cachesize,shp):
    ntrees = 0
    if rank == 0:
        pbar = tqdm(total=end-start,position=0)
    window_gen = window_generator(start,end,n_windows)
    with rasterio.Env(GDAL_CACHEMAX=cachesize):
        with rasterio.open(raster_file) as src:
            for x,y in window_gen:
                window = src.block_window(1,x,y)
                image = src.read(1, window=window)
                if np.all(image == 0):
                    if rank == 0:
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
                        shp[rank].append({'properties': {'raster_val': ntrees}, 'geometry': s})
                        ntrees += 1
                if rank == 0:
                    pbar.update(1)
    if rank == 0:
        pbar.close()
    return

def seg_watershed(image,footprint,min_distance):
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones(footprint), labels=image,min_distance=min_distance)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    return labels

def window_generator(start,end,n_windows):   
    total = int(np.prod(n_windows))
    ny,nx = n_windows
    grid = np.arange(total).reshape(ny,nx)
    y,x = np.where(grid == start)
    y = int(y)
    x = int(x)

    for _ in range(end-start):
        yield([y,x])
        if (x != nx-1) and (y != ny-1):
            x += 1
        elif (x == nx-1) and (y != ny-1):
            x = 0
            y += 1
        elif (x != nx-1) and (y == ny-1):
            x += 1

