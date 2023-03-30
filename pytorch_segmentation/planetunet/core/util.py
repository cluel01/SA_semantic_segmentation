import os
import sys
import math
import shapely
import rasterio
import resource
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import geopandas as gpd
from functools import partial


def gdal_progress_callback(complete, _, data):
    """Callback function to show progress during GDAL operations such gdal.Warp() or gdal.Translate().

    Expects a tqdm progressbar in 'data', which is passed as the 'callback_data' argument of the GDAL method.
    'complete' is passed by the GDAL methods, as a float from 0 to 1
    """
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1


def raster_copy(output_fp, input_fp, mode="warp", resample=1, out_crs=None, bands=None, bounds=None, bounds_crs=None,
                multi_core=False, pbar=None, compress=False, cutline_fp=None, resample_alg=gdal.GRA_Bilinear):
    """ Copy a raster using GDAL Warp or GDAL Translate, with various options.

    The use of Warp or Translate can be chosen with 'mode' parameter. GDAL.Warp allows full multiprocessing,
    whereas GDAL.Translate allows the selection of only certain bands to copy.
    A specific window to copy can be specified with 'bounds' and 'bounds_crs' parameters.
    Optional resampling with bi-linear interpolation is done if passed in as 'resample'!=1.
    """

    # Common options
    base_options = dict(
        creationOptions=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER",
                         "NUM_THREADS=ALL_CPUS"],
        callback=gdal_progress_callback,
        callback_data=pbar
    )
    if compress:
        base_options["creationOptions"].append("COMPRESS=LZW")
    if resample != 1:
        # Get input pixel sizes
        raster = gdal.Open(input_fp)
        gt = raster.GetGeoTransform()
        x_res, y_res = gt[1], -gt[5]
        base_options["xRes"] = x_res / resample,
        base_options["yRes"] = y_res / resample,
        base_options["resampleAlg"] = resample_alg

    # Use GDAL Warp
    if mode.lower() == "warp":
        warp_options = dict(
            dstSRS=out_crs,
            cutlineDSName=cutline_fp,
            outputBounds=bounds,
            outputBoundsSRS=bounds_crs,
            multithread=multi_core,
            warpOptions=["NUM_THREADS=ALL_CPUS"] if multi_core else [],
            warpMemoryLimit=1000000000,  # processing chunk size. higher is not always better, around 1-4GB seems good
        )
        return gdal.Warp(output_fp, input_fp, **base_options, **warp_options)

    # Use GDAL Translate
    elif mode.lower() == "translate":
        translate_options = dict(
            bandList=bands,
            outputSRS=out_crs,
            projWin=[bounds[0], bounds[3], bounds[2], bounds[1]] if bounds is not None else None,
            projWinSRS=bounds_crs,
        )
        return gdal.Translate(output_fp, input_fp, **base_options, **translate_options)

    else:
        raise Exception("Invalid mode argument, supported modes are 'warp' or 'translate'.")


def get_raster_band_means_and_stds(raster_fp, bands=None, nodata=None, pbar=None):
    means = []
    stds = []
    with rasterio.open(raster_fp) as src:
        if bands is None:
            bands = range(1, src.count+1)
        if (nodata is None) and ('nodata' in src.profile):
            nodata = src.profile['nodata']
        for i_band in tqdm(bands, desc=pbar, disable=not pbar):
            overviews = src.overviews(i_band)
            max_overview = max(overviews) if (len(overviews) > 0) else 1
            shape = int(src.shape[0]/max_overview), int(src.shape[1]/max_overview)
            shape = shape[0], shape[1]
            band = src.read(i_band, out_shape=shape)
            if nodata is not None:
                band = band.astype(np.float32)
                band[band == nodata] = np.nan
            means.append(np.nanmean(band))
            stds.append(np.nanstd(band))
    return np.array(means), np.array(stds)


def resolution_metres2degrees(xres_metres, yres_metres, latitude):
    """Calculate the resolution in degrees equivalent to a desired resolution in metres."""

    xres_degrees = xres_metres / (111320 * math.cos(math.radians(abs(latitude))))  # at equator 1°lon ~= 111.32 km
    yres_degrees = yres_metres / 110540  # and        1°lat ~= 110.54 km
    return xres_degrees, yres_degrees


def get_driver_name(extension):
    """Get GDAL/OGR driver names from file extension"""
    if extension.lower().endswith("tif"):
        return "GTiff"
    elif extension.lower().endswith("jp2"):
        return "JP2OpenJPEG"
    elif extension.lower().endswith("shp"):
        return "ESRI Shapefile"
    elif extension.lower().endswith("gpkg"):
        return "GPKG"
    else:
        raise Exception(f"Unable to find driver for unsupported extension {extension}")


# Query a SQL database, supports postgres and sqlite/spatialite DBs
def query_db(sql_statement, db):
    if db.dialect.name == 'postgresql':
        return gpd.read_postgis(sql_statement, db, "geometry")
    elif db.dialect.name == 'sqlite':
        print(db, type(db))
        print(db.engine, type(db.engine))
        print(db.engine.url)

        # Have to do an ugly workaround via WKT format, because reading the WKB via geopandas/shapely wkb.loads or fails
        sql_statement = sql_statement.lower().replace("select ", "select ST_AsText(GEOMETRY) as geometry, ")
        result = db.execute(sql_statement)
        df = gpd.GeoDataFrame(result.fetchall(), columns=result.keys())
        for col in ['GEOMETRY', 'ogc_fid']:
            if col in df.columns:
                df = df.drop(columns=[col])
        df.geometry = df.geometry.apply(shapely.wkt.loads)
        return df
    else:
        raise ValueError(f"DB queries not yet implemented for {db.dialect.name} databases!")


def memory_limit(percentage: float):
    """Set soft memory limit to a percentage of total available memory."""
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * percentage), -1))
    # print(f"Set memory limit to {int(percentage*100)}% : {get_memory() * percentage/1024/1024:.2f} GiB")


def get_memory():
    """Get available memory from linux system.

    NOTE: Including 'SwapFree:' also counts cache as available memory (so remove it to only count physical RAM).
    This can still cause OOM crashes with a memory-heavy single thread, as linux won't necessarily move it to cache...
    """
    with open('/proc/meminfo', 'r') as mem_info:
        free_memory = 0
        for line in mem_info:
            if str(line.split()[0]) in ('MemFree:', 'Buffers:', 'Cached:', 'SwapFree:'):
                free_memory += int(line.split()[1])
    return free_memory


def memory(percentage):
    """Decorator to limit memory of a python method to a percentage of available system memory"""
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 / 1024
                print('Available memory: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator


def safe_delete(fp):
    """Safely delete a file, ignoring any errors or missing files."""
    try:
        if os.path.exists(fp):
            os.remove(fp)
    except OSError:
        pass

class AttributeDict(dict):
    def __init__(self, *args, **kwargs):    # Wrapper for dict that allows dot notation access for dicts
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_tile_id(mosaic_fp, identifier="ps_PSScene"):
    return ",".join([str(int(i)) for i in os.path.basename(mosaic_fp).split(identifier)[1].split("_")[2:4]])


def get_mosaic_id(mosaic_fp, identifier="ps_PSScene"):
    return "_".join(os.path.basename(mosaic_fp).split(identifier)[1].split("_")[2:6])


class Partial(partial):
    """Wrapper for partial that includes __name__ and __module__ of the original function.
    Allows the use of partials for tensorflow loss functions.
    """
    def __new__(cls, func, /, *args, **keywords):
        self = super().__new__(cls, func, *args, **keywords)
        self.__name__ = func.__name__
        self.__module__ = func.__module__
        return self
