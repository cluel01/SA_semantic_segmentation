from shapely.geometry import box
import geopandas as gpd
import pyproj
from shapely.ops import transform
from shapely import geometry
import fiona
import rasterio
from rasterio.mask import mask
import numpy as np
import math
import os

def raster_bounds_to_shape(raster_path,shape_path,crs="EPSG:4326"):

    shapes = []
    if os.path.isfile(raster_path):
        raster_files = [raster_path]
    else:
        raster_files = [os.path.join(raster_path,i) for i in os.listdir(raster_path) if i.endswith(".tif")]
    for n,r in enumerate(raster_files):
        ra = rasterio.open(r)
        bounds  = ra.bounds

        old_crs = pyproj.CRS(ra.crs)
        new_crs = pyproj.CRS(crs)

        geom = box(*bounds)
        if old_crs != new_crs:
            project = pyproj.Transformer.from_crs(ra.crs, new_crs, always_xy=True).transform
            geom = transform(project, geom)
        shapes.append(geom)

    df = gpd.GeoDataFrame(geometry=shapes,crs=new_crs)
    df.to_file(shape_path)

def shapes_intersecting_with_raster(shape_path,raster_file):
    shape_idxs = []
    
    with rasterio.open(raster_file) as src:
        sat_shape = geometry.box(*src.bounds)
        nodata = src.meta["nodata"]
        with fiona.open(shape_path) as shapes:
            for i,shp in enumerate(shapes):
                s = geometry.shape(shp["geometry"])
                if sat_shape.intersects(s):
                    out_image, _ = mask(src, [s], crop=True)
                    if not np.all(out_image == nodata):
                        shape_idxs.append(i)
    return shape_idxs


def rotatedRectWithMaxArea(w, h, angle,degrees=True):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    degrees), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if degrees:
        angle = math.radians(angle)

    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    return wr,hr

