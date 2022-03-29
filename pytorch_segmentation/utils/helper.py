from shapely.geometry import box
import geopandas as gpd
import pyproj
from shapely.ops import transform
import rasterio

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