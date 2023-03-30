import os
import time
import json
import pandas as pd
from datetime import timedelta
import numpy as np
from tqdm import tqdm

import rasterio
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box

from core.util import raster_copy, get_raster_band_means_and_stds
from core.frame_info import image_normalize


def get_areas_and_polygons(rectangle_fp, polygons_fp):
    """Read in the training rectangles and polygon shapefiles.

    Runs a spatial join on the two DBs, which assigns rectangle ids to all polygons in a column "index_right".
    """

    print("Reading training data shapefiles.. ", end="")
    start = time.time()

    # Read in areas and remove all columns except geometry
    areas = gpd.read_file(rectangle_fp)
    areas = areas.drop(columns=[c for c in areas.columns if c != "geometry"])

    # Read in polygons and remove all columns except geometry
    polygons = gpd.read_file(polygons_fp)
    polygons = polygons.drop(columns=[c for c in polygons.columns if c != "geometry"])

    print(f"Done in {time.time()-start:.2f} seconds. Found {len(polygons)} polygons in {len(areas)} areas.\n"
          f"Assigning polygons to areas..      ", end="")
    start = time.time()

    # Perform a spatial join operation to pre-index all polygons with the rectangle they are in
    polygons = gpd.sjoin(polygons, areas, op="intersects", how="inner")

    print(f"Done in {time.time()-start:.2f} seconds.")
    return areas, polygons


def get_images_with_training_areas(areas, image_source):
    """Get a list of input images and the training areas they cover.

    Returns a list of tuples with img path and its area ids, eg [(<img_path>, [0, 12, 17, 18]), (...), ...]
    """

    print("Assigning areas to input images..  ", end="")
    start = time.time()

    # Get all input image paths
    image_paths = image_source.get_image_fps()

    # Find the images that contain training areas
    images_with_areas = []
    for im in image_paths:

        # Get image bounds
        with rasterio.open(im) as raster:
            im_bounds = box(*raster.bounds)

        # Get training areas that are in this image
        areas_in_image = np.where(areas.envelope.intersects(im_bounds))[0]

        if len(areas_in_image) > 0:
            images_with_areas.append((im, [int(x) for x in list(areas_in_image)]))

    print(f"Done in {time.time()-start:.2f} seconds. Found {len(image_paths)} training "
          f"images of which {len(images_with_areas)} contain training areas.")

    return images_with_areas


def calculate_boundary_weights(polygons, scale):
    """Find boundaries between close polygons.

    Scales up each polygon, then get overlaps by intersecting. The overlaps of the scaled polygons are the boundaries.
    Returns geopandas data frame with boundary polygons.
    """
    # Scale up all polygons around their center, until they start overlapping
    # NOTE: scale factor should be matched to resolution and type of forest
    scaled_polys = gpd.GeoDataFrame({"geometry": polygons.geometry.scale(xfact=scale, yfact=scale, origin='center')})

    # Get intersections of scaled polygons, which are the boundaries.
    boundaries = []
    for i in range(len(scaled_polys)):

        # For each scaled polygon, get all nearby scaled polygons that intersect with it
        nearby_polys = scaled_polys[scaled_polys.geometry.intersects(scaled_polys.iloc[i].geometry)]

        # Add intersections of scaled polygon with nearby polygons [except the intersection with itself!]
        for j in range(len(nearby_polys)):
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                boundaries.append(scaled_polys.iloc[i].geometry.intersection(nearby_polys.iloc[j].geometry))

    # Convert to df and ensure we only return Polygons (sometimes it can be a Point, which breaks things)
    boundaries = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(boundaries)}).explode()
    boundaries = boundaries[boundaries.type == "Polygon"]

    # If we have boundaries, difference overlay them with original polygons to ensure boundaries don't cover labels
    if len(boundaries) > 0:
        boundaries = gpd.overlay(boundaries, polygons, how='difference')
    if len(boundaries) == 0:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries
    

def get_vectorized_annotation(polygons, areas, area_id, xsize, ysize):
    """Get the annotation as a list of shapes with geometric properties (center, area).

    Each entry in the output dictionary corresponds to an annotation polygon and has the following info:
        - center: centroid of the polygon in pixel coordinates
        - area(m): surface covered by the polygon in meters
        - area(px): surface covered by the polygon in pixels on the output frame
        - pseudo_radius(m): radius of the circle with the same area as the polygon in meters
        - pseudo_radius(px): radius of the circle with the same area as the polygon in pixels on the output frame
        - geometry: list of polygon points in pixel coordinates
    """
    # Find tree centers and pseudo-radius for this area
    isinarea = polygons[polygons.within(box(*areas.bounds.iloc[area_id]))]

    # Explode geodf to avoid handling multipolygons
    isinarea = isinarea.explode(column="geometry", index_parts=False)

    # Convert to equal area projection to compute area
    isinarea_ea = isinarea.to_crs(epsg=6933)
    isinarea.loc[:, "area(m)"] = isinarea_ea.area
    isinarea.loc[:, "pseudo_radius(m)"] = isinarea_ea.area.apply(lambda x: np.sqrt(x / np.pi))

    # Get transform from pixel coordinates to real-world coordinates
    bounds = areas.iloc[[area_id]].to_crs(epsg=6933).bounds.iloc[0]
    trsfrm = rasterio.transform.from_bounds(*bounds, xsize, ysize)

    # Deduce ground resolution
    gr = np.mean([np.abs(trsfrm.a), np.abs(trsfrm.e)])
    isinarea.loc[:, "area(px)"] = isinarea["area(m)"] / (gr**2)
    isinarea.loc[:, "pseudo_radius(px)"] = isinarea["pseudo_radius(m)"] / gr

    # Invert to get transform from real world coordinates to pixel coordinates
    trsfrm = ~trsfrm
    trsfrm = [element for tupl in trsfrm.column_vectors for element in tupl]
    isinarea.loc[:, "geometry"] = isinarea_ea["geometry"].affine_transform(trsfrm[:6])
    isinarea.loc[:, "center"] = isinarea.centroid
    isinarea.loc[:, "center"] = isinarea["center"].apply(lambda p: (p.x, p.y))
    isinarea.loc[:, "geometry"] = isinarea["geometry"].apply(lambda x: list(x.exterior.coords))

    # Convert dataframe to dict
    isinarea.drop(labels=["index_right"], inplace=True, axis=1)
    isinarea = pd.DataFrame(isinarea)
    dic = isinarea.to_dict(orient="records")
    return dic


def preprocess_all(config):
    """Run preprocessing for all training data."""

    print("Starting preprocessing.")
    start = time.time()

    # Create output folder
    output_dir = os.path.join(config.preprocessed_base_dir, time.strftime('%Y%m%d-%H%M')+'_'+config.run_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read in area and polygon shapefiles
    areas, polygons = get_areas_and_polygons(config.training_rectangles_fp, config.training_polygons_fp)

    # Scan input images and find which images contain which training areas
    images_with_areas = get_images_with_training_areas(areas, config.training_images_src)

    # For each input image, get all training areas in the image
    for im_path, area_ids in tqdm(images_with_areas, "Processing images with training areas", position=1):

        # Get min max per band
        if config.normalize_method == "tile_norm":
            means, stds = get_raster_band_means_and_stds(im_path)

        # For each area, extract the image channels and write img and annotation channels to a merged file
        for area_id in tqdm(area_ids, f"Extracting areas for {os.path.basename(im_path)}", position=0):

            # Extract the part of input image that overlaps training area, with optional resampling
            extract_ds = raster_copy("/vsimem/extracted", im_path, mode="translate", bounds=areas.bounds.iloc[area_id],
                                     resample=config.resample_factor, bands=config.training_images_src.bands)

            # Create new  raster with two extra bands for labels and boundaries (float because we normalise im to float)
            n_bands = len(config.preprocessing_bands)
            mem_ds = gdal.GetDriverByName("MEM").Create("", xsize=extract_ds.RasterXSize, ysize=extract_ds.RasterYSize,
                                                        bands=n_bands + 2, eType=gdal.GDT_Float32)
            mem_ds.SetProjection(extract_ds.GetProjection())
            mem_ds.SetGeoTransform(extract_ds.GetGeoTransform())

            # Normalise image bands of the extract and write into new raster
            for i in range(1, n_bands):
                mem_ds.GetRasterBand(i).WriteArray(image_normalize(extract_ds.GetRasterBand(i).ReadAsArray(), "ref_stretch"))
            mem_ds.GetRasterBand(n_bands).WriteArray(extract_ds.GetRasterBand(n_bands).ReadAsArray())
            # Write annotation polygons into second-last band       (GDAL only writes the polygons in the area bounds)
            polygons_fp = os.path.join(config.training_data_dir, config.training_polygon_fn)
            gdal.Rasterize(mem_ds, polygons_fp, bands=[n_bands+1], burnValues=[1], allTouched=config.rasterize_borders)

            # Get boundary weighting polygons for this area and write into last band
            polys_in_area = polygons[polygons.index_right == area_id]           # index_right was added in spatial join
            calculate_boundary_weights(polys_in_area, scale=config.boundary_scale).to_file("/vsimem/weights")
            gdal.Rasterize(mem_ds, "/vsimem/weights", bands=[n_bands+2], burnValues=[1], allTouched=True)
            
            if config.get_json:
                # Get annotation dict for current frame, save
                dic = get_vectorized_annotation(polygons, areas, area_id, extract_ds.RasterXSize, extract_ds.RasterYSize)
                output_fp = os.path.join(output_dir, f"{area_id}.json")
                with open(output_fp, 'w') as fp:
                    json.dump(dic, fp)
                    
            # Write extracted area to disk
            output_fp = os.path.join(output_dir, f"{area_id}.tif")
            if not os.path.exists(output_fp):
                gdal.GetDriverByName("GTiff").CreateCopy(output_fp, mem_ds, 0)

            else:
                # Special case: area is split across multiple images, need to merge the frames
                gdal.Translate("/vsimem/existing.tif", output_fp)   # copy to memory to avoid circular VRT read/write
                gdal.GetDriverByName("GTiff").CreateCopy("/vsimem/additional.tif", mem_ds, 0)
                gdal.BuildVRT("/vsimem/merged.vrt", ["/vsimem/existing.tif", "/vsimem/additional.tif"])
                gdal.Translate(output_fp, "/vsimem/merged.vrt")     # write merged one back to file

    if len(areas) > len(os.listdir(output_dir)):
        print(f"WARNING: Training images not found for {len(areas)-len(os.listdir(output_dir))} areas!")

    print(f"Preprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
