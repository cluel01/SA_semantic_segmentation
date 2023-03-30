import os
import glob
import json
import math
import re
import warnings
warnings.filterwarnings('ignore')
import scipy
import socket
import time
import traceback
import multiprocessing
from datetime import datetime

import skimage.transform
from itertools import product
from datetime import timedelta

import h5py
from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine

import rasterio
import rasterio.warp
import rasterio.mask
import rasterio.merge
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import Window, bounds, from_bounds

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # must be set before importing TF. for parallel predictions on one GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TF logs.  [Levels: 0->DEBUG, 1->INFO, 2->WARNING, 3->ERROR]
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from core.optimizers import get_optimizer
from core.frame_info import image_normalize
from core.util import memory, raster_copy, AttributeDict
from core.losses import accuracy, dice_coef, dice_loss, specificity, sensitivity, get_loss


def load_model(model_path, config):
    """Load a saved Tensorflow model into memory"""

    # Load and compile the model
    with h5py.File(model_path, 'r') as model_file:
        if "custom_meta" in model_file.attrs:
            try:
                custom_meta = json.loads(model_file.attrs["custom_meta"].decode("utf-8"))
            except:
                custom_meta = json.loads(model_file.attrs["custom_meta"])
            loss_fn, optimizer_fn = custom_meta['loss'], custom_meta['optimizer']
            tversky_alphabeta = (custom_meta['tversky_alpha'], custom_meta['tversky_beta'])
        else:
            loss_fn, optimizer_fn, tversky_alphabeta = config.loss_fn, config.optimizer, config.tversky_alphabeta

    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'tversky': get_loss(loss_fn, tversky_alphabeta),
                                                       'dice_coef': dice_coef, 'dice_loss': dice_loss,
                                                       'accuracy': accuracy, 'specificity': specificity,
                                                       'sensitivity': sensitivity},
                                       compile=False)
    model.compile(optimizer=get_optimizer(optimizer_fn), loss=get_loss(loss_fn, tversky_alphabeta),
                  metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])

    return model


def load_model_info():
    """Get and display config of the pre-trained model. Store model metadata as json in the output prediction folder."""

    # If no specific trained model was specified, use the most recent model
    if config.trained_model_path is None:
        model_fps = glob.glob(os.path.join(config.saved_models_dir, "*.h5"))
        config.trained_model_path = sorted(model_fps, key=lambda t: os.stat(t).st_mtime)[-1]

    # Print metadata from model training if available
    print(f"Loaded pretrained model from {config.trained_model_path} :")
    with h5py.File(config.trained_model_path, 'r') as model_file:
        if "custom_meta" in model_file.attrs:
            try:
                custom_meta = json.loads(model_file.attrs["custom_meta"].decode("utf-8"))
            except:
                custom_meta = json.loads(model_file.attrs["custom_meta"])
            print(custom_meta, "\n")

            # Save metadata in output prediction folder for future reference
            with open(os.path.join(config.prediction_output_dir, "model_custom_meta.json"), "a") as out_file:
                json.dump(custom_meta, out_file)
                out_file.write("\n\n")

            # Read patch size to use for prediction from model
            if config.prediction_patch_size is None:
                config.prediction_patch_size = custom_meta["patch_size"]


def get_images_to_predict():
    """ Get all input images to predict

    Either takes only the images specifically listed in a text file at config.to_predict_filelist,
    or all images in config.to_predict_dir with the correct prefix and file type
    """
    input_images = []
    if config.to_predict_filelist is not None and os.path.exists(config.to_predict_filelist):
        for line in open(config.to_predict_filelist):
            if os.path.isabs(line.strip()) and os.path.exists(line.strip()):                 # absolute paths
                input_images.append(line.strip())
            elif os.path.exists(os.path.join(config.to_predict_dir, line.strip())):          # relative paths
                input_images.append(os.path.join(config.to_predict_dir, line.strip()))

        print(f"Found {len(input_images)} images to predict listed in {config.to_predict_filelist}.")
    else:
        for root, dirs, files in os.walk(config.to_predict_dir):
            for file in files:
                if file.endswith(config.predict_images_file_type) and file.startswith(config.predict_images_prefix):
                    input_images.append(os.path.join(root, file))

        print(f"Found {len(input_images)} valid images to predict in {config.to_predict_dir}.")
    if len(input_images) == 0:
        raise Exception("No images to predict.")

    return sorted(input_images)


def merge_validity_masks(config, input_images):
    """Merge multiple configured validity masks (eg. land borders, water mask..) into a single mask file. """

    merged_validity_mask_fp = None
    if config.prediction_mask_fps is not None and len(config.prediction_mask_fps) > 0:

        # Initialise valid area as the combined extent of input images
        valid_area = gpd.GeoDataFrame({"geometry": [box(*rasterio.open(im).bounds) for im in input_images]},
                                      crs=rasterio.open(input_images[0]).crs).to_crs("EPSG:4326")

        # Overlay all validty masks, reading only the current validity area for faster loading
        for mask_fp in tqdm(config.prediction_mask_fps, desc=f"{'Merging validity masks':<25}", leave=False):
            df = gpd.read_file(mask_fp, mask=valid_area).to_crs("EPSG:4326")
            valid_area = gpd.overlay(valid_area, df, how="intersection", make_valid=False)

            if len(valid_area.geometry) == 0:
                raise Exception(f"No areas to predict. Prediction images have no overlap with validity mask {mask_fp}")

        # Write the merged validity mask to file, to be used in patch filtering and cropping
        merged_validity_mask_fp = os.path.join(config.prediction_output_dir, "validity_mask.gpkg")
        valid_area.to_file(merged_validity_mask_fp, driver="GPKG", crs="EPSG:4326")

    return merged_validity_mask_fp


def split_image_to_chunks(image_fp, output_file, config):
    """Split an image into smaller chunks for parallel processing, for lower memory usage and higher GPU utilisation.

    Setting  config.prediction_gridsize = (1, 1) means no splitting is done and the entire image is predicted at once.
    Returns a list of params used by predict_image() during parallel processing.
    """

    # Load validity mask if available
    validity_mask = None
    if config.validity_mask_fp is not None:
        validity_mask = gpd.read_file(config.validity_mask_fp)

    # Split image into grid of n_rows x n_cols chunks
    n_rows, n_cols = config.prediction_gridsize
    params = []
    with rasterio.open(image_fp) as raster:
        chunk_width, chunk_height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)

        # Create a list of chunk parameters used for parallel processing
        for i, j in product(range(n_rows), range(n_cols)):
            chunk_bounds = bounds(Window(chunk_width*j, chunk_height*i, chunk_width, chunk_height), raster.transform)

            # Exclude image chunks that are entirely outside valid area
            if validity_mask is None or np.any(validity_mask.intersects(box(*chunk_bounds))):
                params.append([image_fp, chunk_bounds, f"{output_file}_{chunk_width*j}_{chunk_height*i}.tif", config])

    return params


def get_patch_offsets(image, patch_width, patch_height, stride, validity_mask_fp=None):
    """Get a list of patch offsets based on image size, patch size and stride.

    If a validity mask is configured, patches outside the valid area are filtered out so they will not be predicted.
    """

    # Create iterator of all patch offsets, as tuples (x_off, y_off)
    patch_offsets = list(product(range(0, image.width, stride), range(0, image.height, stride)))

    # Optionally filter prediction area by a shapefile validity mask, with any mask intersecting patches not predicted
    if validity_mask_fp is not None:
        mask_polygons = gpd.read_file(validity_mask_fp, bbox=box(*image.bounds))
        offset_geom = [box(*bounds(Window(col_off, row_off, patch_width, patch_height), image.transform))
                        for col_off, row_off in patch_offsets]
        offsets_df = gpd.GeoDataFrame({"geometry": offset_geom, "col_off": list(zip(*patch_offsets))[0],
                                       "row_off": list(zip(*patch_offsets))[1]}, crs="EPSG:4326")
        offsets_df["unique_patch"] = offsets_df.index
        filtered_df = gpd.sjoin(offsets_df, mask_polygons, op="intersects", how="inner").drop_duplicates("unique_patch")
        patch_offsets = list(zip(filtered_df.col_off, filtered_df.row_off))

    return patch_offsets


def add_to_result(res, prediction, row, col, he, wi, operator='MAX'):
    """Add results of a patch to the total results of a larger area.

    The operator can be MIN (useful if there are too many false positives), or MAX (useful for tackling false negatives)
    """
    curr_value = res[row:row + he, col:col + wi]
    new_predictions = prediction[:he, :wi]
    if operator == 'MIN':
        curr_value[curr_value == -1] = 1  # For MIN case mask was initialised with -1, and replaced here to get min()
        resultant = np.fmin(curr_value, new_predictions)
    elif operator == 'MAX':
        resultant = np.fmax(curr_value, new_predictions)
    elif operator == 'MEAN':
        resultant = np.nanmean([curr_value, new_predictions], axis=0)
    else:  # operator == 'REPLACE':
        resultant = new_predictions
    res[row:row + he, col:col + wi] = resultant
    return res


def predict_one_batch(model, batch, batch_pos, mask, operator):
    """Predict one batch of patches with tensorflow, and add result to the output prediction. """

    tm = np.stack(batch, axis=0)
    prediction = model.predict(tm)
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis=-1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        mask = add_to_result(mask, p, row, col, he, wi, operator)
    return mask


def write_prediction_to_disk(detected_mask, profile, output_fp, config, threshold=None):
    """Write the output prediction mask to a raster file"""

    # For non-float formats, convert predictions to 1/0 with a given threshold
    if config.output_dtype != "float32" and threshold is not None:
        detected_mask[detected_mask < threshold] = 0
        detected_mask[detected_mask >= threshold] = 1

    # Set format specific profile options
    profile.update(height=detected_mask.shape[0], width=detected_mask.shape[1],
                   dtype=config.output_dtype, count=1, tiled=True, compress="LZW")
    if config.output_dtype == "uint8":
        profile.update(nodata=255)                            # for uint8, use 255 as nodata
    if config.output_dtype == "bool":
        profile.update(dtype="uint8", nbits=1, nodata=None)   # for binary geotiff, write as byte and pass NBITS=1

    # If we have a validity mask, mask by its cutline to get smooth edges in partially valid patches (no step blocks)
    if config.validity_mask_fp is not None:

        # We have to write to memory array first, because rasterio doesn't allow masking of an array directly..
        with rasterio.open(f"/vsimem/temp.tif", 'w', **profile) as out_ds:
            out_ds.write(detected_mask.astype(profile["dtype"]), 1)

        # Mask by valid areas
        with rasterio.open(f"/vsimem/temp.tif") as src:
            valid_areas = gpd.read_file(config.validity_mask_fp, bounds=src.bounds).geometry
            detected_mask, _ = rasterio.mask.mask(src, valid_areas, indexes=1)

    # Write prediction to file
    with rasterio.open(output_fp, 'w', **profile) as out_ds:
        out_ds.write(detected_mask.astype(profile["dtype"]), 1)


def resolution_degrees2metres(xres_degrees, yres_degrees, latitude):
    """Calculate the resolution in degrees equivalent to a desired resolution in metres."""
    xres_metres = xres_degrees * (111320 * math.cos(math.radians(abs(latitude))))  # at equator 1°lon ~= 111.32 km
    yres_metres = yres_degrees * 110540  # and        1°lat ~= 110.54 km
    return xres_metres, yres_metres


def add_additional_band(image_fp, image_bounds, out_fp, new_band, pbar_pos=0):
    # image_fp, out_fp, coverband_fp, image_bounds, average_res_m = params
    pbar = tqdm(total=5, desc=f"{'Adding coverband...':<25}", leave=False, position=pbar_pos, disable=True)

    # Read window of source image
    with rasterio.open(image_fp) as image_ds:
        image_window = from_bounds(*image_bounds, image_ds.transform)
        img = image_ds.read(window=image_window)
        pbar.update()

        # Read window of new band
        with rasterio.open(new_band["source_fp"]) as src:
            band_index = new_band["source_band"] if "source_band" in new_band.keys() else 1
            new_band_img = src.read(band_index, window=from_bounds(*image_bounds, src.transform))
        pbar.update()

        # Mask new band invalid values [optional]
        if "maskvals" in new_band.keys() and len(new_band["maskvals"]) > 0:
            mask = np.isin(new_band_img, new_band["maskvals"])
            new_band_img[mask] = 0

        # Scale new band values [optional]
        if "scale_factor" in new_band.keys() and new_band["scale_factor"] is not None:
            new_band_img = new_band_img.astype(np.float32) * new_band["scale_factor"]
        pbar.update()

        # Resample new band values [optional]
        if "average_to_resolution_m" in new_band.keys() and new_band["average_to_resolution_m"] is not None:
            try:
                scale = resolution_degrees2metres(*image_ds.res, 0)[1] / new_band["average_to_resolution_m"]
                new_band_img = skimage.transform.rescale(new_band_img, scale=scale, order=0, mode='reflect')
            except Exception:
                print(new_band_img.shape)
                assert False
        pbar.update()

        # Ensure new band is same resolution as input bands
        new_band_img = skimage.transform.resize(new_band_img, img.shape[1:], order=0, mode='reflect')

        # Insert extra band into merged img
        merged_img = np.concatenate([img, [new_band_img]], axis=0)

        # Write output merged image to file
        profile = image_ds.profile
        profile["count"] = profile["count"] + 1
        profile["transform"] = image_ds.window_transform(image_window)
        profile["width"] = img.shape[2]
        profile["height"] = img.shape[1]
        # profile["dtype"] = "uint16"
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(merged_img.astype(profile["dtype"]))

    pbar.update()
    return out_fp


def predict_image_patches(image, model, config, pos):

    # Get list of patch offsets to predict for this image
    patch_width, patch_height = model.patch_size
    stride = int((1 - (model.stride_percent / 100)) * patch_width)
    patch_offsets = get_patch_offsets(image, patch_width, patch_height, stride, config.validity_mask_fp)

    # Initialise prediction mask to zeros, or -1 for MIN operator
    prediction = np.zeros((image.height, image.width), dtype=np.float32)   # prediction is a float
    if model.prediction_operator == "MIN":
        prediction = prediction - 1

    # Load tensorflow model
    tf_model = load_model(model.model_fp, config)

    batch, batch_pos = [], []
    big_window = Window(0, 0, image.width, image.height)
    for col_off, row_off in tqdm(patch_offsets, position=pos, disable=True, leave=False, desc=f"Predicting {len(patch_offsets)}/"
                                 f"{math.ceil(image.width/stride)*math.ceil(image.height/stride)} patches..."):

        # Initialise patch with zero padding in case of corner images. size is based on number of channels
        patch = np.zeros((patch_height, patch_width, np.sum(model.channels_used)))  # + len(config.additional_input_bands)))

        # Load patch window from image, reading only necessary channels
        patch_window = Window(col_off=col_off, row_off=row_off, width=patch_width, height=patch_height).intersection(
            big_window)
        channel_list = list(np.where(model.channels_used)[0] + 1)                              # selected input bands
        # channel_list += [(image.count - i) for i in range(len(config.additional_input_bands))]  # extra bands at the end
        temp_im = image.read(channel_list, window=patch_window)
        temp_im = np.transpose(temp_im, axes=(1, 2, 0))      # switch channel order for TF

        # with rasterio.open(model.model_fp + f"_debug_{col_off}_{row_off}.tif", "w", **image.profile) as dst:
        #     dst.write(np.transpose(temp_im, axes=(2, 0, 1)), window=patch_window)

        # Normalize the image along the width and height i.e. independently per channel. Ignore nodata for normalization
        temp_im = image_normalize(temp_im, model.normalize_method, axis=(0, 1), nodata_val=image.nodatavals[0])

        # p = image.profile
        # p["dtype"] = "float32"
        # with rasterio.open(model.model_fp + f"_debugnorm_{col_off}_{row_off}.tif", "w", **p) as dst:
        #     dst.write(np.transpose(temp_im, axes=(2, 0, 1)), window=patch_window)
        # assert False

        # Add to batch list
        patch[:patch_window.height, :patch_window.width] = temp_im
        batch.append(patch)
        batch_pos.append((patch_window.col_off, patch_window.row_off, patch_window.width, patch_window.height))

        # Predict one batch at a time
        if len(batch) == model.pred_batch_size:
            prediction = predict_one_batch(tf_model, batch, batch_pos, prediction, model.prediction_operator)
            batch, batch_pos = [], []

    # Run once more to process the last partial batch (when image not exactly divisible by N batches)
    if batch:
        prediction = predict_one_batch(tf_model, batch, batch_pos, prediction, model.prediction_operator)

    return prediction


def merge_predictions(model_predictions, config):

    models = config.prediction_models
    max_shape = sorted([p.shape for p in model_predictions])[-1]
    for i in range(len(models)):
        pred = model_predictions[i]

        # Match shape of non-resampled predictions to max resampled shape
        if pred.shape != max_shape:
            pred = scipy.ndimage.zoom(pred, round(max_shape[0] / pred.shape[0]), order=0)
            if pred.shape != max_shape:
                pred = skimage.transform.resize(pred, max_shape, order=0, mode='reflect')
            assert pred.shape == max_shape

        # For non-float formats, convert predictions to 1/0 with a given threshold
        if config.output_dtype != "float32":
            pred = (pred >= models[i]["prediction_threshold"]) * 1.0

        model_predictions[i] = pred

    # Merge predictions by MIN or MAX ensemble
    predictions = np.array(model_predictions)
    method = config.ensemble_merge_mode
    if method is None or method == "":
        method = "MAX"
    if method.upper() in ["MIN", "MINIMUM"]:
        merged_prediction = np.nanmin(predictions, axis=0)
    elif method.upper() in ["MAX", "MAXIMUM"]:
        merged_prediction = np.nanmax(predictions, axis=0)
    # elif method.upper() in ["MEAN", "AVERAGE", "AVG"]:           # doesn't work with individual model pred threholds
    #     merged_prediction = np.nanmean(predictions, axis=0)
    else:
        raise ValueError(f"Unsupported ensemble predictions merging method: {method}")

    return merged_prediction


def predict_image(params):

    image_fp, predict_bounds, out_fp, config = params

    # Load the model
    pos = min(int(multiprocessing.current_process().name[-1].replace('s', '0')), config.prediction_workers) + 1

    # For jp2 compressed files, we first decompress in memory to speed up prediction and resampling
    if image_fp.lower().endswith(".jp2"):
        raster_copy("/vsimem/decompressed.tif", image_fp, multi_core=True, bounds=predict_bounds,
                    pbar=tqdm(total=100, disable=True, position=pos, leave=False, desc=f"{'Decompressing jp2':<25}"))
        image_fp = "/vsimem/decompressed.tif"

    # Add additional bands which may be configured
    if config.additional_input_bands is not None and len(config.additional_input_bands) > 0:
        for i in range(len(config.additional_input_bands)):
            merged_fp = f"/vsimem/with_new_band_{i}.tif"
            add_additional_band(image_fp, predict_bounds, merged_fp, config.additional_input_bands[i])
            image_fp = merged_fp

    # with rasterio.open(image_fp) as src:
    #     with rasterio.open(out_fp + "debug_img.tif", "w", **src.profile) as dst:
    #         dst.write(src.read())

    # Process all models
    model_predictions = []
    config.prediction_models = sorted(config.prediction_models, key=lambda x: x["resample_factor"])
    img = rasterio.open(image_fp, tiled=True, bounds=predict_bounds)

    for i in range(len(config.prediction_models)):

        model = AttributeDict(config.prediction_models[i])

        # Optionally resample the image in memory
        resample_factor = model["resample_factor"]
        if resample_factor != 1:
            raster_copy("/vsimem/resampled.tif", image_fp, resample=resample_factor, bounds=predict_bounds,
                        multi_core=True, pbar=tqdm(total=100, disable=True, position=pos, leave=False,
                        desc=f"Resampling x{resample_factor:<13}"))
            img = rasterio.open("/vsimem/resampled.tif", tiled=True, bounds=predict_bounds)

        model_preds_folder = os.path.join(os.path.dirname(out_fp), model.name)
        model_pred_out_fp = os.path.join(model_preds_folder, os.path.basename(out_fp))

        # Check for already existing predicted chunk
        if os.path.exists(model_pred_out_fp):
            # print(f"Skipping chunk prediction, loading from existing chunk prediction at {model_pred_out_fp}")
            prediction = rasterio.open(model_pred_out_fp).read(1)

        # Check for already existing predicted tile
        elif os.path.exists(str(model_pred_out_fp.split(".tif")[0]) + ".tif"):
            with rasterio.open(str(model_pred_out_fp.split(".tif")[0]) + ".tif") as src:
                # print(f"""Skipping chunk prediction, loading from existing tile prediction at
                #         {str(model_pred_out_fp.split(".tif")[0]) + ".tif"}""")
                prediction = src.read(1, window=from_bounds(*predict_bounds, transform=src.transform))

        # Predict this chunk
        else:
            prediction = predict_image_patches(img, model, config, pos)
        model_predictions.append(prediction)
        del model

        # Store individual model prediction
        if len(config.prediction_models) > 1:
            os.makedirs(model_preds_folder, exist_ok=True)
            write_prediction_to_disk(model_predictions[i], img.profile, model_pred_out_fp, config,
                                     config.prediction_models[i]["prediction_threshold"])

    # Merge model predictions into final prediction and write to file
    final_prediction = merge_predictions(model_predictions, config)
    write_prediction_to_disk(final_prediction, img.profile, out_fp, config)

    return out_fp


@memory(percentage=99)
def predict_all(conf):
    """Predict trees in all the files in the input image dir. """

    global config
    config = conf

    print("Starting prediction.")
    start = time.time()

    # Create folder for output predictions
    if config.prediction_output_dir is None:
        config.prediction_output_dir = os.path.join(config.predictions_base_dir, time.strftime('%Y%m%d-%H%M') + '_' + config.prediction_name)
    if not os.path.exists(config.prediction_output_dir):
        os.mkdir(config.prediction_output_dir)
    rasters_dir = os.path.join(config.prediction_output_dir, "rasters")
    if not os.path.exists(rasters_dir):
        os.mkdir(rasters_dir)

    # Load model info
    load_model_info()

    # Get list of images to analyse
    input_images = get_images_to_predict()

    # Combine possible multiple validity masks into one mask
    # config.validity_mask_fp = merge_validity_masks(config, input_images)

    # Process all input images
    for image_fp in tqdm(input_images, desc=f"{'Analysing images':<25}", position=0):

        ################ TODO

        # Get validity mask for this tile
        if config.prediction_masks_dir is not None:
            validity_mask = get_mosaic_file(image_fp, config.prediction_masks_dir, ".gpkg")
            if validity_mask is not None:
                print(f"Using prediction valid data mask at {validity_mask}")
                config.validity_mask_fp = validity_mask
            else:
                config.validity_mask_fp = None
                print(f"Predicting entire image, no validity mask found in {config.prediction_masks_dir}")

        ##################

        # Check if image has already been predicted
        output_file = os.path.join(rasters_dir, config.output_prefix +
                                   image_fp.split("/")[-1].replace(config.predict_images_file_type, ".tif"))
        if os.path.isfile(output_file) and not config.overwrite_existing:
            print(f"File already analysed, skipping {image_fp}")
            continue

        print(f"\nAnalysing {image_fp}")
        t0 = time.time()

        # Split into several smaller image chunks
        params = split_image_to_chunks(image_fp, output_file, config)
        if len(params) == 0:
            print(f"No parts of the image intersect validity mask, skipping {image_fp}")
            continue

        # Process image chunks in parallel
        chunk_fps = []
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=config.prediction_workers) as pool:
            with tqdm(total=len(params), desc="Processing image chunks", position=1, leave=False) as pbar:
                for result in pool.imap_unordered(predict_image, params, chunksize=1):
                    pbar.update()
                    if result:
                        chunk_fps.append(result)
                pbar.update()

        # Merge chunks back into one output raster
        print(f"\nWriting raster to {output_file}")
        gdal.BuildVRT(f"/vsimem/merged.vrt", chunk_fps)
        options = ["TILED=YES", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "NBITS=1" if config.output_dtype == "bool" else ""]
        gdal.Translate(output_file, f"/vsimem/merged.vrt", creationOptions=options)

        # Delete temp chunks
        for f in [fp for fp in chunk_fps if os.path.exists(fp)]:
            os.remove(f)
        print(f"Processed {image_fp} in: {str(timedelta(seconds=time.time() - t0)).split('.')[0]}\n")

    print(f"Prediction completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))  # doesn't even have to be reachable
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def update_config(config_a, config_b):
    for key in config_b.keys():
        config_a[key] = config_b[key]
    return AttributeDict(config_a)


def get_mosaic_file(mosaic_fp, folder, ext="", pattern=r"ps_PSScene_[0-9]{4}_(.{19})_"):
    try:
        fps = sorted(glob.glob(f"{folder.rstrip('/')}/*{ext}"))
        id = re.findall(pattern, mosaic_fp)[0]
        mosaics = [fp for fp in fps if re.findall(pattern, fp)[0] == id]
        if len(mosaics) > 1:
            print(f"!! WARNING: multiple mosaics for id {id} in folder {folder}: \n", "\n".join(mosaics))
        return mosaics[0]
    except IndexError:
        return None


def get_seconds_wait(current_time, start, end):
    now = current_time.tm_hour
    if end >= start and start < now < end:
        return 0
    elif end < start and (now > start or now < end):
        return 0
    else:
        return (((24 + start - now) % 24)*60 - current_time.tm_min)*60 - current_time.tm_sec


@memory(percentage=99)
def predict_all_database(conf):
    """Predict trees in all the files in the input image dir. """

    global config
    config = conf

    config.scheduler_end = (config.scheduler_end + 24) % 24
    print(f"Starting prediction job, will predict daily between {config.scheduler_start:02}:00h "
          f"and {config.scheduler_end:02}:00h.\n")

    # Connect to DB and get tiles to pred
    db = create_engine(config.database_url)

    # Set GPU identifier, with hostname and GPU id. to-do: add gpu name
    gpu_id = f"{get_ip()}:{f'GPU{config.selected_GPU}' if config.selected_GPU >= 0 else 'CPU'}"

    progress_bar, chunk_fps = tqdm(total=1, desc=f"{'Analysing images':<25}"), []
    while True:

        # Check scheduler time
        t_wait = get_seconds_wait(time.localtime(), config.scheduler_start, (24+config.scheduler_end) % 24-1)
        if t_wait > 0:
            print(f"\nSleeping... will start predicting again at {config.scheduler_start:02}:00h")
            time.sleep(t_wait)

        # Get tiles to predict
        tiles_to_predict = gpd.read_postgis(f"select * from predictions "
                                            f"where run_name = '{config.run_name}' and status = 'queued' "
                                            f"order by priority desc;", db, "geometry")
        if len(tiles_to_predict) == 0:
            if progress_bar.total == 1:
                print(f"No images found to predict for run_name {config.run_name} in database {config.database_url}")
            else:
                print(f"No more images found to predict in database, prediction run completed for {config.run_name}")
            break
        progress_bar.total = len(tiles_to_predict)
        progress_bar.refresh()

        # Start prediction of first tile
        tile = tiles_to_predict.iloc[0]
        image_fp = tile.mosaic_fp
        db.execute(f"update predictions "
                   f"set (status, gpu_id, time_start) = ('started', '{gpu_id}', '{datetime.now().isoformat()[:-7]}') "
                   f"where id = '{tile.id}' and run_name = '{config.run_name}'")
        status = "started"

        print(f"\nAnalysing {image_fp}")
        t0 = time.time()

        try:
            # Get config for this tile
            config = update_config(config.__dict__, json.loads(tile.config))
            # config.prediction_patch_size = config.patch_size   # TODO remove
            # config.trained_model_path = tile.model_fp

            # Create folder for output predictions if necessary
            if not os.path.exists(config.prediction_output_dir):
                os.mkdir(config.prediction_output_dir)
            rasters_dir = os.path.join(config.prediction_output_dir, "rasters")
            if not os.path.exists(rasters_dir):
                os.mkdir(rasters_dir)

            # Check if image has already been predicted - should never happen as we check status
            output_file = os.path.join(rasters_dir, config.output_prefix +
                                       os.path.basename(image_fp).replace(".tif", ".jp2").replace(".jp2", ".tif"))
            if os.path.isfile(output_file) and not config.overwrite_analysed_files:
                print(f"WARNING: File already analysed, setting status to complete for {tile.id} {image_fp}")
                db.execute(f"update predictions set status='completed' "
                           f"where id = '{tile.id}' and run_name = '{config.run_name}';")
                continue

            # Get validity mask for this tile
            if config.prediction_masks_dir is not None:
                validity_mask = get_mosaic_file(tile.mosaic_fp, config.prediction_masks_dir, ".gpkg")
                if validity_mask is not None:
                    print(f"Using prediction valid data mask at {validity_mask}")
                    config.validity_mask_fp = validity_mask
                else:
                    config.validity_mask_fp = None
                    print(f"Predicting entire image, no validity mask found in {config.prediction_masks_dir}")

            # Split into several smaller image chunks
            params = split_image_to_chunks(image_fp, output_file, config)
            if len(params) == 0:
                raise ValueError(f"No parts of the image intersect the validity mask {config.validity_mask_fp}")

            # Process image chunks in parallel
            chunk_fps = []
            multiprocessing.set_start_method("spawn", force=True)
            with multiprocessing.Pool(processes=config.prediction_workers) as pool:
                with tqdm(total=len(params), desc="Processing image chunks", position=1, leave=False) as pbar:
                    for result in pool.imap_unordered(predict_image, params, chunksize=1):
                        pbar.update()
                        if result:
                            chunk_fps.append(result)
                    pbar.update()

            # Merge chunks back into one output raster
            print(f"\nWriting raster to {output_file}")
            gdal.BuildVRT(f"/vsimem/merged.vrt", chunk_fps)
            options = ["TILED=YES", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "NBITS=1" if config.output_dtype == "bool" else ""]
            gdal.Translate(output_file, f"/vsimem/merged.vrt", creationOptions=options)

            # Also store individual model predictions
            if len(config.prediction_models) > 0:
                for i in range(len(config.prediction_models)):
                    d = os.path.join(os.path.dirname(output_file), config.prediction_models[i]["name"])
                    model_chunk_fps = [f.replace(os.path.dirname(f), d) for f in chunk_fps]
                    gdal.BuildVRT(f"/vsimem/merged.vrt", model_chunk_fps)
                    options = ["TILED=YES", "BIGTIFF=IF_SAFER", "COMPRESS=LZW",
                               "NBITS=1" if config.output_dtype == "bool" else ""]
                    out_fp = os.path.join(d, os.path.basename(output_file))
                    gdal.Translate(out_fp, f"/vsimem/merged.vrt", creationOptions=options)
                    for f in model_chunk_fps:
                        os.remove(f)

            # Delete temp chunks
            for f in chunk_fps:
                os.remove(f)

            # Update DB with completed status
            db.execute(f"update predictions "
                       f"set (status, gpu_id, output_fp, time_elapsed_m) = "
                       f"('completed', '{gpu_id}', '{output_file}', {int((time.time() - t0)/60*100)/100}) "
                       f"where id = '{tile.id}' and run_name = '{config.run_name}';")
            status = "completed"
            progress_bar.update()

        except Exception as err:
            print(f"ERROR: Exception on processing {image_fp}: ", err)
            print(traceback.format_exc())
            db.execute(f"update predictions "
                       f"set (status, gpu_id, msg) = "
                       f"('error', '{gpu_id}', $${err}$$) "
                       f"where id = '{tile.id}' and run_name = '{config.run_name}';")
            status = "error"
        finally:
            # Clean up after interrupted prediction and reset status to queued
            if status != "completed":
                for f in [fp for fp in chunk_fps if os.path.exists(fp)]:
                    os.remove(f)
            if status == "started":
                db.execute(f"update predictions "
                           f"set (status, gpu_id) = "
                           f"('queued', '{gpu_id}') "
                           f"where id = '{tile.id}' and run_name = '{config.run_name}' and status = 'started';")

        print(f"Processed {image_fp} in: {str(timedelta(seconds=int(time.time() - t0)))}\n")


    #print(f"Prediction completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
