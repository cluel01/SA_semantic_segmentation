import os
import glob
import h5py
import json
#from sqlalchemy import create_engine

#from core.util import query_db


class ConfigError(Exception):
    pass


class ImageSource:
    def __init__(self, file=None, folder=None, database=None, bands=(1, 2, 3, 4), filetype=".tif", prefix="",
                 db_query=None, db_table=None, db_column=None):
        self.file = file
        self.folder = folder
        self.database = database
        self.bands = bands
        self.filetype = filetype
        self.prefix = prefix
        self.db_query = db_query
        self.db_table = db_table
        self.db_column = db_column

        if file is not None:
            self.image_source_type = "file"
        if folder is not None:
            self.image_source_type = "folder"
        if database is not None:
            self.image_source_type = "database"

        self.validate_config()

    def get_image_fps(self):
        if self.image_source_type == "file":
            return [self.file]
        elif self.image_source_type == "folder":
            return glob.glob(f"{self.folder}/{self.prefix}*.{self.filetype.lstrip('.')}")
        # elif self.image_source_type == "database":
        #     db = create_engine(self.database)
        #     if self.db_query is not None:
        #         results = query_db(self.db_query, db)
        #     else:
        #         results = query_db(f"select * from {self.db_table}", db)
        #     return list(results[self.db_column])

    def validate_config(self):
        if sum(src is not None for src in [self.file, self.folder, self.database]) != 1:
            raise ConfigError("An ImageSource takes only one of 'filepath', 'folder' or 'database'.")
        if self.image_source_type == "file":
            if not os.path.exists(self.file):
                raise ConfigError(f"Input file not found: {self.file}")
        if self.image_source_type == "folder":
            image_fps = glob.glob(f"{self.folder}/{self.prefix}*.{self.filetype.lstrip('.')}")
            if len(image_fps) == 0:
                raise ConfigError(f"No {self.prefix}*.{self.filetype.lstrip('.')} input images found in {self.folder}")


class AdditionalInputBand:
    def __init__(self,
                 name,
                 image_source,
                 values_to_mask=None,
                 mask_to=0,
                 scale_factor=1,
                 average_to_resolution_m=None):
        self.name = name
        self.image_source = image_source
        self.values_to_mask = values_to_mask
        self.mask_to = mask_to
        self.scale_factor = scale_factor
        self.average_to_resolution_m = average_to_resolution_m

        self.validate_config()

    def validate_config(self):
        # TODO
        return


class PreprocessConfig:
    def __init__(self,
                 run_name,
                 training_rectangles_fp,
                 training_polygons_fp,
                 training_images_src,
                 preprocessed_base_dir,
                 normalize_method,
                 resample_factor=1,
                 additional_input_bands=None,
                 rasterize_label_borders=False,
                 add_frames_json=False,
                 enable_boundary_weights=False,
                 boundary_scale=1.5,
                 boundary_weight=5,
                 ):
        self.run_name = run_name
        self.training_rectangles_fp = training_rectangles_fp
        self.training_polygons_fp = training_polygons_fp
        self.training_images_src = training_images_src
        self.preprocessed_base_dir = preprocessed_base_dir
        self.normalize_method = normalize_method
        self.resample_factor = resample_factor
        self.additional_input_bands = additional_input_bands
        self.rasterize_label_borders = rasterize_label_borders
        self.add_frames_json = add_frames_json
        self.enable_boundary_weights = enable_boundary_weights
        self.boundary_weights_scale = boundary_scale
        self.boundary_weights_weight = boundary_weight

        self.validate_config()

    def validate_config(self):
        # TODO
        return


class TrainConfig:
    def __init__(self,
                 run_name,
                 preprocessed_frames_dir,
                 saved_models_dir,
                 training_logs_dir,
                 normalize_method,
                 normalize_ratio,
                 normalize_bands,
                 patch_size,
                 train_batch_size,
                 loss_function,
                 optimizer_function,
                 num_epochs,
                 num_training_steps,
                 num_validation_steps,
                 test_split_ratio,
                 val_split_ratio,
                 use_all_frames_for_train=False,
                 area_based_frame_weighting=True,
                 resample_factor=1,
                 model_save_interval=None,
                 continue_model_fp=None,
                 tversky_alphabeta=None,
                 selected_GPU=0
                 ):
        self.run_name = run_name
        self.preprocessed_frames_dir = preprocessed_frames_dir
        self.saved_models_dir = saved_models_dir
        self.training_logs_dir = training_logs_dir
        self.normalize_method = normalize_method
        self.normalize_bands = list(normalize_bands)
        self.normalize_ratio = normalize_ratio
        self.patch_size = patch_size
        self.train_batch_size = train_batch_size
        self.loss_function = loss_function
        self.optimizer_function = optimizer_function
        self.num_epochs = num_epochs
        self.num_training_steps = num_training_steps
        self.num_validation_steps = num_validation_steps
        self.test_split_ratio = test_split_ratio
        self.val_split_ratio = val_split_ratio
        self.use_all_frames_for_train = use_all_frames_for_train
        self.area_based_frame_weighting = area_based_frame_weighting
        self.resample_factor = resample_factor
        self.model_save_interval = model_save_interval
        self.continue_model_fp = continue_model_fp
        self.tversky_alphabeta = tversky_alphabeta
        self.selected_GPU = selected_GPU

        # Set up tensorflow environment variables before importing tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide TF logs.  [Levels: 0->DEBUG, 1->INFO, 2->WARNING, 3->ERROR]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_GPU)

        self.validate_config()

    def validate_config(self):
        # TODO
        return


class ModelPrediction:

    def __init__(self,
                 name,
                 model_fp,
                 patch_size=None,
                 stride_percent=0,
                 input_bands_used=(1, 2, 3, 4),
                 # additional_bands=None,
                 normalize_method="stddev",
                 normalize_bands=(1, 2, 3, 4),
                 resample_factor=1,
                 prediction_threshold=0.5,
                 pred_batch_size=10,
                 overlay_merge_mode="MAX",
                 ):
        self.name = name
        self.model_fp = model_fp
        self.patch_size = self.get_patch_size(patch_size)
        self.stride_percent = stride_percent
        self.stride_pixels = round((1 - (stride_percent / 100)) * self.patch_size[0])
        self.input_bands_used = list(input_bands_used)
        # self.additional_bands = additional_bands
        self.normalize_method = normalize_method
        self.normalize_bands = list(normalize_bands)
        self.resample_factor = resample_factor
        self.prediction_threshold = prediction_threshold
        self.pred_batch_size = pred_batch_size
        self.overlay_merge_mode = overlay_merge_mode

        self.validate_config()

    def get_patch_size(self, input_patch_size):
        if input_patch_size is not None:
            return input_patch_size
        # If patch size is not passed, load it automatically from first layer shape in config of h5 model file
        with h5py.File(self.model_fp, 'r') as h5:
            return json.loads(h5.attrs["model_config"])["config"]["layers"][0]["config"]["batch_input_shape"][1:3]

    def validate_config(self):
        # TODO
        return


class PredictionConfig:
    """
    :param run_name:               Name for this prediction run, used in names of output files.
    :param prediction_images_src:  Images to predict: either a folder, list of files or database source.
    :param prediction_output_dir:  Output base directory where predictions will be saved.
    :param prediction_masks_src:   Mask files to mask source images: either a folder, list of files or database source.
    :param prediction_models:      List of model configurations to predict with, each a ModelPrediction class.
    :param additional_input_bands: List of AdditionalInputBand definitions to be appended to input images as extra bands
    :param output_dtype:           Either 'bool' (smallest size), 'uint8' (has nodata in 255), or 'float32' (raw preds)
    :param ensemble_merge_mode:    'MAX' or 'MIN' - how predictions of multiple models are merged
    :param ignore_nodata:          If true, nodata areas in the source images are not predicted
    :param prediction_gridsize:    Number of rows/cols in which images are split for parallel processing
    :param prediction_workers:     Number of parallel prediction threads (current impl loads model for each worker...)
    :param overwrite_existing:     overwrite existing predictions
    :param selected_GPU:           The tensorflow ID of the GPU to use for prediction: 0, 1, 2 etc. -1 selects CPU.
    """
    def __init__(self,
                 run_name,
                 prediction_images_src,
                 prediction_output_dir,
                 prediction_masks_src,
                 prediction_models,
                 additional_input_bands,
                 output_dtype="bool",
                 ensemble_merge_mode="MAX",
                 ignore_nodata=True,
                 prediction_gridsize=(1, 1),
                 prediction_workers=1,
                 overwrite_existing=False,
                 selected_GPU=0
                 ):
        self.run_name = run_name
        self.prediction_images_src = prediction_images_src
        self.prediction_output_dir = prediction_output_dir
        self.prediction_masks_src = prediction_masks_src
        self.prediction_models = prediction_models
        self.additional_input_bands = additional_input_bands
        self.output_dtype = output_dtype
        self.ensemble_merge_mode = ensemble_merge_mode
        self.ignore_nodata = ignore_nodata,
        self.prediction_gridsize = prediction_gridsize
        self.prediction_workers = prediction_workers
        self.overwrite_existing = overwrite_existing
        self.selected_GPU = selected_GPU

        # Set up tensorflow environment variables before importing tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide TF logs.  [Levels: 0->DEBUG, 1->INFO, 2->WARNING, 3->ERROR]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_GPU)

    def validate_config(self):
        # TODO
        return


class PostprocessingConfig:
    """
    :param run_name:                 Name for this postprocessing run, used in names of output files.
    :param postprocessing_dir:       Path to prediction folder to be postprocessed. If none, it will use most recent.
    :param create_polygons:          To polygonize the raster predictions to polygon VRT
    :param create_centroids:         To create centroids from polygons, with area in m2. [needs polygons first]
    :param create_density_maps:      To create tree density maps by crown area classes.  [needs centroids first]
    :param create_canopy_cover_maps: To create canopy cover maps from raster predictions
    :param postproc_workers:         number of CPU threads for parallel processing of polygons/centroids
    :param postproc_gridsize:        number of rows/cols in which images are split for parallel processing
    :param canopy_resolutions:       list of resolutions of canopy cover maps to create, in m
    :param density_resolutions:      list of resolutions of density maps to create, in m
    :param area_thresholds:          thresholds of area classes used for bands in density maps, in m2
    :param canopy_map_dtype:         uint8 or float32. uint8 is smaller, float32 is useful for smooth scatterplots
    :param no_vsimem:                Disable the use of temp files in virtual memory using GDAL's /vsimem/ paths
    """
    def __init__(self,
                 run_name,
                 postprocessing_dir,
                 create_polygons=True,
                 create_centroids=True,
                 create_density_maps=True,
                 create_canopy_cover_maps=True,
                 postproc_workers=64,
                 postproc_gridsize=(8,8),
                 canopy_resolutions=(100,),
                 density_resolutions=(100,),
                 area_thresholds=(3, 15, 50, 200),
                 canopy_map_dtype='float32',
                 no_vsimem=False,
                 ):
        self.run_name = run_name
        self.postprocessing_dir = postprocessing_dir
        self.create_polygons = create_polygons
        self.create_centroids = create_centroids
        self.create_density_maps = create_density_maps
        self.create_canopy_cover_maps = create_canopy_cover_maps
        self.postproc_workers = postproc_workers
        self.postproc_gridsize = postproc_gridsize
        self.canopy_resolutions = canopy_resolutions
        self.density_resolutions = density_resolutions
        self.area_thresholds = area_thresholds
        self.canopy_map_dtype = canopy_map_dtype
        self.no_vsimem = no_vsimem

    def validate_config(self):
        # TODO
        return