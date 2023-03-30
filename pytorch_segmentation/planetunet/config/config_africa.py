import os
import warnings
import numpy as np
from osgeo import gdal

from config.config_types import PreprocessConfig, ImageSource, AdditionalInputBand, TrainConfig, PostprocessingConfig, \
    PredictionConfig, ModelPrediction


class Configuration:
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py """
    def __init__(self):

        # ----- PREPROCESSING CONFIG ------
        self.preprocess_config = \
            PreprocessConfig(run_name="mgi19c_frap515_wc",
                             training_rectangles_fp="/nfs/Planet/backup_old/Africa/training_data/v19/v19_rectangles_forest.gpkg",
                             training_polygons_fp="forest_polys_5_10k_500_2_v19.gpkg",
                             training_images_src=ImageSource(folder="/nfs/Planet/backup_old/Africa/training_images/v2",
                                                             filetype=".jp2", bands=[1, 2, 3, 4]),
                             preprocessed_base_dir="/home/pingu/ign/data/planet/preprocessed_frames/",
                             normalize_method="nonorm",
                             resample_factor=1,
                             additional_input_bands=[AdditionalInputBand(name="worldcover",
                                                                         image_source=ImageSource(
                                                                             file="/nfs/Other_data/WorldCover_10m_2020/ESA_WorldCover_10m.vrt"),
                                                                         values_to_mask=[20, 30, 40, 50, 60, 70, 80, 90,
                                                                                         95, 100],
                                                                         scale_factor=0.1)],
                             enable_boundary_weights=False,
                             )

        # ----- TRAINING CONFIG ------
        self.train_config = \
            TrainConfig(run_name="test",
                        preprocessed_frames_dir=None,
                        saved_models_dir="/home/pingu/ign/data/planet/saved_models/",
                        training_logs_dir="/home/pingu/ign/data/planet/logs/",
                        normalize_method="stddev",
                        normalize_ratio=0.5,
                        normalize_bands=[1, 2, 3, 4],
                        patch_size=[512, 512],
                        train_batch_size=8,
                        loss_function="BCE",
                        optimizer_function="adaDelta",
                        num_epochs=1000,
                        num_training_steps=400,
                        num_validation_steps=100,
                        test_split_ratio=None,
                        val_split_ratio=None,
                        use_all_frames_for_train=True,
                        area_based_frame_weighting=True,
                        resample_factor=1,
                        model_save_interval=50,
                        continue_model_fp=None,
                        )

        # --- PREDICTION CONFIG ----
        self.predict_config = \
            PredictionConfig(run_name="test",
                             prediction_images_src=ImageSource(folder="/nfs/Planet/backup_old/Africa/training_images/v2",
                                                               filetype=".jp2", bands=[1, 2, 3, 4]),
                             prediction_output_dir="/home/pingu/ign/data/planet/output_predictions/",
                             prediction_masks_src="/nfs/Users/Florian/masks/prediction_masks",
                             prediction_models=[
                                 ModelPrediction(
                                     name="20230220-1930_mgi19c_stddev50_frap515_fi100uwp1_vghcw_BCE_s400_3m_cover5b10esaf_130",
                                     model_fp="/nfs/Users/Florian/models/20230220-1930_mgi19c_stddev50_frap515_fi100uwp1_vghcw_BCE_s400_3m_cover5b10esaf_130.h5",
                                     patch_size=[512, 512],
                                     stride_percent=50,
                                     input_bands_used=[1, 2, 3, 4, 5],
                                     normalize_method="std_dev_except_coverband",
                                     normalize_bands=[1, 2, 3, 4],
                                     resample_factor=1,
                                     prediction_threshold=0.5,
                                     pred_batch_size=3,
                                     overlay_merge_mode="MAX"),
                                 ModelPrediction(
                                     name="20221227-1303_mgi18_nonorm_nonforest_rnf25_pa_ecw_750epochs",
                                     model_fp="/nfs/Users/Florian/models/20221227-1303_mgi18_nonorm_nonforest_rnf25_pa_ecw_750epochs.h5",
                                     patch_size=[512, 512],
                                     stride_percent=50,
                                     input_bands_used=[1, 2, 3, 4],
                                     normalize_method="stddev",
                                     normalize_bands=[1, 2, 3, 4],
                                     resample_factor=3,
                                     prediction_threshold=0.5,
                                     pred_batch_size=3,
                                     overlay_merge_mode="MAX"),
                             ],
                             additional_input_bands=[AdditionalInputBand(name="worldcover",
                                                                         image_source=ImageSource(
                                                                             file="/nfs/Other_data/WorldCover_10m_2020/ESA_WorldCover_10m.vrt"),
                                                                         values_to_mask=[20, 30, 40, 50, 60, 70, 80, 90,
                                                                                         95, 100],
                                                                         scale_factor=0.1)],
                             output_dtype="bool",
                             ensemble_merge_mode="MAX",
                             ignore_nodata=True,
                             prediction_gridsize=(1,1),
                             prediction_workers=1,
                             overwrite_existing=False,
                             selected_GPU=0,
                             )
                             
        # --- POSTPROCESSING CONFIG ----
        self.postproc_config = \
            PostprocessingConfig(run_name="test",
                                 postprocessing_dir="/nfs/Users/Florian/predictions/africa_2019_v2",
                                 create_polygons=False,
                                 create_centroids=False,
                                 create_density_maps=False,
                                 create_canopy_cover_maps=True,
                                 postproc_workers=64,
                                 postproc_gridsize=(8, 8),
                                 canopy_resolutions=(100,),
                                 density_resolutions=(100,),
                                 area_thresholds=(3, 15, 50, 200),
                                 canopy_map_dtype='float32',
                                 no_vsimem=False
                                 )
                             

        # Set overall GDAL settings
        gdal.UseExceptions()                       # Enable exceptions, instead of failing silently
        gdal.SetCacheMax(32000000000)              # IO cache size in KB, used when warping/resampling. higher is better
        gdal.SetConfigOption('CPL_LOG', '/dev/null')
        warnings.filterwarnings('ignore')          # Disable warnings
