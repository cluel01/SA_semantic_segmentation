import os
import warnings
import numpy as np
from osgeo import gdal

from config.config_types import PostprocessingConfig


class Configuration:
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py """
    def __init__(self):

        # --- POSTPROCESSING CONFIG ----
        self.postproc_config = \
            PostprocessingConfig(run_name="test",
                                 postprocessing_dir="/home/jovyan/work/satellite_data/tmp/inference/unet_18_11_2022_103158_new.pth/2020",
                                 create_polygons=False,
                                 create_centroids=False,
                                 create_density_maps=False,
                                 create_canopy_cover_maps=True,
                                 postproc_workers=25,
                                 postproc_gridsize=(8, 8),
                                 canopy_resolutions=(100,),
                                 density_resolutions=(100,),
                                 area_thresholds=(3, 15, 50, 200),
                                 canopy_map_dtype='float32',
                                 no_vsimem=True
                                 )
                             

        # Set overall GDAL settings
        gdal.UseExceptions()                       # Enable exceptions, instead of failing silently
        gdal.SetCacheMax(32000000000)              # IO cache size in KB, used when warping/resampling. higher is better
        gdal.SetConfigOption('CPL_LOG', '/dev/null')
        warnings.filterwarnings('ignore')          # Disable warnings
