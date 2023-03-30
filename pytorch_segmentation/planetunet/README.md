# PlanetUNet
A framework for detecting trees in satellite images using deep learning with a UNet architecture.

### Contributors
```
Ankit Kariryaa - Original design and implementation of core framework for preprocessing, training and prediction
Sizhuo Li      - Conversion to multi-band planet images and implementation of boundary weights channel
Florian Reiner - New structure and config workflow, support for resampling and jp2 format, postprocessing
```

# Structure
The code is structured around four main steps of the pipeline: `preprocessing.py`, `training.py`, `prediction.py` and `postprocessing.py`.  
These steps are called from `main.py`, and rely on methods in `/core`.  
The overall configuration class is stored under `/config/`, and is initialised and passed to the pipeline in `main.py`.

During preprocessing, the training areas are extracted from the training images, and stored in temporary preprocessed frames. Each frame contains the image channels and the two annotation channels (labels and boundary weights).

During training, the UNet model is trained with the pre-processed frames and saved to file. Metadata about the model and it's config settings is stored in the .h5 file as an attribute.

During prediction, the trained model is used to predict trees in the prediction images. The output predictions are stored as compressed single-channel raster images.

During postprocessing, the output predictions are polygonised to polygon files. These are then converted to simplified centroid files, with the area of the polygon as an attribute.
For both these operations the prediction rasters are split into smaller grid chunks and processed in parallel. Vector VRT files are created to prevent creation and merging of very large geopackages, while still allowing viewing of large areas as one file.
Finally density and canopy cover rasters are produced at coarser resolutions, with one file for the entire area. The density maps show number of trees per ha, with different crown size classes as bands.


# Setup
PlanetUNet runs on python 3.8 and needs the libraries tensorflow, gdal>=3.2, rasterio>=1.2, geopandas>=0.9, matplotlib, scikit-learn, h5py, tqdm and imgaug.

A quick way to create the right environment is to use conda with the `example_env.yml` file from this repo and run:  
`conda env create --file example_env.yml`  
`conda activate tfgdal`  
`pip3 install tensorflow`

Tensorflow is installed via pip because conda doesn't yet have version 2.4+ which is required for CUDA11/RTX3090.

If using planet images with x3 resampling, the memory usage can be very high, and it is recommended to create a large swapfile, via:  
`sudo swapoff -a`  
`sudo dd if=/dev/zero of=/swapfile bs=1G count=150`  
`sudo chmod 600 /swapfile`  
`sudo mkswap /swapfile`  
`sudo swapon /swapfile`

# Usage
Before running `main.py`, the paths and general settings need to be configured in `config_default.py`.
You can make separate config files for separate datasets or use cases, which can be selected in the first import in `main.py`.

Running `main.py` will run the whole pipeline of preprocessing, training, prediction and postprocessing. By default, training will use the most recent preprocessing data it can find, prediction will use the latest trained model, and postprocessing the latest predictions, but these can also be set in the config.

To run individual steps, either convert `main.py` to a jupyter notebook, or simply comment out the other steps.

# Other functionality
### Upsampling
For Planet images, upsampling by a factor of 3 can give better results, which is set in `config.resample_factor`. This means that the training images are resampled during preprocessing, and prediction images are resampled before predicting.

### JPEG2000 images
To use JPEG2000 images, set `config.image_file_type=".jp2"`. This will allow preprocessing with jp2 training images, and prediction of jp2 images. Note that the preprocessed frames and the compressed output predictions will still use tif format.

### GPU selection
For multi-GPU environments, the GPU can be specified by its CUDA id in `config.seledted_gpu`

### Loading saved models
To continue training a previously trained model, specify its path in `config.continue_model_path`

### Custom prediction area
To predict only certain parts of the input prediction images, eg a specific study area, a validity shapefile mask can be set in `config.prediction_mask_fp`.
Only patches intersecting with shapes in this mask will be predicted.

### Chunked image processing for prediction and postprocessing
To improve performance during prediction and postprocessing, images can be split into smaller chunks internally, and optionally processed in parallel.
The number of chunks (rows and columns) that images are split into is set in `config.prediction_gridsize` and `config.postproc_gridsize`, 
and the number of worker processes to use for parallel processing is set in `config.prediction_workers` and `config.postproc_workers` respectively.

