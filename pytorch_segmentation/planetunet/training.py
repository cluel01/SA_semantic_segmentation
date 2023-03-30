import os
import json
import time
import shutil
from datetime import datetime, timedelta
import geopandas as gpd
import h5py
import rasterio
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from core.UNet import UNet
from core.frame_info import FrameInfo
from core.optimizers import get_optimizer
from core.split_frames import split_dataset
from core.dataset_generator import DataGenerator as Generator
from core.losses import accuracy, dice_coef, dice_loss, specificity, sensitivity, get_loss
from core.util import Partial


class GeneratorImportanceCallback(Callback):
    def __init__(self, TGen, model_path="", smoothing=100, exp_scale=1, init_imp=1):
        super().__init__()
        self.TGen = TGen
        self.smoothing = smoothing
        self.exp_scale = exp_scale
        self.df = gpd.GeoDataFrame(
            {"geometry": self.TGen.frame_bounds, "frame_id": self.TGen.frame_list, "cover": self.TGen.frame_list_covers,
             "-1": init_imp})
        self.weights_out_fp = os.path.basename(model_path) + "_frame_importance.gpkg"

    def on_epoch_end(self, epoch, logs=None):
        im = np.array([np.nanmean(self.TGen.frame_importance[f][-self.smoothing:]) for f in self.TGen.frame_list])
        im = im**self.exp_scale
        combined_weights = (im * self.TGen.area_weights) / np.sum(im * self.TGen.area_weights)
        self.TGen.frame_list_weights = combined_weights

        # Write frame importance to file
        self.df[str(epoch)] = im
        self.df.to_file(self.weights_out_fp, "GPKG", append=True)


# class LossWithImportanceMapping():
#     """Wrapper to allow for custom loss functions"""
#
#     def __init__(self, loss_fn, frame_ids):
#         self.__name__ = "LossWithImportanceMapping"
#         self.loss_fn = loss_fn
#         self.frame_importance = {fid: [0.5] for fid in frame_ids}
#
#     def __call__(self, y_true, y_pred):
#         fids = y_true[..., -1]
#         for i in range(y_true.shape[0]):
#             ls = self.loss_fn(y_true[[i]], y_pred[[i]])
#             # get non-zero id value from fid
#             #            fid = tf.math.reduce_max(tf.boolean_mask(fid[i], fid[i] > 0))
#             #            self.frame_importance[f_id].append(tf.math.reduce_mean(ls))
#             fid = int(np.max(fids[i]))
#             self.frame_importance[fid].append(np.mean(ls))
#         # It is not ideal to calculate the loss twice but it's the safer way to do it
#         return self.loss_fn(y_true, y_pred)


def loss_importance_mapping(y_true, y_pred, loss_fn, frame_importance):
    # y_true shape:  [batch_size, patch_width, patch_height, 3]  -> true, weights, fid
    for i in range(y_true.shape[0]):
        ls = loss_fn(y_true[[i]], y_pred[[i]])
        fid = int(np.max(y_true[i, ..., -1]))   # FID is in last channel of y_true, use max because of augmentations
        frame_importance[fid].append(np.mean(ls))
    return loss_fn(y_true, y_pred)


def get_all_frames():
    """Get all pre-processed frames which will be used for training."""

    # If no specific preprocessed folder was specified, use the most recent preprocessed data
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(config.preprocessed_base_dir,
                                               sorted(os.listdir(config.preprocessed_base_dir))[-1])

    # Get paths of preprocessed images
    image_paths = [os.path.join(config.preprocessed_dir, fn) for fn in os.listdir(config.preprocessed_dir) if
                   fn.endswith(".tif")]
    print(f"Found {len(image_paths)} input frames in {config.preprocessed_dir}")

    # Build a frame for each input image
    frames = []
    for idx, im_path in tqdm(enumerate(image_paths), desc="Processing frames", total=len(image_paths)):

        # Open preprocessed image
        preprocessed = rasterio.open(im_path).read()

        # Get image channels   (last two channels are labels + weights)
        image_channels = preprocessed[:-2, ::]

        # Transpose to have channels at the end
        image_channels = np.transpose(image_channels, axes=[1, 2, 0])

        # Get annotation and weight channels
        annotations = preprocessed[-2, ::]
        weights = preprocessed[-1, ::]

        # Create frame with combined image, annotation, and weight bands
        frames.append(FrameInfo(image_channels, annotations, weights, config.normalize_method, np.float32, im_path, frame_id=idx))

    return frames


def create_train_val_datasets(frames):
    """ Create the training, validation and test datasets """

    # If override set, ignore split and use all frames for everything
    if config.override_use_all_frames:
        training_frames = validation_frames = test_frames = list(range(len(frames)))

    else:
        frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
        training_frames, validation_frames, test_frames = split_dataset(frames, frames_json, config.test_ratio,
                                                                        config.val_ratio)

    # Define input and annotation channels
    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)     # because label and weights are directly after the input channels
    weight_channel = len(config.channel_list) + 1
    #annotation_channels = [label_channel, weight_channel]
    # cover_channel = len(config.channel_list) + 2
    # annotation_channels = [label_channel, weight_channel, cover_channel]
    #
    # # Define model patch size: Height * Width * (Input + Output) channels
    # patch_size = [*config.patch_size, len(config.channel_list) + len(annotation_channels)]
    #
    # # Create generators for training, validation and test data
    # train_generator = Generator(input_channels, patch_size, training_frames, frames, annotation_channels,
    #                             augmenter='iaa', boundary_weight=config.boundary_weight).random_generator(
    #                             config.train_batch_size, config.normalise_ratio)
    annotation_channels = [label_channel, weight_channel]

    cover_channel = len(config.channel_list) + 2
    # annotation_channels = [label_channel, weight_channel, cover_channel]

    # Define model patch size: Height * Width * (Input + Output) channels
    patch_size = [*config.patch_size, len(config.channel_list) + len(annotation_channels)]

    # Create generators for training, validation and test data
    TGen = Generator(input_channels, patch_size, training_frames, frames, annotation_channels,
                     augmenter='iaa', boundary_weight=config.boundary_weight)
    train_generator = TGen.random_generator(config.train_batch_size, config.normalise_ratio)
    val_generator = Generator(input_channels, patch_size, validation_frames, frames, annotation_channels,
                              augmenter=None, boundary_weight=config.boundary_weight).random_generator(
                              config.train_batch_size, config.normalise_ratio)
    test_generator = Generator(input_channels, patch_size, test_frames, frames, annotation_channels,
                               augmenter=None, boundary_weight=config.boundary_weight).random_generator(
                               config.train_batch_size, config.normalise_ratio)

    # return train_generator, val_generator, test_generator
    return TGen, train_generator, val_generator, test_generator


def create_callbacks(model_path):
    """ Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing"""

    # Add checkpoint callback to save model during training.
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=False)

    # Add tensorboard callback to follow training progress
    log_dir = os.path.join(config.logs_dir, os.path.basename(model_path)[:-3])
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, embeddings_freq=0,
                              write_images=False, embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch', profile_batch='500,520')

    # Add a callback to store custom metadata in the model .h5 file
    # This allows us to later remember settings that were used when this model was trained (not only those in filename)
    # The metadata is saved at the end of every epoch to preserve info when training is ended early
    class CustomMeta(Callback):
        def __init__(self):
            super().__init__()
            self.start_time = datetime.now()

        def on_epoch_end(self, epoch, logs=None):
            # Create object with all custom metadata
            meta_data = {
                "name": config.model_name,
                "model_path": model_path,
                "patch_size": config.patch_size,
                "channels_used": config.channels_used,
                "resample_factor": config.resample_factor,
                "frames_dir": config.preprocessed_dir,
                "train_ratio": float(f"{1-config.val_ratio-config.test_ratio:.2f}"),
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "normalise_ratio": config.normalise_ratio,
                "loss": config.loss_fn,
                "optimizer": config.optimizer_fn,
                "tversky_alpha": config.tversky_alphabeta[0],
                "tversky_beta": config.tversky_alphabeta[1],
                "batch_size": config.train_batch_size,
                "epoch_steps": config.num_training_steps,
                "val_steps": config.num_validation_images,
                "epochs_trained": f"{epoch + 1}/{config.num_epochs}",
                "last_sensitivity": float(f"{logs['sensitivity']:.4f}"),        # could also add other metrics if needed
                "start_time": self.start_time.strftime("%d.%m.%Y %H:%M:%S"),
                "elapsed_time": (datetime.utcfromtimestamp(0) + (datetime.now() - self.start_time)).strftime("%H:%M:%S")
            }
            # Serialise to json string and inject into the .h5 model file as an attribute
            with h5py.File(model_path, "a") as file:
                file.attrs["custom_meta"] = bytes(json.dumps(meta_data), "utf-8")
            # Optionally save the model at regular intervals
            if config.model_save_interval and (epoch + 1) % config.model_save_interval == 0:
                shutil.copy(model_path, model_path.replace(".h5", f"_{epoch+1}epochs.h5"))

    return [checkpoint, tensorboard, CustomMeta()]


# @tf.function
# def train_step(images, labels, gradients = None, cs = 0, bs = 32):
#
#   with tf.GradientTape() as tape:
#     # training=True is only needed if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     predictions = model(images, training=True)
#     loss = loss_object(labels, predictions)
#
#   if gradients is None:
#       gradients = tape.gradient(loss, model.trainable_variables)
#   else:
#       gradients +=tape.gradient(loss, model.trainable_variables)
#
#   if cs == bs:
#       optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#       gradients = None
#       cs = 0
#   train_loss(loss)
#   train_accuracy(labels, predictions)
#   return  cs, gradients


def train_model(conf):
    """Create and train a new model"""
    global config
    config = conf

    print("Starting training.")
    start = time.time()

    # Get all training frames
    frames = get_all_frames()

    # Split into training, validation and test datasets (proportions are set in config)
    # train_generator, val_generator, test_generator = create_train_val_datasets(frames)
    TGen, train_generator, val_generator, test_generator = create_train_val_datasets(frames)

    # Create model name from timestamp and custom name
    model_path = os.path.join(config.saved_models_dir, f"{time.strftime('%Y%m%d-%H%M')}_{config.model_name}.h5")
    starting_epoch = 0

    # loss = LossWithImportanceMapping(get_loss(config.loss_fn, config.tversky_alphabeta), TGen.frame_list)
    loss = Partial(loss_importance_mapping, loss_fn=get_loss(config.loss_fn, config.tversky_alphabeta),
                   frame_importance=TGen.frame_importance)

    # Check if we want to continue training an existing model
    if config.continue_model_path is not None:

        # Load previous model
        print(f"Loading pre-trained model from {config.continue_model_path} :")
        model = tf.keras.models.load_model(config.continue_model_path,
                                           custom_objects={'tversky': loss,
                                                           'dice_coef': dice_coef, 'dice_loss': dice_loss,
                                                           'accuracy': accuracy, 'specificity': specificity,
                                                           'sensitivity': sensitivity}, compile=False)

        # Get starting epoch from metadata
        with h5py.File(config.continue_model_path, 'r') as model_file:
            if "custom_meta" in model_file.attrs:
                custom_meta = json.loads(model_file.attrs["custom_meta"])#.decode("utf-8"))
                starting_epoch = int(custom_meta["epochs_trained"].split("/")[0])

        # Copy logs from previous training so that tensorboard shows combined epochs
        old_log_dir = os.path.join(config.logs_dir, os.path.basename(config.continue_model_path)[:-3])
        new_log_dir = os.path.join(config.logs_dir, os.path.basename(model_path)[:-3])
        if os.path.exists(old_log_dir):
            shutil.copytree(old_log_dir, new_log_dir)

    # Otherwise define new model
    else:
        model = UNet([config.train_batch_size, *config.patch_size, len(config.channel_list)], [len(config.channel_list)])

    # Create callbacks to be used during training
    # callbacks = create_callbacks(model_path)
    callbacks = create_callbacks(model_path) + [GeneratorImportanceCallback(TGen, model_path)]

    # Train the model
    tf.config.run_functions_eagerly(True)
    model.compile(optimizer=get_optimizer(config.optimizer_fn), loss=loss,
                  metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])
    #model.summary()
    model.fit(train_generator,
              steps_per_epoch=config.num_training_steps,
              epochs=config.num_epochs,
              initial_epoch=starting_epoch,
              validation_data=val_generator,
              validation_steps=config.num_validation_images,
              callbacks=callbacks,
              workers=1)

    print(f"Training completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
