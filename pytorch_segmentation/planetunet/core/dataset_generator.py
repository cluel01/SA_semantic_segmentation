#    Author: Ankit Kariryaa, University of Bremen
import rasterio
from imgaug import augmenters as iaa
import numpy as np
from shapely.geometry import box


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        sometimes(iaa.CropAndPad(percent=(0, 0.1))),  # random crops

        #
        # # Gaussian blur and gamma contrast
        # sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
        # sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),

        # iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)
        # sometimes(iaa.Multiply((0.75, 1.25), per_channel=True), 0.3),
        sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        # iaa.Add(value=(-0.5,0.5),per_channel=True),
        sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    return seq


class DataGenerator:
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """

    def __init__(self, input_image_channel, patch_size, frame_list, frames, annotation_channel, augmenter=None,
                 boundary_weight=10):
        """Datagenerator constructor

        Args:
            input_image_channel (list(int)): Describes which channels is the image are input channels.
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class.
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined
            in imageAugmentationWithIAA function.
        """
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.augmenter = augmenter
        self.boundary_weight = boundary_weight
        self.frame_bounds = [box(*rasterio.open(frames[i_f].fp).bounds) for i_f in frame_list]

        # Calculate area-based frame weights (to sample big frames more often, for even overall sampling coverage)
        areas_px = np.array([np.prod(f.img.shape[:2]) for f in frames])
        area_weights = areas_px/areas_px.sum()
        self.area_weights = area_weights

        # Calculate tree cover based weights (to sample areas with scattered trees more often than dense forest)
        tree_areas_px = np.array([f.annotations.sum() for f in frames])
        tree_areas_proportion = tree_areas_px/areas_px
        # tree_areas_proportion[tree_areas_proportion == 0] = 1   # weight empty rectangles same as full forest rectangles
        # tree_areas_weights = 1 + 2*(1 - tree_areas_proportion)**20 + 0.2*(1 - tree_areas_proportion)

        # Combine weights and normalise for total sum = 1
        # combined_weights = (area_weights * tree_areas_weights) / (area_weights*tree_areas_weights).sum()
        # self.frame_list_weights = combined_weights[frame_list]
        self.frame_list_weights = area_weights
        self.frame_list_covers = tree_areas_proportion
        self.frame_importance = {fid: [1] for fid in frame_list}

        # import pandas as pd
        # df = pd.DataFrame(dict(areas_px=areas_px,
        #                        area_weights=area_weights,
        #                        tree_areas_px=tree_areas_px,
        #                        tree_areas_proportion=tree_areas_proportion,
        #                        tree_areas_weights=tree_areas_weights,
        #                        combined_weights=combined_weights,
        #                        frame_list_weights=self.frame_list_weights))
        #df.to_csv("frame_weights.csv")

    # Return all training and label images and weights, generated sequentially with the given step size
    def all_sequential_patches(self, step_size, normalize = 1):
        """Generate all patches from all assigned frames sequentially.

            step_size (tuple(int,int)): Size of the step when generating frames.
            normalize (float): Probability with which a frame is normalized.
        """
        patches = []
        for fn in self.frame_list:
            frame = self.frames[fn]
            ps = frame.sequential_patches(self.patch_size, step_size, normalize)
            patches.extend(ps)
        data = np.array(patches)
        img = data[..., self.input_image_channel]
        y = data[..., self.annotation_channel]
        # y would have two channels, i.e. annotations and weights.
        ann = y[...,[0]]
        # boundaries have a weight of 10 other parts of the image has weight 1
        weights = y[...,[1]]
        weights[weights>=0.5] = self.boundary_weight
        weights[weights<0.5] = 1
        ann_joint = np.concatenate((ann,weights), axis=-1)
        return (img, ann_joint)

    # Return a batch of training and label images, generated randomly
    # def random_patch(self, BATCH_SIZE, normalize):
    #     """Generate patches from random location in randomly chosen frames.
    #
    #     Args:
    #         BATCH_SIZE (int): Number of patches to generate (sampled independently).
    #         normalize (float): Probability with which a frame is normalized.
    #     """
    #     patches = []
    #     for i in range(BATCH_SIZE):
    #         fn = np.random.choice(self.frame_list, p=self.frame_list_weights)
    #         frame = self.frames[fn]
    #         patch = frame.random_patch(self.patch_size, normalize)
    #         patches.append(patch)
    #     data = np.array(patches)
    #
    #     img = data[..., self.input_image_channel]
    #     ann_joint = data[..., self.annotation_channel]
    #     return (img, ann_joint)
    def random_patch(self, BATCH_SIZE, normalize):
        """Generate patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently).
            normalize (float): Probability with which a frame is normalized.
        """
        patches = []
        fids = []
        for i in range(BATCH_SIZE):
            if np.random.rand() < 0.99:
                fn = np.random.choice(self.frame_list, p=self.frame_list_weights)
            else:
                fn = np.random.choice(self.frame_list)
            frame = self.frames[fn]
            fids.append(fn)
            patch = frame.random_patch(self.patch_size, normalize)
            patches.append(patch)
        data = np.array(patches)

        img = data[..., self.input_image_channel]
        ann_joint = data[..., self.annotation_channel]
        return (img, ann_joint, fids)
#     print("Wrote {} random patches to {} with patch size {}".format(count,write_dir,patch_size))

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def random_generator(self, BATCH_SIZE, normalize = 1):
        """Generator for random patches, yields random patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled independently).
            normalize (float): Probability with which a frame is normalized.
        """
        seq = imageAugmentationWithIAA()

        while True:
            # X, y = self.random_patch(BATCH_SIZE, normalize)
            X, y, frame_ids = self.random_patch(BATCH_SIZE, normalize)
            if self.augmenter == 'iaa':
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
                # y would have two channels, i.e. annotations and weights. We need to augment y for operations such as crop and transform
                y = seq_det.augment_images(y)
                # Some augmentations can change the value of y, so we re-assign values just to be sure.
                ann =  y[...,[0]]
                ann[ann<0.5] = 0
                ann[ann>=0.5] = 1
                # boundaries have a weight of 10 other parts of the image has weight 1
                weights = y[...,[1]]
                weights[weights>=0.5] = self.boundary_weight
                weights[weights<0.5] = 1
                # cover = y[..., [2]]
                #
                # ann_joint = np.concatenate((ann,weights, cover), axis=-1)
                fid = np.ones(ann.shape[:-1], dtype=np.float32)
                for i in range(X.shape[0]):
                    fid[i, ...] *= frame_ids[i]
                fid = np.expand_dims(fid, axis=-1)
                ann_joint = np.concatenate((ann,weights,fid), axis=-1)
                yield X, ann_joint
            else:
                # y would have two channels, i.e. annotations and weights.
                ann =  y[...,[0]]
                # boundaries have a weight of 10 other parts of the image has weight 1
                weights = y[...,[1]]
                weights[weights>=0.5] = self.boundary_weight
                weights[weights<0.5] = 1
                # cover = y[..., [2]]
                #
                # ann_joint = np.concatenate((ann,weights, cover), axis=-1)
                fid = np.ones(ann.shape[:-1], dtype=np.float32)
                for i in range(X.shape[0]):
                    fid[i, ...] *= frame_ids[i]
                fid = np.expand_dims(fid, axis=-1)
                ann_joint = np.concatenate((ann,weights,fid), axis=-1)
                yield X, ann_joint
