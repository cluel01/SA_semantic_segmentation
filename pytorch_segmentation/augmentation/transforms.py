from configparser import Interpolation
import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size()[1:])
    if min_size < size:
        ow, oh = img.size()[1:]
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomApply:
    def __init__(self, transforms,p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            for t in self.transforms:
                image, target = t(image, target)
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target.unsqueeze(0), self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target.squeeze(0)


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        if target is not None:
            target = F.resize(target.unsqueeze(0), size, interpolation=T.InterpolationMode.NEAREST)
        return image, target.squeeze(0)


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if target is not None:
                target = F.vflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation  = saturation
        self.hue = hue
        self.transform = T.ColorJitter(brightness,contrast,saturation,hue)

    def __call__(self, image, target):
        extra_channels = image[3:,:,:]
        image = image[:3,:,:]
        image = self.transform(image)

        if extra_channels.size(0) > 0:
            image = torch.cat((image,extra_channels),0)

        return image, target

class RandomRotation:
    def __init__(self, degrees,fill=0):
        self.degrees = degrees
        self.fill = fill

    def __call__(self, image, target):
        angle = int(np.random.choice(range(self.degrees)))
        image = F.rotate(image,angle=angle,fill=self.fill)
        if target is not None:
            target = F.rotate(target.unsqueeze(0),angle=angle,fill=self.fill).squeeze()
        
        return image,target

class Pad:
    def __init__(self, padding,fill=0,padding_mode="reflect"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, target):
        image = F.pad(image,self.padding,self.fill,self.padding_mode)

        if target is not None:
            target = F.pad(target,self.padding,self.fill,self.padding_mode)
        
        return image,target


class RandomPixel:
    def __init__(self, pixel_value=0,ratio=0.01,random_apply=1.,seed=42):
        np.random.seed(seed)
        self.pixel_value = pixel_value
        self.ratio = ratio
        self.seed = seed
        self.random_apply = random_apply

    def __call__(self, image, target):

        if np.random.rand() < self.random_apply:
            if (len(image.shape) == 2):
                shape = image.shape 
                dim = 2
            else:
                shape = image.shape[1:]
                dim = 3
                if type(self.pixel_value) == int:
                    pixel_value = [self.pixel_value] * image.shape[0]
                else:
                    assert len(self.pixel_value) == image.shape[0]
                    pixel_value = self.pixel_value
            mask = np.random.rand(*shape) < self.ratio
            if dim == 2:
                image[mask] = self.pixel_value
            else:
                for i in range(dim):
                    image[i,mask] = pixel_value[i]
        return image, target

class RandomResizeCrop:
    def __init__(self, size,scale=(0.5,1),ratio=(0.75,1.333333),interpolation=F.InterpolationMode.BILINEAR):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, image, target):
        crop_params = T.RandomResizedCrop.get_params(image, scale=self.scale,ratio=self.ratio)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        image = F.resize(image,self.size,self.interpolation)
        if target is not None:
            target = F.resize(target.unsqueeze(0),self.size,self.interpolation)
        return image,target.squeeze(0)


class UnmaskEdges:
    def __init__(self, size,patch_size=[256,256]):
        self.height = (patch_size[0] - size[0]) // 2
        self.width = (patch_size[1] - size[1]) // 2

    def __call__(self, image, target):
        h,w = target.shape
        if target is not None:
            target_new = torch.zeros(target.shape,dtype=target.dtype)
            target_new[self.height:h-self.height,self.width:w-self.width] = target[self.height:h-self.height,self.width:w-self.width]
            target = target_new
        return image,target




class CLAHE_Norm:
    def __init__(self,clipLimit=1.0,tileGridSize=(4,4)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, image, target):
        extra_channels = image[3:,:,:]
        image = image[:3,:,:]

        image = (image * 255).byte()
        image = image.permute(2,1,0).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        image[:,:,0] = clahe.apply(image[:,:,0])

        # Converting image from LAB Color model to BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        
        image = torch.from_numpy(image.transpose(2,1,0)).float() / 255
        
        if extra_channels.size(0) > 0:
            image = torch.cat((image,extra_channels),0)

        return image, target

class Equalize:
    #Simple Histogram Equalization 
    def __init__(self):
        pass 

    def __call__(self, image, target):
        extra_channels = image[3:,:,:]
        image = image[:3,:,:]

        image = (image * 255).byte()
        image = image.permute(2,1,0).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image[:,:,0] = cv2.equalizeHist(image[:,:,0])

        # Converting image from LAB Color model to BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        
        image = torch.from_numpy(image.transpose(2,1,0)).float() / 255
        
        image = torch.cat((image,extra_channels),0)

        return image, target

class Add_ExG_Index:
    def __init__(self):
        pass 

    def __call__(self, image, target):
        i = image * 255
        new_channel = 2*i[1,:,:] - i[0,:,:] - i[2,:,:]
        # mean = new_channel.mean()
        # std = new_channel.std()
        # new_channel = (new_channel - mean) / std
        image = torch.cat((image,new_channel.unsqueeze(0)),0)

        return image, target

class Add_ExGR_Index:
    def __init__(self):
        pass 

    def __call__(self, image, target):
        i = image * 255
        new_channel = 3*i[1,:,:] - 2.4 * i[0,:,:] - i[2,:,:]
        # mean = new_channel.mean()
        # std = new_channel.std()
        # new_channel = (new_channel - mean) / std
        image = torch.cat((image,new_channel.unsqueeze(0)),0)
    

        return image, target

class Add_VDVI:
    def __init__(self,eps=1e-8):
        self.eps = eps

    def __call__(self, image, target):
        i = image * 255
        new_channel = 2*((2* i[1,:,:] -  i[0,:,:] - i[2,:,:]) / (2*i[1,:,:] + i[0,:,:] + i[2,:,:] + self.eps)) 
        # mean = new_channel.mean()
        # std = new_channel.std()
        # new_channel = (new_channel - mean) / std
        image = torch.cat((image,new_channel.unsqueeze(0)),0)
        

        return image, target



