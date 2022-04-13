from rasterio.mask import raster_geometry_mask
import geopandas as gdp
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
from patchify import patchify
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..utils.preprocessing import pad_image_even

class RwandaDataset(Dataset):
    def __init__(self,data_file_path=None,shape_path=None,transform=None,patch_size=[256,256,3],overlap=0,padding=False,pad_value=0,file_extension=".tif",X=None,y=None,indices=None):
        self.transform = transform
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding = padding
        self.pad_value = pad_value
        self.data_file_path = data_file_path
        self.shape_path = shape_path

        if (X is not None) and (y is not None):
            assert len(X) == len(y)
            self.X = torch.as_tensor(X).float().contiguous()
            self.y = torch.as_tensor(y).long().contiguous()
        else:
            shape_df = gdp.read_file(shape_path).geometry
            raster_files = [os.path.join(data_file_path,i) for i in os.listdir(data_file_path) if i.endswith(file_extension)]
            patches_satellite,patches_mask = self._create_patches(raster_files,shape_df)
            
            del shape_df

            X = np.array(patches_satellite)/255
            y = np.array(patches_mask)

            assert len(X) == len(y)

            self.X = torch.as_tensor(X).float().contiguous()
            self.y = torch.as_tensor(y).long().contiguous()

        if indices is None:
            self.indices = np.arange(len(self.y))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        mask = self.y[idx]

            
        if self.transform:
            sample = self.transform(image=image,target=mask)
            image,mask = sample

        return {"x":image, "y":mask}

    def get_img(self,idx):
        image = self.X[idx]
        mask = self.y[idx]
        if self.transform:
            sample = self.transform(image,mask)
            image,_ = sample
        plt.imshow(image.permute(1, 2, 0).numpy()  )

    def get_mask(self,idx):
        image = self.X[idx]
        mask = self.y[idx]
        if self.transform:
            sample = self.transform(image,mask)
            _,mask = sample
        plt.imshow(  mask.numpy()  )

    def get_train_test_set(self,test_size,train_transform=None,test_transform=None,seed=42):
        if train_transform is None:
            train_transform = self.transform
        
        X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(self.X.numpy(), self.y.numpy(),self.indices, test_size=test_size, random_state=seed)
        train_set = RwandaDataset(X=X_train,y=y_train,data_file_path=self.data_file_path,shape_path=self.shape_path,transform=self.transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_train)
        test_set = RwandaDataset(X=X_test,y=y_test,data_file_path=self.data_file_path,shape_path=self.shape_path,transform=test_transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_test)
        
        return train_set,test_set

    def _create_patches(self,raster_files,shape_df):
        patches_satellite = []
        patches_mask = []
        for f in tqdm(raster_files):
            with rasterio.open(f) as src_sat:
                b_left,b_bottom,b_right,b_top = src_sat.bounds
                shape_df_filter = shape_df.cx[b_left:b_right,b_bottom:b_top]
                if len(shape_df_filter) == 0:
                     continue

                satellite_area_arr = src_sat.read()
                
                if self.padding: 
                    satellite_area_arr,_ = pad_image_even(satellite_area_arr,self.patch_size,self.overlap)
                patches = patchify(satellite_area_arr, 
                                        (self.patch_size[2], self.patch_size[0], self.patch_size[1]), 
                                        step=self.patch_size[0]-self.overlap)[0]
                reshaped_patches = np.reshape(patches, 
                                                (patches.shape[0]*patches.shape[1], 
                                                patches.shape[2], patches.shape[3], patches.shape[3]))
                                                # = (#patches, 3, 256, 256)
                patches_satellite.extend(reshaped_patches)

                mask_arr,_,_ = raster_geometry_mask(src_sat,shape_df_filter,invert=True)
                pm = self._create_mask_patches(mask_arr)
                patches_mask.extend(pm)
        return patches_satellite,patches_mask

    def _create_mask_patches(self,mask_arr):
        patches_masks = []

        if self.padding:
            mask_arr,_  = pad_image_even(mask_arr,self.patch_size,self.overlap,dim=2,border_val=self.pad_value)
        patches = patchify(mask_arr,(self.patch_size[0], self.patch_size[1]), 
                                step=self.patch_size[0]-self.overlap)
        reshaped_patches = np.reshape(patches, 
                                    (patches.shape[0]*patches.shape[1], 
                                    patches.shape[2], patches.shape[3])) 
                                    # = (#patches, 256, 256)
        patches_masks.extend(reshaped_patches)
        return patches_masks