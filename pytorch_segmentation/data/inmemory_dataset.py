import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
from patchify import patchify
from shapely import geometry
from sklearn.model_selection import train_test_split

from ..utils.preprocessing import pad_image_even

class InMemorySatDataset(Dataset):
    def __init__(self,data_file_path=None,mask_path=None,X=None,y=None,transform=None,patch_size=[256,256,3],overlap=0,padding=False,pad_value=0,file_extension=".tif",indices=None):
        self.transform = transform
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding = padding
        self.pad_value = pad_value
        self.data_file_path = data_file_path
        self.mask_path = mask_path

        if (X is not None) and (y is not None):
            self.X = torch.as_tensor(X).float().contiguous()
            self.y = torch.as_tensor(y).long().contiguous()
        else:
            if (data_file_path is not None) and (mask_path is not None):
                mask_areas = [rasterio.open(os.path.join(mask_path,i)) for i in os.listdir(mask_path) if i.endswith(file_extension)]
                satellite_img = rasterio.open(data_file_path)


                patches_masks = self._create_mask_patches(mask_areas)
                patches_data = self._create_data_patches(satellite_img,mask_areas)

                assert len(patches_masks) == len(patches_data)

                X = np.array(patches_data)/255
                y = np.array(patches_masks)

                self.X = torch.as_tensor(X).float().contiguous()
                self.y = torch.as_tensor(y).long().contiguous()
                
                del X
                del y
                for i in mask_areas:
                    i.close()
                satellite_img.close()
    
            else:
                raise Exception("Wrong input given!")
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
        plt.imshow(  mask.numpy())

    def show_tuple(self,idx):
        image = self.X[idx]
        mask = self.y[idx]
        if self.transform:
            sample = self.transform(image,mask)
            image,mask = sample
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(image.permute(1, 2, 0).numpy()  )
        axs[1].imshow(  mask.numpy())
        #plt.show()


    def get_train_test_set(self,test_size,train_transform=None,test_transform=None,seed=42):
        if train_transform is None:
            train_transform = self.transform
        
        X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(self.X.numpy(), self.y.numpy(),self.indices, test_size=test_size, random_state=seed)
        train_set = InMemorySatDataset(X=X_train,y=y_train,data_file_path=self.data_file_path,mask_path=self.mask_path,transform=self.transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_train)
        test_set = InMemorySatDataset(X=X_test,y=y_test,data_file_path=self.data_file_path,mask_path=self.mask_path,transform=test_transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_test)

        return train_set,test_set

    def get_config(self):
        return {"patch_size":self.patch_size,"overlap":self.overlap,"padding":self.padding,"pad_value":self.pad_value,"size":len(self)}

    def _create_mask_patches(self,mask_areas):
        patches_masks = []
        for ma in mask_areas:
            label_area_arr = ma.read(1)#, #window = from_bounds(*bounds, la.transform))
            if self.padding:
                label_area_arr,_  = pad_image_even(label_area_arr,self.patch_size,self.overlap,dim=2,border_val=self.pad_value)
            patches = patchify(label_area_arr,(self.patch_size[0], self.patch_size[1]), 
                                    step=self.patch_size[0]-self.overlap)
            reshaped_patches = np.reshape(patches, 
                                        (patches.shape[0]*patches.shape[1], 
                                        patches.shape[2], patches.shape[3])) 
                                        # = (#patches, 256, 256)
            patches_masks.extend(reshaped_patches)
        return patches_masks
        

    def _create_data_patches(self,satellite_img,mask_areas):
        patches_satellite = []
        for ma in mask_areas:
            # get coordinates 
            geom = geometry.box(*ma.bounds)
            satellite_area_arr, sat_patch_arr_transform = rasterio.mask.mask(satellite_img,[geom]
                                                                        , crop=True)
            #bounds =  la.bounds
            #satellite_area_arr = satellite_img.read(None, window = from_bounds(*bounds, satellite_img.transform))
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
        return patches_satellite







## SatDataset for containing satellite imagery areas based on given masks + label masks -> returning X,y
# class SatDataset(Dataset):
#     def __init__(self,data_file_path,shape_path=None,mask_path=None,transform=None,patch_size=[256,256,3],overlap=0,padding=False,pad_value=0,file_extension=".tif"):
#         self.transform = transform
#         self.patch_size = patch_size
#         self.overlap = overlap
#         self.padding = padding
#         self.pad_value = pad_value
#         self.data_file_path = data_file_path
#         self.shape_path = shape_path

#         


