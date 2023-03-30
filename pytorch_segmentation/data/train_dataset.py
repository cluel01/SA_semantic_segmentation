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
import pickle

from ..utils.preprocessing import pad_image_even
from ..utils.plotting import save_ground_truth_plots

class TrainDataset(Dataset):
    def __init__(self,dataset_path=None,data_file_path=None,shape_path=None,transform=None,n_channels=3,
                patch_size=[256,256],overlap=0,padding=False,pad_value=0,tree_size_threshold=0,file_extension=".tif",
                X=None,y=None,indices=None,mode="segmentation"):
        self.transform = transform
        if (dataset_path is None) or (not os.path.isfile(dataset_path)):
            if data_file_path is not None:
                self.patch_size = patch_size
                self.n_channels = n_channels
                self.overlap = overlap
                self.padding = padding
                self.pad_value = pad_value
                self.data_file_path = data_file_path
                self.shape_path = shape_path
                self.mode = mode
                self.tree_size_threshold = tree_size_threshold

                if (X is not None) and (y is not None):
                    assert len(X) == len(y)
                    self.X = np.array(X,dtype="uint8")#torch.as_tensor(X).float().contiguous()
                    self.y = np.array(y,dtype="uint8")#torch.as_tensor(y).long().contiguous()
                else:
                    shape_df = gdp.read_file(shape_path).geometry
                    raster_files = [os.path.join(data_file_path,i) for i in os.listdir(data_file_path) if i.endswith(file_extension)]
                    patches_satellite,patches_mask,file_mapping = self._create_patches(raster_files,shape_df,tree_size_threshold)
                    
                    del shape_df
                    # X = np.empty(tuple((len(patches_satellite),*patch_size[::-1])),dtype="uint8")
                    # for i in range(len(patches_satellite)):
                    #     X[i,:,:,:] = patches_satellite[i]
                    X = np.stack(patches_satellite).astype("uint8")
                    y = np.stack(patches_mask).astype("uint8")
                    assert len(X) == len(y)

                    # X = torch.from_numpy(X).byte().contiguous()
                    # self.y = torch.from_numpy(y).byte().contiguous()
                    self.X = X #/ 255
                    self.y = y
                    self.file_mapping = file_mapping
                if indices is None:
                    self.indices = np.arange(len(self.y))
                else:
                    self.indices = indices
            else:
                raise ValueError("Missing dataset_path or data_file_path!")
        else:
            self.load(dataset_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.X[idx] ).float()
        mask = torch.from_numpy(self.y[idx] ).long()

        image = image / 255

        if self.transform:
            sample = self.transform(image=image,target=mask)
            image,mask = sample
        
        if self.mode == "classification":
            count = torch.sum(mask)
            mask = 0
            if count > 0:
                mask = 1

        return {"x":image, "y":mask}

    def save(self,filename):
        if not os.path.isfile(filename): 
            cfg = self._get_config()
            obj = [self.X,self.y,self.indices,cfg]
            with open(filename, 'wb') as outp:  
                pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
            del obj
        else:
            print("INFO: File already exists, skip saving!")

    def load(self,filename):
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        self.X,self.y,self.indices,cfg  = obj
        for k in cfg.keys():
            setattr(self,k,cfg[k])

    def _get_config(self):
        return {"patch_size":self.patch_size,"n_channels":self.n_channels,"overlap":self.overlap,"padding":self.padding,"pad_value":self.pad_value,
            	"data_file_path":self.data_file_path,"shape_path":self.shape_path,"file_mapping":self.file_mapping}

    def get_img(self,idx,transform=True):
        image = torch.from_numpy(self.X[idx]).float()
        mask = torch.from_numpy(self.y[idx]).long()
        
        image = image / 255

        if (self.transform) and (transform):
            sample = self.transform(image,mask)
            image,_ = sample

        if image.size(0) > 3:
            image = image[:3,:,:]

        plt.imshow(image.permute(1, 2, 0).numpy()  )

    def get_mask(self,idx,transform=True):
        image = torch.from_numpy(self.X[idx]).float()
        mask = torch.from_numpy(self.y[idx]).long()

        image = image / 255

        if (self.transform) and (transform):
            sample = self.transform(image,mask)
            _,mask = sample
        plt.imshow(  mask.numpy()  )
    
    def show_tuple(self,idx,transform=True):
        image = torch.from_numpy(self.X[idx]).float()
        mask = torch.from_numpy(self.y[idx]).long()

        image = image / 255
        
        if (self.transform) and (transform):
            sample = self.transform(image,mask)
            image,mask = sample

        if image.size(0) > 3:
            image = image[:3,:,:]

        fig, axs = plt.subplots(1,2)
        axs[0].imshow(image.permute(1, 2, 0).numpy()  )
        axs[1].imshow(  mask.numpy())

    def get_train_test_set(self,test_size,train_transform=None,test_transform=None,seed=42):
        if train_transform is None:
            train_transform = self.transform
        
        indices_train,indices_test = train_test_split(self.indices, test_size=test_size, random_state=seed)
        train_set = TrainDataset(X=self.X[indices_train],y=self.y[indices_train],data_file_path=self.data_file_path,shape_path=self.shape_path,transform=train_transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_train,n_channels=self.n_channels)
        test_set = TrainDataset(X=self.X[indices_test],y=self.y[indices_test],data_file_path=self.data_file_path,shape_path=self.shape_path,transform=test_transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_test,n_channels=self.n_channels)
        
        return train_set,test_set

    def _create_patches(self,raster_files,shape_df,tree_size_threshold=0.2):
        patches_satellite = []
        patches_mask = []
        start_idx = 0
        mapping = {}
        for f in tqdm(raster_files):
            with rasterio.open(f) as src_sat:
                b_left,b_bottom,b_right,b_top = src_sat.bounds
                shape_df_filter = shape_df.cx[b_left:b_right,b_bottom:b_top]

                satellite_area_arr = src_sat.read()


                if self.padding: 
                    satellite_area_arr,_ = pad_image_even(satellite_area_arr,self.patch_size,self.overlap)

                if (satellite_area_arr.shape[1] < self.patch_size[0]) or (satellite_area_arr.shape[2] < self.patch_size[0]):
                    print(f"Shape {f} is too small for patch size with size: {satellite_area_arr.shape}")
                    continue

                patches = patchify(satellite_area_arr, 
                                        (self.n_channels, self.patch_size[0], self.patch_size[1]), 
                                        step=self.patch_size[0]-self.overlap)[0]

                reshaped_patches = np.reshape(patches, 
                                                (patches.shape[0]*patches.shape[1], 
                                                patches.shape[2], patches.shape[3], patches.shape[3]))
                                                # = (#patches, 3, 256, 256)
                patches_satellite.extend(reshaped_patches)

                if len(shape_df_filter) > 0:
                    if tree_size_threshold > 0:
                        #Create mask patches
                        big_trees= shape_df_filter[shape_df_filter.area*1000000000 > tree_size_threshold]
                        small_trees=shape_df_filter[shape_df_filter.area*1000000000 <= tree_size_threshold]

                        masks = []
                        if len(big_trees)>0:
                            tree_mask_big, _, _ = raster_geometry_mask(src_sat, big_trees, invert=True)
                            masks.append(tree_mask_big)
                        if len(small_trees) > 0:
                            tree_mask_small, _, _ = raster_geometry_mask(src_sat, small_trees, invert=True)
                            masks.append(tree_mask_small*2)
                        concat = np.stack(masks)
                        mask_arr = np.max(concat, axis=0)
                    else:
                        mask_arr, _, _ = raster_geometry_mask(src_sat, shape_df_filter, invert=True)
                else:
                    mask_arr = np.zeros(satellite_area_arr.shape[1:],dtype="uint8")
                pm = self._create_mask_patches(mask_arr)
                patches_mask.extend(pm)
                mapping[f] = [start_idx,start_idx+len(pm)]
                start_idx += len(pm)
        return patches_satellite,patches_mask,mapping

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

    
    def export_patches(self,save_dir,figsize=(10,5),alpha=0.6):
        for fname, idxs in self.file_mapping.items():
            i = torch.arange(idxs)
            imgs = torch.from_numpy(self.X[i])
            masks = torch.from_numpy(self.y[i])

            save_ground_truth_plots(imgs,masks,save_dir,i,fname,figsize=figsize,alpha=alpha)