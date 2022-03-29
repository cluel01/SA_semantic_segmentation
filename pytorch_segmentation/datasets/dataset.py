import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds, Window
import os
import numpy as np
from patchify import patchify
from shapely import geometry
from sklearn.model_selection import train_test_split
import fiona
import cv2
import pickle
import pandas as pd

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
        plt.imshow(  mask.numpy()  )

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



## SatInferenceDataset for containing satellite imagery areas based on given shapes -> only returning X without mask
class SatInferenceDataset(Dataset):
    def __init__(self,data_file_path,shape_path=None,transform=None,patch_size=[256,256,3],overlap=128,padding=64,pad_value=0,file_extension=".shp"):
        self.transform = transform
        if type(patch_size) == list:
            self.patch_size = np.array(patch_size)
        else: #numpy array
            self.patch_size = patch_size
        self.t_patch_size = self.patch_size[[2,0,1]]  # patch_size for Tensors 
        self.overlap = overlap
        self.padding = padding
        self.pad_value = pad_value
        self.data_file_path = data_file_path
        self.shape_path = shape_path

        patches = []
        shapes = []
        shape_idx = 0
        shape_files = [os.path.join(shape_path,i) for i in os.listdir(shape_path) if i.endswith(file_extension)]
        with rasterio.open(data_file_path) as src_sat:
            sat_shape = geometry.box(*src_sat.bounds)
            for f in shape_files:
                with fiona.open(f) as src_shape:
                    for i,shp in enumerate(src_shape):
                        name = os.path.basename(f).split(".")[0] + "_" + str(i+1)
                        s = geometry.shape(shp["geometry"])
                        if not sat_shape.intersects(s):
                            print(f"Shape {name} does not intersect!")
                            continue
                        win = from_bounds(*s.bounds,src_sat.transform)
                        pad = self._get_padding(win,patch_size,overlap,padding)
                        win_list,grid_shape = self._patchify_window(win,src_sat,patch_size,overlap,padding)
                        shapes.append({"shape_id":shape_idx,"transform":src_sat.window_transform(win),"padding":pad,
                                        "start_idx":len(patches),"grid_shape":grid_shape,"name":name,
                                        "width":win.width,"height":win.height})
                        patches.extend(win_list)
                        shape_idx += 1
            self.sat_meta = src_sat.meta.copy()
        self.patches = pd.DataFrame(patches)
        self.shapes = pd.DataFrame(shapes)
        

    def __getitem__(self, index):
        patch = self.patches.iloc[index]
        with rasterio.open(self.data_file_path) as src_sat:
            win = patch["window"]
            img = src_sat.read(window=win)
            if img.size == 0:
                img = torch.empty(self.t_patch_size)
                img = img.fill_(self.pad_value)
            elif tuple(img.shape) != tuple(self.t_patch_size):
                top = 0
                left = 0
                if win.row_off < 0:
                    top = (self.patch_size[0]- img.shape[1]) // 2
                if win.col_off < 0:
                    left = (self.patch_size[0]- img.shape[2]) // 2
                bottom = self.patch_size[0]-img.shape[1]+top
                right = self.patch_size[0]-img.shape[2]+left
                img = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_REPLICATE,value=self.pad_value)
                img = img.transpose(2,0,1)
        img = img / 255
        img = torch.as_tensor(img).float().contiguous() 
        if self.transform:
            img = self.transform(img)
        return {"img":img,"idx":torch.tensor(index)}
    
    def __len__(self):
        return len(self.patches)

    def save(self,filename):
        with open(filename, 'wb') as outp:  
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def _patchify_window(window,satellite_img,patch_size,overlap,padding):
        step_size = patch_size[0] - overlap
        y = window.row_off-padding
        w,h = window.width,window.height

        max_y = window.row_off+h-step_size
        max_x = window.col_off+w-step_size

        
        n_y = 0#int((h-patch_size[0]+2*padding) // step_size)

        w_list = []
        while y <= max_y:
            n_x = 0 #int((w-patch_size[0]+2*padding) // step_size)
            x = window.col_off-padding
            while x <= max_x:
                w_patch = patch_size[0]
                h_patch = patch_size[0]
                w = Window(col_off=x,row_off=y,width=w_patch,height=h_patch)
                #transform = satellite_img.window_transform(w)
                w_list.append({"window":w})#,"transform":transform}) #TODO Transform really required?
                x += step_size
                n_x += 1
            n_y += 1
            y += step_size

        grid_shape = [n_y,n_x]
        
        return w_list,grid_shape

    @staticmethod
    def _get_padding(window,patch_size,overlap,pad_size):
        step = patch_size[0]-overlap
        x_rest = (int(window.width)-patch_size[0]+pad_size) % step 
        y_rest = (int(window.height)-patch_size[1]+pad_size) % step

        x_pad,y_pad = step-x_rest, step-y_rest
        left,right = pad_size,x_pad
        top,bottom = pad_size,y_pad
        return top,bottom,left,right








        
        





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



