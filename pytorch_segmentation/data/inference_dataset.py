import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import from_bounds, Window
import os
import numpy as np
from shapely import geometry
import fiona
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import cv2

## SatInferenceDataset for containing satellite imagery areas based on given shapes -> only returning X without mask
class SatInferenceDataset(Dataset):
    def __init__(self,dataset_path=None,data_file_path=None,shape_file=None,shape_idx=None,transform=None,
                patch_size=[256,256,3],overlap=128,padding=64,nodata=0,pad_mode="reflect"):
        if dataset_path is None:
            if data_file_path is not None:
                self.transform = transform
                if type(patch_size) == list:
                    self.patch_size = np.array(patch_size)
                else: #numpy array
                    self.patch_size = patch_size
                self.t_patch_size = self.patch_size[[2,0,1]]  # patch_size for Tensors 
                self.overlap = overlap
                self.padding = padding
                self.pad_mode = pad_mode

                self.nodata = nodata
                self.data_file_path = data_file_path
                self.shape_file = shape_file
                self.shape_idx = shape_idx

                patches = []
                shapes = []

                with rasterio.open(data_file_path) as src_sat:
                    self.sat_meta = src_sat.meta.copy()
                    sat_shape = geometry.box(*src_sat.bounds)
                    start_idx = 0
                    with fiona.open(shape_file) as src_shp:
                        if shape_idx is None:
                            s_idxs = range(len(src_shp))
                        else:
                            s_idxs = shape_idx
                        for i in s_idxs:
                            shp = src_shp[i]
                            name = os.path.basename(shape_file).split(".")[0] + "_" + str(i+1)
                            s = geometry.shape(shp["geometry"])
                            if not sat_shape.intersects(s):
                                print(f"Shape {name} does not intersect!")
                                continue
                            win = from_bounds(*s.bounds,src_sat.transform)
                            pad = self._get_padding(win,patch_size,overlap,padding)
                            win_list,grid_shape = self._patchify_window(win,src_sat,patch_size,overlap,padding)
                            shapes.append({"shape_id":i,"transform":src_sat.window_transform(win),"padding":pad,
                                            "start_idx":start_idx,"grid_shape":grid_shape,"name":name,
                                            "width":win.width,"height":win.height,"sat_meta":src_sat.meta.copy()})
                            patches.append(win_list)
                            start_idx += len(win_list)

                self.patches = np.vstack(patches)
                self.shapes = pd.DataFrame(shapes)
            else:
                raise ValueError("Missing dataset_path or data_file_path!")
        else:
            self.load(dataset_path)
            self.transform = transform
        

    def __getitem__(self, index):
        patch = self.patches[index]
        with rasterio.open(self.data_file_path) as src_sat:
            win = Window(*patch)
            if self.nodata == 0: #TODO Bug with Rwanda dataset which is not filling with 0 if nodata == 0
                nodata = -1
            else:
                nodata = self.nodata
            img = src_sat.read(window=win,boundless=True,fill_value=nodata)

            if not np.all(img == self.nodata):
                img,padding = self._crop_nodata(img,self.nodata)
                #elif tuple(img.shape) != tuple(self.t_patch_siez):
                if np.sum(padding) > 0:
                    top,bottom,left,right = padding
                    img = np.pad(img,[(0,0),(top,bottom),(left,right)],self.pad_mode)
                    #img = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_REPLICATE,value=self.pad_value)
        img = torch.from_numpy(img).float().contiguous()

        img = img / 255

        if self.transform:
            sample = self.transform(image=img,target=None)
            img,_=  sample

        return img,index
    
    def __len__(self):
        return len(self.patches)

    def _get_config(self):
        return {"sat_meta":self.sat_meta,"transform":self.transform,"patch_size":self.patch_size,"t_patch_size":self.t_patch_size,"shape_idx":self.shape_idx,
                "overlap":self.overlap,"padding":self.padding,"nodata":self.nodata,"data_file_path":self.data_file_path,"shape_file":self.shape_file,"shape_idx":self.shape_idx}

    def save(self,filename):
        cfg = self._get_config()
        obj = [self.patches,self.shapes,cfg]
        with open(filename, 'wb') as outp:  
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        del obj

    def load(self,filename):
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        self.patches,self.shapes,cfg  = obj
        for k in cfg.keys():
            setattr(self,k,cfg[k])

    def get_img(self,index,transform=True):
        patch = self.patches[index]
        with rasterio.open(self.data_file_path) as src_sat:
            win = Window(*patch)
            if self.nodata == 0: #TODO Bug with Rwanda dataset which is not filling with 0 if nodata == 0
                nodata = -1
            else:
                nodata = self.nodata
            img = src_sat.read(window=win,boundless=True,fill_value=nodata)
            if not np.all(img == self.nodata):
                img,padding = self._crop_nodata(img,self.nodata)
                #elif tuple(img.shape) != tuple(self.t_patch_siez):
                if np.sum(padding) > 0:
                    top,bottom,left,right = padding
                    img = np.pad(img,[(0,0),(top,bottom),(left,right)],self.pad_mode)
                    #img = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_REPLICATE,value=self.pad_value)
        img = torch.from_numpy(img).float().contiguous()

        img = img / 255

        if (self.transform) and (transform):
            sample = self.transform(image=img,target=None)
            img,_=  sample

        if img.size(0) > 3:
            img = img[:3,:,:]

        plt.imshow(img.permute(1, 2, 0)  )

    @staticmethod
    #For speedup purposes it is only checked one band
    def _crop_nodata(arr,nodata):
        #check if edges are nodata
        padding =  np.zeros(4,dtype="int")
        arr_T = arr.T
        if (np.all(arr_T[0][0][:] == nodata)) or (np.all(arr_T[-1][-1][:] == nodata)): #if there are nodata values in the corners
            horizontal = (np.sum(arr[0] == nodata,axis=0) == arr.shape[1])
            vertical = (np.sum(arr[0] == nodata,axis=1) == arr.shape[2])

            top = np.argmax(vertical == False) #stops after first false
            bottom = np.argmax(vertical[::-1] == False)
            left = np.argmax(horizontal == False)
            right = np.argmax(horizontal[::-1] == False)            
            padding[:] =  [top,bottom,left,right]
            arr = arr[:,top:arr.shape[1]-bottom,left:arr.shape[2]-right]
        return arr,padding

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
                #w_list.append({"window":w})#,"transform":transform}) #TODO Transform really required?
                w_list.append([x,y,w_patch,h_patch])
                x += step_size
                n_x += 1
            n_y += 1
            y += step_size

        grid_shape = [n_y,n_x]
        w_arr = np.array(w_list,dtype="float")
        return w_arr,grid_shape

    @staticmethod
    def _get_padding(window,patch_size,overlap,pad_size):
        step = patch_size[0]-overlap
        x_rest = (int(window.width)-patch_size[0]+pad_size) % step 
        y_rest = (int(window.height)-patch_size[1]+pad_size) % step

        x_pad,y_pad = step-x_rest, step-y_rest
        left,right = pad_size,x_pad
        top,bottom = pad_size,y_pad
        return top,bottom,left,right