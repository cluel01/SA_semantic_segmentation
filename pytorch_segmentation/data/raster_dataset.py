import rasterio
from rasterio.windows import  Window
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

class RasterDataset(Dataset):
    def __init__(self,data_path,file_list=None,patch_size=[256,256],step_size=128,n_channels=3,remove_empty=True,padding=True,nodata_val=0,transform=None,file_extension=".tif"):
        


        self.patch_size = patch_size
        self.step_size = step_size
        self.n_channels = n_channels
        self.remove_empty = remove_empty
        self.nodata_val = nodata_val
        self.padding = padding
        self.transform = transform

        save_file = os.path.join(data_path,"data.csv")
        if os.path.isfile(save_file):
            self.df = pd.read_csv(save_file)
        else:
            windows = []

            if file_list is not None:
                files = file_list
            else:
                files = [os.path.join(data_path,i) for i in os.listdir(data_path) if i.endswith(file_extension)]

            for f in tqdm(files):
                w_list = self.split_windows(f,step_size,patch_size,remove_empty,padding,nodata_val)
                if len(w_list) > 0:
                    windows.extend(w_list)
            self.df = pd.DataFrame(windows,columns=["file","w_x","w_y","w_patch","h_patch"])
            self.df.to_csv(save_file,index=False)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raster_file,w_x,w_y,w_patch,h_patch = self.df.iloc[idx]
        window = Window(col_off=w_x,row_off=w_y,width=w_patch,height=h_patch)
        with rasterio.open(raster_file) as src:
            arr = src.read(window=window).transpose(1,2,0)
        
        if self.padding:
            p_size = (self.patch_size[0],self.patch_size[1],self.n_channels)
            if arr.shape != p_size:
                arr = np.pad(arr,[(0, p_size[i] - arr.shape[i]) for i in range(len(p_size))],"constant")
            
            
        
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)

        return img

    @staticmethod
    def split_windows(raster_file,step_size,patch_size,remove_empty=True,padding=False,nodata_val=0):
        with rasterio.open(raster_file) as src:
            if remove_empty or (padding == False):
                arr = src.read()
            w,h = src.width,src.height
            w_patch = patch_size[0]
            h_patch = patch_size[1]
            y = 0
            w_list = []
            while y <= h:
                x = 0
                while x <= w:
                    correct = True
                    win = [x,y,w_patch,h_patch] #Window(col_off=x,row_off=y,width=w_patch,height=h_patch)
                    if remove_empty:
                        win_arr = arr[:,y:y+h_patch,x:x+w_patch]
                        if  np.all(win_arr == nodata_val):
                            correct = False

                    if padding == False:
                        win_arr = arr[:,y:y+h_patch,x:x+w_patch]
                        if (w_patch,h_patch) != win_arr.shape[1:]:
                            correct = False
                      
                    if correct:
                        w_list.append([raster_file,*win])        
 
                    x += step_size
                y += step_size

        return w_list

class RasterfileDataset(Dataset):
    def __init__(self,target_path,data_path=None,patch_size=[256,256],step_size=128,n_channels=3,remove_empty=True,padding=True,nodata_val=0,transform=None,file_extension=".tif"):
        self.patch_size = patch_size
        self.step_size = step_size
        self.n_channels = n_channels
        self.remove_empty = remove_empty
        self.nodata_val = nodata_val
        self.padding = padding
        self.transform = transform

        save_file = os.path.join(target_path,"data.csv")
        if os.path.isfile(save_file):
            self.df = pd.read_csv(save_file)
        else:
            if not os.path.isdir(target_path):
                os.makedirs(target_path)

            windows = []
            files = [os.path.join(data_path,i) for i in os.listdir(data_path) if i.endswith(file_extension)]
            for f in tqdm(files):
                w_list = self.split_windows(f,target_path,step_size,patch_size,remove_empty,padding,nodata_val,file_extension)
                if len(w_list) > 0:
                    windows.extend(w_list)
            self.df = pd.DataFrame(windows,columns=["file"])
            self.df.to_csv(save_file,index=False)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raster_file = self.df.iloc[idx]["file"]
        with rasterio.open(raster_file,num_threads='all_cpus') as src:
            arr = src.read().transpose(1,2,0)
        
        if self.padding:
            p_size = (self.patch_size[0],self.patch_size[1],self.n_channels)
            if arr.shape != p_size:
                arr = np.pad(arr,[(0, p_size[i] - arr.shape[i]) for i in range(len(p_size))],"constant")
            
            
        
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)

        return img

    @staticmethod
    def split_windows(raster_file,target_path,step_size,patch_size,remove_empty=True,padding=False,nodata_val=0,file_extension=".tif"):
        with rasterio.open(raster_file,num_threads='all_cpus') as src:
            arr = src.read()
            kwargs = src.meta.copy()
            
            w,h = src.width,src.height
            w_patch = patch_size[0]
            h_patch = patch_size[1]
            y = 0
            w_list = []

            idx = 0
            while y <= h:
                x = 0
                while x <= w:
                    write = True
                    win_arr = arr[:,y:y+h_patch,x:x+w_patch]
                    win = Window(col_off=x,row_off=y,width=w_patch,height=h_patch)
                    if padding == False:
                        if (w_patch,h_patch) != win_arr.shape[1:]:
                            write = False
                    if remove_empty:
                        if np.all(win_arr == nodata_val):
                            write = False
                    if write:
                        idx += 1
                        dst_file = raster_file.split(".")[0] + "_" + str(idx) + file_extension
                        dst_file = os.path.join(target_path,os.path.basename(dst_file))

                        if not os.path.isfile(dst_file):
                            kwargs.update({
                                'tiled': True,
                                'height': win_arr.shape[1],
                                'width': win_arr.shape[2],
                                "compress": "DEFLATE",
                                "predictor": 2,
                                'transform': rasterio.windows.transform(win, src.transform)})
                            with rasterio.open(dst_file,'w',**kwargs,num_threads='all_cpus') as dst:
                                dst.write(win_arr)
                        w_list.append(dst_file)
                    x += step_size
                y += step_size

        return w_list