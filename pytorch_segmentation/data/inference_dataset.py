import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import from_bounds, Window
import os
import numpy as np
from shapely import geometry
import fiona
import pickle
import pandas as pd


## SatInferenceDataset for containing satellite imagery areas based on given shapes -> only returning X without mask
class SatInferenceDataset(Dataset):
    def __init__(self,dataset_path=None,data_file_path=None,shape_path=None,transform=None,patch_size=[256,256,3],overlap=128,padding=64,pad_value=0,file_extension=".shp"):
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
                self.pad_value = pad_value
                self.data_file_path = data_file_path
                self.shape_path = shape_path

                patches = []
                shapes = []
                shape_idx = 0
                
                if type(shape_path) == list:
                    shape_files = shape_path
                elif type(shape_path) == str:
                    shape_files = [shape_path]
                #shape_files = [os.path.join(shape_path,i) for i in os.listdir(shape_path) if i.endswith(file_extension)]
                with rasterio.open(data_file_path) as src_sat:
                    self.sat_meta = src_sat.meta.copy()
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
                                                "width":win.width,"height":win.height,"sat_meta":src_sat.meta.copy()})
                                patches.append(win_list)
                                shape_idx += 1

                self.patches = np.vstack(patches)
                self.shapes = pd.DataFrame(shapes)
            else:
                raise ValueError("Missing dataset_path or data_file_path!")
        else:
            self.load(dataset_path)
        

    def __getitem__(self, index):
        patch = self.patches[index]
        with rasterio.open(self.data_file_path) as src_sat:
            win = Window(*patch)
            img = src_sat.read(window=win)
            if img.size == 0:
                img = np.empty(self.t_patch_size)
                img = img.fill_(self.pad_value)
            elif tuple(img.shape) != tuple(self.t_patch_size):
                top = 0
                left = 0
                if win.row_off < 0:
                    top = (self.patch_size[0]- img.shape[1]) // 2
                if win.col_off < 0:
                    left = (self.patch_size[0]- img.shape[2]) // 2
                bottom = self.patch_size[0]-img.shape[1]-top
                right = self.patch_size[0]-img.shape[2]-left
                img = np.pad(img,[(0,0),(top,bottom),(left,right)],"edge")
                #img = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_REPLICATE,value=self.pad_value)
        img = img / 255
        #img = torch.as_tensor(img).float().contiguous() 
        if self.transform:
            img = self.transform(img)
        return img,index
    
    def __len__(self):
        return len(self.patches)

    def _get_config(self):
        return {"sat_meta":self.sat_meta,"transform":self.transform,"patch_size":self.patch_size,"t_patch_size":self.t_patch_size,
                "overlap":self.overlap,"padding":self.padding,"pad_value":self.pad_value,"data_file_path":self.data_file_path,"shape_path":self.shape_path}

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