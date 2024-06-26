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
from torchvision.transforms.functional import resize
import geopandas as gpd
from tqdm import tqdm
import time

## SatInferenceDataset for containing satellite imagery areas based on given shapes -> only returning X without mask
class SatInferenceDataset(Dataset):
    def __init__(self,dataset_path=None,data_file_path=None,shape_file=None,shape_idx=None,transform=None,
                patch_size=[256,256],n_channels=3,overlap=128,padding=64,nodata=0,pad_mode="reflect",
                rescale_factor=1):
        if dataset_path is None:
            if data_file_path is not None:
                self.transform = transform
                self.n_channels = n_channels
                self.overlap = overlap
                self.padding = padding
                self.pad_mode = pad_mode

                self.rescale_factor = rescale_factor
                self.patch_size = (patch_size[0],patch_size[1])
                self.cutout_shape = (patch_size[0] //  rescale_factor ,patch_size[1] // rescale_factor)
                self.nodata = nodata
                self.data_file_path = data_file_path
                self.shape_file = shape_file
                self.shape_idx = shape_idx

                if (overlap >= patch_size[0]) or (overlap >= patch_size[1]):
                    raise ValueError(f"Too large overlap {overlap} for patch_size {patch_size}")

                patches = []
                shapes = []

                with rasterio.open(data_file_path) as src_sat:
                    self.sat_meta = src_sat.meta.copy()
                    sat_shape = geometry.box(*src_sat.bounds)
                    start_idx = 0

                    if shape_file is None:
                        box = geometry.box(*src_sat.bounds)
                        df = gpd.GeoDataFrame({'geometry':[box]})
                        df.crs = src_sat.crs
                        time_str = time.strftime("%d_%m_%Y_%H%M%S")
                        shape_file = "tmp_shapes_"+time_str+".geojson"
                        df.to_file(shape_file)
                    with fiona.open(shape_file) as src_shp:
                        if shape_idx is None:
                            s_idxs = range(len(src_shp))
                        else:
                            s_idxs = shape_idx
        
                        for i in tqdm(s_idxs):
                            shp = src_shp[i]
                            name = os.path.basename(shape_file).split(".")[0] + "_" + str(i+1)
                            if shp is None:
                                continue
                            s = geometry.shape(shp["geometry"])
                            if not sat_shape.intersects(s):
                                print(f"Shape {name} does not intersect!")
                                continue
                            win = from_bounds(*s.bounds,src_sat.transform)
                            #pad = self._get_padding(win,patch_size,overlap,padding,rescale_factor)
                            win_list,grid_shape = self._patchify_window(win,self.cutout_shape,overlap,padding)
                            shapes.append({"shape_id":i,"transform":src_sat.window_transform(win),"padding":padding,
                                            "start_idx":start_idx,"grid_shape":grid_shape,"name":name,
                                            "width":win.width,"height":win.height,"sat_meta":src_sat.meta.copy(),
                                            "rescale_factor":rescale_factor,"patch_size":self.patch_size,"cutout_shape":self.cutout_shape})
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
            img = src_sat.read(window=win,fill_value=nodata)

            img = self._crop_nodata(img,nodata)
                
        img = torch.from_numpy(img).float().contiguous()
        if self.rescale_factor != 1:
            img = resize(img,self.patch_size)

        img = img / 255

        if self.transform:
            sample = self.transform(image=img,target=None)
            img,_=  sample

        return img,index
    
    def __len__(self):
        return len(self.patches)

    def _get_config(self):
        return {"sat_meta":self.sat_meta,"transform":self.transform,"patch_size":self.patch_size,"n_channels":self.n_channels,"shape_idx":self.shape_idx,"rescale_factor":self.rescale_factor,
                "overlap":self.overlap,"padding":self.padding,"nodata":self.nodata,"data_file_path":self.data_file_path,"shape_file":self.shape_file,"shape_idx":self.shape_idx,"pad_mode":self.pad_mode,
                "cutout_shape":self.cutout_shape}

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
            img = src_sat.read(window=win,fill_value=nodata)
            
            img = self._crop_nodata(img,nodata)
        
        img = torch.from_numpy(img).float().contiguous()
        if self.rescale_factor != 1.:
            img = resize(img,self.patch_size)

        img = img / 255

        if (self.transform) and (transform):
            sample = self.transform(image=img,target=None)
            img,_=  sample

        if img.size(0) > 3:
            img = img[:3,:,:]

        plt.imshow(img.permute(1, 2, 0)  )

    def _crop_nodata(self,arr,nodata):
        # create a boolean mask indicating which elements are not equal to nodata
        nodata_mask = (arr != nodata)

        # find the indices of the non-zero elements in the image
        w_idx = np.where(np.any(nodata_mask, axis=(0, 1)))[0]
        h_idx = np.where(np.any(nodata_mask, axis=(0, 2)))[0]

        # find the bounding box of the non-zero region
        top = h_idx[0]
        bottom = h_idx[-1] + 1
        left = w_idx[0]
        right = w_idx[-1] + 1

        # crop the image to remove the padding
        arr = arr[:, top:bottom, left:right]
        bottom = self.cutout_shape[0] - bottom
        right = self.cutout_shape[1] - right

        # top,bottom = bottom,top
        # left,right = right,left

        padding = [
            (0, 0),
            (max(-top, 0), max(bottom, 0)),
            (max(-left, 0), max(right, 0))
        ]

        # if sum([arr.shape[1], top,bottom]) < self.cutout_shape[0]:
        #     if  top > 0 and bottom == 0:
        #          top += self.cutout_shape[0] - sum([arr.shape[1], top]) 
        #     elif bottom > 0 and  top == 0:
        #          bottom += self.cutout_shape[0] - sum([arr.shape[1], bottom])
        #     elif  top > 0 and bottom > 0:
        #         top += self.cutout_shape[0] - (sum([arr.shape[1], top,bottom]) // 2)
        #         bottom += self.cutout_shape[0] - sum([arr.shape[1], top,bottom])
        #     else:
        #         import sys
        #         sys.exit("Error in padding")
        # if sum([arr.shape[2], left,right]) < self.cutout_shape[1]:
        #     if  left > 0 and right == 0:
        #          left += self.cutout_shape[1] - sum([arr.shape[2], left]) 
        #     elif right > 0 and  left == 0:
        #          right += self.cutout_shape[1] - sum([arr.shape[2], left,right])
        #     elif  left > 0 and right > 0:
        #         left += self.cutout_shape[1] - (sum([arr.shape[2], left]) // 2) 
        #         right += self.cutout_shape[1] - sum([arr.shape[2], left,right])
        #     else:
        #         import sys
        #         sys.exit("Error in padding")

        # pad the image with zeros
        arr = np.pad(arr, padding, self.pad_mode)
        # if sum([top,bottom,left,right]) > 0:
        #     arr = np.pad(arr,[(0,0),(top,bottom),(left,right)],)

        return arr

    @staticmethod
    def _patchify_window(window,patch_size,overlap,padding,):
        step_size = patch_size[0] - overlap
        y = window.row_off-padding
        w,h = window.width,window.height

        max_y = window.row_off+h-step_size+padding
        max_x = window.col_off+w-step_size+padding

        
        n_y = 0#int((h-patch_size[0]+2*padding) // step_size)

        w_list = []
        while y <= max_y:
            n_x = 0 #int((w-patch_size[0]+2*padding) // step_size)
            x = window.col_off-padding
            while x <= max_x:
                w_patch = patch_size[0]
                h_patch = patch_size[1]
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

    # @staticmethod
    # def _get_padding(window,patch_size,overlap,pad_size,rescale_factor):
    #     step = patch_size[0]-overlap
    #     x_rest = (int(window.width * rescale_factor)-patch_size[0]+pad_size) % step 
    #     y_rest = (int(window.height * rescale_factor)-patch_size[1]+pad_size) % step 

    #     x_rest = int(window.width * rescale_factor) 

    #     x_pad,y_pad = step-x_rest, step-y_rest
    #     left,right = pad_size,x_pad
    #     top,bottom = pad_size,y_pad
    #     return top,bottom,left,right


# ## SatInferenceDataset for containing satellite imagery areas based on given shapes -> only returning X without mask
# class SatInferenceDataset(Dataset):
#     def __init__(self,dataset_path=None,data_file_path=None,shape_file=None,shape_idx=None,transform=None,
#                 patch_size=[256,256],n_channels=3,overlap=128,padding=64,nodata=0,pad_mode="reflect",
#                 rescale_factor=1):
#         if dataset_path is None:
#             if data_file_path is not None:
#                 self.transform = transform
#                 self.patch_size = patch_size
#                 patch_size =  np.array(patch_size,dtype=int)
#                 self.n_channels = n_channels
#                 self.overlap = overlap
#                 self.padding = padding
#                 self.pad_mode = pad_mode

#                 patch_size = patch_size // rescale_factor
#                 self.rescale_factor = rescale_factor
 
#                 self.nodata = nodata
#                 self.data_file_path = data_file_path
#                 self.shape_file = shape_file
#                 self.shape_idx = shape_idx

#                 if (overlap >= patch_size[0]) or (overlap >= patch_size[1]):
#                     raise ValueError(f"Too large overlap {overlap} for patch_size {patch_size}")

#                 patches = []
#                 shapes = []

#                 with rasterio.open(data_file_path) as src_sat:
#                     self.sat_meta = src_sat.meta.copy()
#                     sat_shape = geometry.box(*src_sat.bounds)
#                     start_idx = 0

#                     if shape_file is None:
#                         box = geometry.box(*src_sat.bounds)
#                         df = gpd.GeoDataFrame({'geometry':[box]})
#                         df.crs = src_sat.crs
#                         shape_file = "tmp_"+ data_file_path + ".shp"
#                         df.to_file(shape_file)
#                     with fiona.open(shape_file) as src_shp:
#                         if shape_idx is None:
#                             s_idxs = range(len(src_shp))
#                         else:
#                             s_idxs = shape_idx
        
#                         for i in tqdm(s_idxs):
#                             shp = src_shp[i]
#                             name = os.path.basename(shape_file).split(".")[0] + "_" + str(i+1)
#                             if shp is None:
#                                 continue
#                             s = geometry.shape(shp["geometry"])
#                             if not sat_shape.intersects(s):
#                                 print(f"Shape {name} does not intersect!")
#                                 continue
#                             win = from_bounds(*s.bounds,src_sat.transform)
#                             #pad = self._get_padding(win,patch_size,overlap,padding,rescale_factor)
#                             win_list,grid_shape = self._patchify_window(win,patch_size,overlap,padding)
#                             shapes.append({"shape_id":i,"transform":src_sat.window_transform(win),"padding":padding,
#                                             "start_idx":start_idx,"grid_shape":grid_shape,"name":name,
#                                             "width":win.width,"height":win.height,"sat_meta":src_sat.meta.copy(),
#                                             "rescale_factor":rescale_factor,"patch_size":self.patch_size})
#                             patches.append(win_list)
#                             start_idx += len(win_list)

#                 self.patches = np.vstack(patches)
#                 self.shapes = pd.DataFrame(shapes)
#             else:
#                 raise ValueError("Missing dataset_path or data_file_path!")
#         else:
#             self.load(dataset_path)
#             self.transform = transform
        

#     def __getitem__(self, index):
#         patch = self.patches[index]
#         with rasterio.open(self.data_file_path) as src_sat:
#             win = Window(*patch)
#             if self.nodata == 0: #TODO Bug with Rwanda dataset which is not filling with 0 if nodata == 0
#                 nodata = -1
#             else:
#                 nodata = self.nodata
#             img = src_sat.read(window=win,boundless=True,fill_value=nodata)

#             if not np.all(img == self.nodata):
#                 img,padding = self._crop_nodata(img,self.nodata)
#                 #elif tuple(img.shape) != tuple(self.t_patch_siez):
#                 if np.sum(padding) > 0:
#                     top,bottom,left,right = padding
#                     img = np.pad(img,[(0,0),(top,bottom),(left,right)],self.pad_mode)
#                     #img = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_REPLICATE,value=self.pad_value)
#         img = torch.from_numpy(img).float().contiguous()
#         if self.rescale_factor != 1:
#             img = resize(img,self.patch_size)

#         img = img / 255

#         if self.transform:
#             sample = self.transform(image=img,target=None)
#             img,_=  sample

#         return img,index
    
#     def __len__(self):
#         return len(self.patches)

#     def _get_config(self):
#         return {"sat_meta":self.sat_meta,"transform":self.transform,"patch_size":self.patch_size,"n_channels":self.n_channels,"shape_idx":self.shape_idx,"rescale_factor":self.rescale_factor,
#                 "overlap":self.overlap,"padding":self.padding,"nodata":self.nodata,"data_file_path":self.data_file_path,"shape_file":self.shape_file,"shape_idx":self.shape_idx,"pad_mode":self.pad_mode}

#     def save(self,filename):
#         cfg = self._get_config()
#         obj = [self.patches,self.shapes,cfg]
#         with open(filename, 'wb') as outp:  
#             pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
#         del obj

#     def load(self,filename):
#         with open(filename, 'rb') as inp:
#             obj = pickle.load(inp)
#         self.patches,self.shapes,cfg  = obj
#         for k in cfg.keys():
#             setattr(self,k,cfg[k])

#     def get_img(self,index,transform=True):
#         patch = self.patches[index]
#         with rasterio.open(self.data_file_path) as src_sat:
#             win = Window(*patch)
#             if self.nodata == 0: #TODO Bug with Rwanda dataset which is not filling with 0 if nodata == 0
#                 nodata = -1
#             else:
#                 nodata = self.nodata
#             img = src_sat.read(window=win,boundless=True,fill_value=nodata)
#             if not np.all(img == self.nodata):
#                 img,padding = self._crop_nodata(img,self.nodata)
#                 #elif tuple(img.shape) != tuple(self.t_patch_siez):
#                 if np.sum(padding) > 0:
#                     top,bottom,left,right = padding
#                     img = np.pad(img,[(0,0),(top,bottom),(left,right)],self.pad_mode)
#                     #img = cv2.copyMakeBorder(img.transpose(1,2,0), top, bottom, left, right, cv2.BORDER_REPLICATE,value=self.pad_value)
#         img = torch.from_numpy(img).float().contiguous()
#         if self.rescale_factor != 1.:
#             img = resize(img,self.patch_size)

#         img = img / 255

#         if (self.transform) and (transform):
#             sample = self.transform(image=img,target=None)
#             img,_=  sample

#         if img.size(0) > 3:
#             img = img[:3,:,:]

#         plt.imshow(img.permute(1, 2, 0)  )

#     @staticmethod
#     #For speedup purposes it is only checked one band
#     def _crop_nodata(arr,nodata):
#         #check if edges are nodata
#         padding =  np.zeros(4,dtype="int")
#         arr_T = arr.T
#         if (np.all(arr_T[0][0][:] == nodata)) or (np.all(arr_T[-1][-1][:] == nodata)): #if there are nodata values in the corners
#             horizontal = (np.sum(arr[0] == nodata,axis=0) == arr.shape[1])
#             vertical = (np.sum(arr[0] == nodata,axis=1) == arr.shape[2])

#             top = np.argmax(vertical == False) #stops after first false
#             bottom = np.argmax(vertical[::-1] == False)
#             left = np.argmax(horizontal == False)
#             right = np.argmax(horizontal[::-1] == False)            
#             padding[:] =  [top,bottom,left,right]
#             arr = arr[:,top:arr.shape[1]-bottom,left:arr.shape[2]-right]
#         return arr,padding

#     @staticmethod
#     def _patchify_window(window,patch_size,overlap,padding,):
#         step_size = patch_size[0] - overlap
#         y = window.row_off-padding
#         w,h = window.width,window.height

#         max_y = window.row_off+h-step_size+padding
#         max_x = window.col_off+w-step_size+padding

        
#         n_y = 0#int((h-patch_size[0]+2*padding) // step_size)

#         w_list = []
#         while y <= max_y:
#             n_x = 0 #int((w-patch_size[0]+2*padding) // step_size)
#             x = window.col_off-padding
#             while x <= max_x:
#                 w_patch = patch_size[0]
#                 h_patch = patch_size[1]
#                 w = Window(col_off=x,row_off=y,width=w_patch,height=h_patch)
#                 #transform = satellite_img.window_transform(w)
#                 #w_list.append({"window":w})#,"transform":transform}) #TODO Transform really required?
#                 w_list.append([x,y,w_patch,h_patch])
#                 x += step_size
#                 n_x += 1
#             n_y += 1
#             y += step_size

#         grid_shape = [n_y,n_x]
#         w_arr = np.array(w_list,dtype="float")
#         return w_arr,grid_shape



    # @staticmethod
    # def _get_padding(window,patch_size,overlap,pad_size,rescale_factor):
    #     step = patch_size[0]-overlap
    #     x_rest = (int(window.width * rescale_factor)-patch_size[0]+pad_size) % step 
    #     y_rest = (int(window.height * rescale_factor)-patch_size[1]+pad_size) % step 

    #     x_rest = int(window.width * rescale_factor) 

    #     x_pad,y_pad = step-x_rest, step-y_rest
    #     left,right = pad_size,x_pad
    #     top,bottom = pad_size,y_pad
    #     return top,bottom,left,right