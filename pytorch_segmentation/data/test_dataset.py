
from re import M
from rasterio.mask import raster_geometry_mask,mask
import geopandas as gdp
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
from patchify import patchify
from rasterio.windows import from_bounds, Window


from rasterio.io import MemoryFile
from tqdm import tqdm

from ..evaluate import evaluate
from ..utils.preprocessing import pad_image_even,resample_raster

class TestSatDataset(Dataset):
    def __init__(self,data_file_path=None,shape_path=None,mask_path=None,years=None,transform=None,patch_size=[256,256,3],overlap=0,resampling_factor = 1,
                resampling_method = "bilinear",padding=False,pad_value=0,file_extension=".vrt"):
        if data_file_path is not None:
            self.patch_size = patch_size
            self.overlap = overlap
            self.padding = padding
            self.pad_value = pad_value
            self.data_file_path = data_file_path
            self.shape_path = shape_path
            self.mask_path = mask_path
            self.transform = transform
            self.resampling_factor = resampling_factor
            self.resampling_method = resampling_method
            if years is None:
                years = list(range(2008,2019))
            self.years = years


            mask_files = sorted([os.path.join(mask_path,i) for i in os.listdir(mask_path) if i.endswith(".shp")])
            data_dict = {}
            for mask_file in mask_files:
                y = int(os.path.basename(mask_file).split("_")[0])
                raster_file = os.path.join(data_file_path,str(y)+file_extension)
                shape_file = os.path.join(shape_path,str(y)+".shp")
                if (y in years) and (os.path.isfile(raster_file)) and (os.path.isfile):
                    data_dict[y] = [shape_file,raster_file,mask_file]
            
            patches = []
            masks = []
            mapping = {}
            n_pixels = {}
            for y,f in data_dict.items():
                shp_file,raster_file,mask_file = f
                shape_df = gdp.read_file(shp_file).geometry
                if type(resampling_factor) == int:
                    res_factor = resampling_factor
                elif type(resampling_factor) == dict:
                    if y not in resampling_factor:
                        print(f"INFO: Year {y} not included in resampling!")
                        continue
                    res_factor = resampling_factor[y]

                n_pixels_year = 0
                with rasterio.open(raster_file) as src_sat:
                    self.sat_meta = src_sat.meta.copy()
                    start_idx = len(patches)
                    for s in range(len(shape_df)):
                        shp = shape_df.iloc[s]
                        b = shp.bounds
                        mask_df = gdp.read_file(mask_file).geometry
                        mask_df =  mask_df.cx[b[0]:b[2],b[1]:b[3]]

                        sat_arr, out_transform = mask(src_sat, [shp], crop=True)
                        # win = from_bounds(*shp.bounds,src_sat.transform)
                        # print(win)
                        # sat_arr = src_sat.read(window=win)

                        if np.all(sat_arr == 255) or np.all(sat_arr == 0):
                            print("Skipped shape due to nodata!")
                            continue 

                        out_meta = src_sat.meta
                        out_meta.update({"driver":"GTiff",
                                        "height": sat_arr.shape[1],
                                        "width": sat_arr.shape[2],
                                        "transform": out_transform})

                        if res_factor != 1:
                            sat_arr,out_transform,org_arr = resample_raster(sat_arr,out_transform,out_meta,res_factor,resampling_method)


                        if (sat_arr.shape[1] < self.patch_size[0]) or (sat_arr.shape[2] < self.patch_size[1]):
                            print("Skipped shape due to small size: ",sat_arr.shape)
                            continue 
                        
                        p = self._create_patches(sat_arr)

                        
                        if len(mask_df) > 0:
                            with MemoryFile() as memfile:
                                with memfile.open(**out_meta) as dataset:
                                    if res_factor != 1:
                                        dataset.write(org_arr)
                                        mask_arr,_,_ = raster_geometry_mask(dataset,mask_df,invert=True)
                                        out_meta["count"] = 1
                                        mask_arr = np.expand_dims(mask_arr,0)
                                        
                                        mask_arr,_,_ = resample_raster(mask_arr,out_meta["transform"],out_meta,res_factor,resampling_method)
                                        mask_arr = mask_arr.squeeze(0)
                                    else:
                                        dataset.write(sat_arr)
                                        mask_arr,_,_ = raster_geometry_mask(dataset,mask_df,invert=True)
                        else:
                            mask_arr = np.zeros(sat_arr.shape[1:],dtype="uint8")

                        m = self._create_mask_patches(mask_arr)

                        assert len(p) == len(m)        
                        masks.extend(m)
                        patches.extend(p)
                        if res_factor != 1:
                            n_pixels_year += np.prod(org_arr.shape)
                        else:
                            n_pixels_year += np.prod(sat_arr.shape)
                    mapping[y] = [start_idx,len(patches)] #start, end
                    n_pixels[y] = n_pixels_year
            
            self.patches = np.stack(patches).astype("uint8")
            self.masks = np.stack(masks).astype("uint8")
            self.mapping = mapping
            self.n_pixels = n_pixels
            print("Size: ",len(self.patches))
        else:
            raise ValueError("Missing dataset_path or data_file_path!")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.patches[idx] ).float()
        mask = torch.from_numpy(self.masks[idx] ).long()

        image = image / 255

        if self.transform:
            sample = self.transform(image=image,target=mask)
            image,mask = sample

        return {"x":image, "y":mask}


    def get_img(self,idx,transform=True):
        image = torch.from_numpy(self.patches[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).long()
        
        image = image / 255

        if (self.transform) and (transform):
            sample = self.transform(image,mask)
            image,_ = sample

        if image.size(0) > 3:
            image = image[:3,:,:]

        plt.imshow(image.permute(1, 2, 0).numpy()  )

    def get_mask(self,idx,transform=True):
        image = torch.from_numpy(self.patches[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).long()

        image = image / 255

        if (self.transform) and (transform):
            sample = self.transform(image,mask)
            _,mask = sample
        plt.imshow(  mask.numpy()  )
    
    def show_tuple(self,idx,transform=True):
        image = torch.from_numpy(self.patches[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).long()

        image = image / 255
        
        if (self.transform) and (transform):
            sample = self.transform(image,mask)
            image,mask = sample

        if image.size(0) > 3:
            image = image[:3,:,:]

        fig, axs = plt.subplots(1,2)
        axs[0].imshow(image.permute(1, 2, 0).numpy()  )
        axs[1].imshow(  mask.numpy())

    def evaluate(self,predictions,reduction=False):
        masks = torch.from_numpy(self.masks).long()
        total_score = 0

        
        total_pixels = sum(self.n_pixels.values())

        #Based on number of pixels per year
        for y,idxs in self.mapping.items():
            start,end = idxs
            y_true = masks[start:end]
            y_pred = predictions[start:end]
            score = evaluate(y_true,y_pred,reduction=reduction)
            print(f"Year {str(y)}: ",score)

            weight = self.n_pixels[y] / total_pixels
            total_score += score * weight
        print("Total score: ",total_score)
        return total_score

        #Weighting based on res_factor
        #n = 0
        # for y,idxs in self.mapping.items():
        #     start,end = idxs
        #     y_true = masks[start:end]
        #     y_pred = predictions[start:end]
        #     score = evaluate(y_true,y_pred,reduction=reduction)
        #     print(f"Year {str(y)}: ",score)

        #     if type(self.resampling_factor) == int:
        #         res_factor = self.resampling_factor
        #     elif type(self.resampling_factor) == dict:
        #         res_factor = self.resampling_factor[y]

        #     n_true = len(y_true) // res_factor
        #     total_score += score * n_true
        #     n += n_true
        # total_score = total_score / n
        # print("Total score: ",total_score)
        # return total_score


    def _create_patches(self,sat_arr):             
        if self.padding: 
            sat_arr,_ = pad_image_even(sat_arr,self.patch_size,self.overlap)

        patches = patchify(sat_arr, 
                                (self.patch_size[2], self.patch_size[0], self.patch_size[1]), 
                                step=self.patch_size[0]-self.overlap)[0]

        reshaped_patches = np.reshape(patches, 
                                        (patches.shape[0]*patches.shape[1], 
                                        patches.shape[2], patches.shape[3], patches.shape[3]))
                                        # = (#patches, 3, 256, 256)

        return reshaped_patches


    def _create_mask_patches(self,mask_arr):
        if self.padding:
            mask_arr,_  = pad_image_even(mask_arr,self.patch_size,self.overlap,dim=2,border_val=self.pad_value)
        patches = patchify(mask_arr,(self.patch_size[0], self.patch_size[1]), 
                                step=self.patch_size[0]-self.overlap)
        reshaped_patches = np.reshape(patches, 
                                    (patches.shape[0]*patches.shape[1], 
                                    patches.shape[2], patches.shape[3])) 
                                    # = (#patches, 256, 256)
        return reshaped_patches


