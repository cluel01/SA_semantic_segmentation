
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
import logging
import sys
import time
import shutil
import tarfile
from torchvision.transforms.functional import resize

from rasterio.io import MemoryFile
from tqdm import tqdm

from ..evaluate import evaluate
from ..utils.plotting import  save_prediction_plots
from ..utils.preprocessing import pad_image_even,resample_raster

class TestSatDataset(Dataset):
    def __init__(self,data_path=None,years=None,transform=None,patch_size=[256,256],n_channels=3,overlap=0,
                padding=False,pad_value=0,file_extension=".tif",shp_extension=".geojson",resampling_factor=1):
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.overlap = overlap
        self.padding = padding
        self.pad_value = pad_value
        self.data_path = data_path
        self.transform = transform
        if years is None:
            years = list(range(2008,2019))
        self.years = years
        self.resampling_factor = resampling_factor

        
        patches = []
        masks = []
        mapping = {}

        start_idx = 0
        for y in years:
            shp_file = os.path.join(data_path,str(y),"labels"+shp_extension)
            r_path = os.path.join(data_path,str(y))
            shape_df = gdp.read_file(shp_file).geometry
            raster_files = [os.path.join(r_path,i) for i in os.listdir(r_path) if i.endswith(file_extension)]
            patches_satellite,patches_mask,file_mapping = self._create_patches(raster_files,shape_df)
            
            del shape_df
            # X = np.empty(tuple((len(patches_satellite),*patch_size[::-1])),dtype="uint8")
            # for i in range(len(patches_satellite)):
            #     X[i,:,:,:] = patches_satellite[i]

            assert len(patches_satellite) == len(patches_mask)


            if type(resampling_factor) == int:
                res_factor = resampling_factor
            elif type(resampling_factor) == dict:
                if y not in resampling_factor:
                    print(f"INFO: Year {y} not included in resampling!")
                    continue
                res_factor = resampling_factor[y]

            if res_factor != 1:
                patches_satellite = np.stack(patches_satellite).astype("uint8")
                patches_mask = np.stack(patches_mask).astype("uint8")

                ps = torch.from_numpy(patches_satellite)
                pm = torch.from_numpy(patches_mask)

                res_shape = torch.tensor(patches_satellite.shape[-2:]) * res_factor
                ps = resize(ps,res_shape.tolist()).numpy()
                pm = resize(pm,res_shape.tolist()).numpy()

                patches_satellite = []
                patches_mask = []
                for i in range(len(ps)):
                    s = ps[i]
                    m = pm[i]
                    s = patchify(s,[n_channels]+patch_size,patch_size[0]).reshape(-1,*[n_channels]+patch_size)
                    m = patchify(m,patch_size,patch_size[0]).reshape(-1,*patch_size)
                    patches_satellite.extend(s)
                    patches_mask.extend(m)

            patches.extend(patches_satellite)
            masks.extend(patches_mask)


            mapping[y] = [start_idx,start_idx+len(patches_satellite)] #start, end
            start_idx += len(patches_satellite)
        
        self.patches = np.stack(patches).astype("uint8")
        self.masks = np.stack(masks).astype("uint8")
        self.mapping = mapping
        print("Size: ",len(self.patches))


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

    def evaluate(self,predictions,model_name,save_dir,save_images=False,filter_true_empty=False,
                reduction=False,tmpdir=None):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        run_name = "run_" + timestr
        save_dir = os.path.join(save_dir,model_name,run_name)
        os.makedirs(save_dir, exist_ok=True)
        
        file = os.path.join(save_dir,"results.txt")
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        if tmpdir is not None:
            tmpdir = os.path.join(tmpdir,timestr)
            os.makedirs(tmpdir,exist_ok=True)
        
        masks = torch.from_numpy(self.masks).long()
        total_score = 0

        logging.info(f"Model: {model_name}")

        results = {}

        for y,idxs in self.mapping.items():
            start,end = idxs
            y_true = masks[start:end]
            y_pred = predictions[start:end]
            
            if save_images:
                score,n_images = evaluate(y_true,y_pred,reduction=reduction,aggregate=False,filter_true_empty=filter_true_empty)
                idxs = score.sort_values(by="iou",ascending=False).index.values #largest scores first
                x = torch.from_numpy(self.patches[start:end])
                
                if tmpdir is not None:
                    save_dir_year = os.path.join(tmpdir,str(y))
                else:
                    save_dir_year = os.path.join(save_dir,str(y))

                save_prediction_plots(x,y_true,y_pred,idxs,score,save_dir=save_dir_year)
                score = score.mean()
            else:
                score,n_images = evaluate(y_true,y_pred,reduction=reduction,filter_true_empty=filter_true_empty)

            logging.info("##############################")
            logging.info(f"Year {str(y)}: {score.to_string(dtype=False)}")
            logging.info(f"Number of patches: {n_images}")
            if filter_true_empty:
                logging.info(f"Number of patches without filtering: {len(y_true)}")

                    

            results[y] = {"score":score,"n_images":n_images}

        total_images,total_score = 0,0

        [total_images := total_images + i["n_images"] for i in results.values()]
        for y,item in results.items():
            score,n_images = item.values()
            weight = (n_images / total_images)
            total_score += score * weight

        logging.info(f"\n##############################")
        logging.info(f"Total score: {total_score.to_string(dtype=False)}")

        if tmpdir is not None:
            out_file = os.path.join(save_dir,"images_"+timestr+".tgz")
            with tarfile.open(out_file, "w:gz") as tar:
                tar.add(tmpdir, arcname=os.path.basename(tmpdir))
            shutil.rmtree(tmpdir)
        
        logger = logging.getLogger()
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

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

    def _create_patches(self,raster_files,shape_df):
        patches_satellite = []
        patches_mask = []
        start_idx = 0
        mapping = {}
        for f in tqdm(raster_files):
            with rasterio.open(f) as src_sat:
                b_left,b_bottom,b_right,b_top = src_sat.bounds
                shape_df_filter = shape_df.cx[b_left:b_right,b_bottom:b_top]

                satellite_area_arr = src_sat.read()
                
                if np.all(satellite_area_arr == 255) or np.all(satellite_area_arr == 0):
                    print("Skipped shape due to nodata!")
                    continue 

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
                    mask_arr,_,_ = raster_geometry_mask(src_sat,shape_df_filter,invert=True)
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