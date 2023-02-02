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
from ..utils.plotting import save_ground_truth_plots

class InMemorySatDataset(Dataset):
    def __init__(self,data_file_path=None,mask_path=None,X=None,y=None,transform=None,n_channels=3,patch_size=[256,256],overlap=0,padding=False,pad_value=0,file_extension=".tif",mode="segmentation",indices=None):
        self.transform = transform
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.overlap = overlap
        self.padding = padding
        self.pad_value = pad_value
        self.data_file_path = data_file_path
        self.mask_path = mask_path
        self.mode = mode

        if (X is not None) and (y is not None):
            self.X = np.array(X,dtype="uint8")
            self.y = np.array(y,dtype="uint8")
        else:
            if (data_file_path is not None) and (mask_path is not None):
                satellite_img = rasterio.open(data_file_path)

                mask_areas = {}
                for i in os.listdir(mask_path):
                    if i.endswith(file_extension):
                        m = rasterio.open(os.path.join(mask_path,i))
                        if (m.shape[0] < self.patch_size[0]) or (m.shape[1] < self.patch_size[1]):
                            print(f"Shape {i} is too small for patch size with size: {m.shape}")
                        else:
                            mask_areas[i] = m

                assert len(mask_areas) > 0
                
                patches_masks,file_mapping = self._create_mask_patches(mask_areas)
                y  = np.vstack(patches_masks).astype("uint8")
                self.y = y

                self.X = np.empty((len(self.y),n_channels,*patch_size),dtype="uint8")
                self._create_data_patches(satellite_img,mask_areas)            

                self.file_mapping = file_mapping
                
                for i in mask_areas.values():
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
        image = torch.from_numpy(self.X[idx]).float()
        mask = torch.from_numpy(self.y[idx]).long()

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

    def get_img(self,idx,transform=True):
        image = torch.from_numpy(self.X[idx]).float()
        mask =torch.from_numpy( self.y[idx]).long()

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
        plt.imshow(  mask.numpy())

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
        #plt.show()


    def get_train_test_set(self,test_size,train_transform=None,test_transform=None,seed=42):
        if train_transform is None:
            train_transform = self.transform
        
        indices_train,indices_test = train_test_split(self.indices, test_size=test_size, random_state=seed)
        train_set = InMemorySatDataset(X=self.X[indices_train],y=self.y[indices_train],data_file_path=self.data_file_path,mask_path=self.mask_path,transform=train_transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_train)
        test_set = InMemorySatDataset(X=self.X[indices_test],y=self.y[indices_test],data_file_path=self.data_file_path,mask_path=self.mask_path,transform=test_transform,patch_size=self.patch_size,overlap=self.overlap,
                                        padding=self.padding,pad_value=self.pad_value,indices=indices_test)

        return train_set,test_set

    def get_config(self):
        return {"patch_size":self.patch_size,"n_channels":self.n_channels,"overlap":self.overlap,"padding":self.padding,"pad_value":self.pad_value,"size":len(self)}

    def export_patches(self,save_dir,figsize=(10,5),alpha=0.6,archived=False,max_n=None):
        import string,random,shutil
        import tarfile

        if max_n is None:
            max_n = np.inf
            filter_idxs = None
        else:
            filter_idxs = np.random.choice(np.arange(len(self)),size=max_n)

        if archived:
            org_save_dir = save_dir
            rnd_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            save_dir = os.path.join("/tmp",rnd_string)

        for fname, idxs in self.file_mapping.items():

            i = torch.arange(*idxs)
            if filter_idxs is not None:
                i = np.intersect1d(i,filter_idxs)

            imgs = torch.from_numpy(self.X[i])
            masks = torch.from_numpy(self.y[i])
            save_ground_truth_plots(imgs,masks,save_dir,i,fname,figsize=figsize,alpha=alpha)


        if archived:
            fname = os.path.join(save_dir,"imgs.tgz")
            with tarfile.open(fname, "w:gz") as tar:
                tar.add(save_dir, arcname=os.path.basename(save_dir))
            dst = os.path.join(org_save_dir,"imgs.tgz")

            shutil.move(fname,dst)
            shutil.rmtree(save_dir)
        return

    def _create_mask_patches(self,mask_areas):
        patches_masks = []
        mapping = {}
        start_idx = 0
        for fname,ma in mask_areas.items():
            label_area_arr = ma.read(1)#, #window = from_bounds(*bounds, la.transform))
            if self.padding:
                label_area_arr,_  = pad_image_even(label_area_arr,self.patch_size,self.overlap,dim=2,border_val=self.pad_value)
            patches = patchify(label_area_arr,(self.patch_size[0], self.patch_size[1]), 
                                    step=self.patch_size[0]-self.overlap)
            reshaped_patches = np.reshape(patches, 
                                        (patches.shape[0]*patches.shape[1], 
                                        patches.shape[2], patches.shape[3])) 
                                        # = (#patches, 256, 256)
            reshaped_patches = reshaped_patches.astype("uint8")
            patches_masks.append(reshaped_patches)
            mapping[fname] = [start_idx,start_idx+len(reshaped_patches)]
            start_idx += len(reshaped_patches)
        return patches_masks,mapping
        

    def _create_data_patches(self,satellite_img,mask_areas):
        idx = 0
        for ma in mask_areas.values():
            # get coordinates 
            geom = geometry.box(*ma.bounds)

            satellite_area_arr, sat_patch_arr_transform = rasterio.mask.mask(satellite_img,[geom]
                                                                        , crop=True)
            if satellite_area_arr.shape[1:] != ma.shape: #in case we have pixel inaccuracies and the cropped version has cropped one pixel more than the mask
                satellite_area_arr = satellite_area_arr[:,:ma.shape[0],:ma.shape[1]]

            #bounds =  la.bounds
            #satellite_area_arr = satellite_img.read(None, window = from_bounds(*bounds, satellite_img.transform))
            if self.padding: 
                satellite_area_arr,_ = pad_image_even(satellite_area_arr,self.patch_size,self.overlap)
            patches = patchify(satellite_area_arr, 
                                    (self.n_channels, self.patch_size[0], self.patch_size[1]), 
                                    step=self.patch_size[0]-self.overlap)[0]
            reshaped_patches = np.reshape(patches, 
                                            (patches.shape[0]*patches.shape[1], 
                                            patches.shape[2], patches.shape[3], patches.shape[3]))
                                            # = (#patches, 3, 256, 256)
            end_idx = idx + len(reshaped_patches)
            self.X[idx:end_idx] = reshaped_patches
            idx += len(reshaped_patches)
        # return patches_satellite






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



