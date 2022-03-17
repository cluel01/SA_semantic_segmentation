import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class InMemorySatDataset(Dataset):
    #augmentation based on Albumentations library  https://github.com/albumentations-team/albumentations
    def __init__(self, X,y,transform=None):
        self.X = torch.as_tensor(X).float().contiguous()
        self.y = torch.as_tensor(y).long().contiguous()
        self.transform = transform
        
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

        
# class SatDataset(Dataset):
#     def __init__(self, data_dir,label_file, transform=None):
#         self.labels = pd.read_csv(label_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label