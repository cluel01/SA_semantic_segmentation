import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F_vis
from torch.functional import F
import numpy as np

from ..evaluate import evaluate

def plot_predictions(net,inputs,labels,nimgs=2,figsize=(6,4),deeplab=False,seed=42):
    np.random.seed(seed)
    idxs = np.random.choice(np.arange(len(inputs)),size=nimgs,replace=False)
    
    out = net(inputs)
    if deeplab:
        out = out["out"]
    #out = F.softmax(out,dim=1)
    out = torch.argmax(out,dim=1)

    fig = plt.figure(figsize=figsize,constrained_layout=True)
    fig.patch.set_facecolor('white')
    #fig.suptitle('Figure title')

    subfigs = fig.subfigures(nrows=nimgs, ncols=1)
    for n_row, subfig in enumerate(subfigs):
        i = idxs[n_row]
        mask_tens = out[i].cpu().byte()
        img_tens = (inputs[i]*255).cpu().byte()
        true_mask_tens = labels[i].byte()
        seg_mask_tens = draw_segmentation_masks(img_tens,mask_tens.bool(),alpha=0.6)
        tens = [("Img",img_tens),("Mask",seg_mask_tens),("GT",true_mask_tens),("Pred",mask_tens)]
        
        scores = evaluate(out[i].unsqueeze(0),labels[i].unsqueeze(0),reduction=False)
        str_scores = " ".join([k+": "+"{:.3f}".format(v) for k,v in scores.iloc[0].to_dict().items()])
        subfig.suptitle(f'Img {i}: {str_scores}')

        

        axs = subfig.subplots(nrows=1, ncols=4)
        for col, ax in enumerate(axs):
            ax.plot()
            title,t = tens[col]
            img = F_vis.to_pil_image(t)
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_title(f'{title}')
    return fig