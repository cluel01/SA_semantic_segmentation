import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F_vis
from torch.functional import F
import numpy as np
import os

from ..evaluate import evaluate

def plot_predictions_subset(net,inputs,labels,nimgs=2,figsize=(6,4),model_class=None,seed=42):
    np.random.seed(seed)
    idxs = np.random.choice(np.arange(len(inputs)),size=nimgs,replace=False)
    
    with torch.no_grad():
        net.eval()
        out = net(inputs)
        if model_class == "deeplab":
            out = out["out"]
        elif model_class == "smp":
            out = out.squeeze().cpu()
            probs = out.sigmoid()
            pred = (probs > 0.5).long()
        else:
            out = out.cpu()
            pred = torch.argmax(out,dim=1)
            probs = F.softmax(out,dim=1)[:,1,:,:] #probs for tree
        


        
    fig = plt.figure(figsize=figsize,constrained_layout=True)
    fig.patch.set_facecolor('white')
    #fig.suptitle('Figure title')

    subfigs = fig.subfigures(nrows=nimgs, ncols=1)
    for n_row, subfig in enumerate(subfigs):
        i = idxs[n_row]
        mask_tens = pred[i].byte()
        if inputs.size(0) > 3:
            img_tens = (inputs[i]*255).cpu().byte()[:3,:,:]
        else:
            img_tens = (inputs[i]*255).cpu().byte()
        true_mask_tens = labels[i].cpu().byte()
        diff_mask_tens = true_mask_tens - mask_tens
        diff_mask_tens[diff_mask_tens == -1] = 2
        #seg_mask_tens = draw_segmentation_masks(img_tens,mask_tens.bool(),alpha=0.6)
        tens = [("Img",img_tens),("GT",true_mask_tens),("SoftPred",probs[i]),("Pred",mask_tens),("Diff",diff_mask_tens),]
        
        scores,_ = evaluate(pred[i].unsqueeze(0),labels[i].unsqueeze(0).cpu(),reduction=False)
        str_scores = " ".join([k+": "+"{:.3f}".format(v) for k,v in scores.to_dict().items()])
        subfig.suptitle(f'Img {i}: {str_scores}')

        

        axs = subfig.subplots(nrows=1, ncols=5)
        for col, ax in enumerate(axs):
            ax.plot()
            title,t = tens[col]
            img = F_vis.to_pil_image(t)
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_title(f'{title}')
    return fig

def save_prediction_plots(x,y_true,y_pred,idxs,scores,save_dir,figsize=(12,10),alpha=0.6):
    os.makedirs(save_dir, exist_ok=True)


    img_tens = x.cpu().byte()
    true_mask_tens = y_true.cpu().byte()
    mask_tens = y_pred.cpu().byte()
    for n,i in enumerate(idxs):
        fig,axes  = plt.subplots(nrows=2,ncols=3,figsize=figsize,constrained_layout=False)
        fig.tight_layout()
        fig.patch.set_facecolor('white')
        fig.delaxes(axes[1][0])

        #fig.suptitle('Figure title')

        filename = os.path.join(save_dir,"img_"+str(n)+".png")
        score = scores.loc[i]
        
        true_mask = draw_segmentation_masks(img_tens[i],true_mask_tens[i].bool(),alpha=alpha)
        pred_mask = draw_segmentation_masks(img_tens[i],mask_tens[i].bool(),alpha=alpha)
        tens = [("Img",img_tens[i]),("GT Mask",true_mask),("Pred Mask",pred_mask),
                ("GT",true_mask_tens[i]),("Pred",mask_tens[i])]
  
        str_scores = " ".join([k+": "+"{:.3f}".format(v) for k,v in score.to_dict().items()])
        fig.suptitle(f'Img {i}: {str_scores}')

        i = 0
        for r,row in enumerate(axes):
            for c,ax in enumerate(row):
                if r == 1 and c == 0:
                    continue
                ax.plot()            
                title,t = tens[i]
                img = F_vis.to_pil_image(t)
                ax.imshow(np.asarray(img))
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                ax.set_title(f'{title}')
                i += 1
        fig.savefig(filename)
        plt.close()

def save_ground_truth_plots(x,y_true,save_dir,idxs,fname,figsize=(10,5),alpha=0.6):
    os.makedirs(save_dir, exist_ok=True)

    img_tens = x.cpu().byte()
    true_mask_tens = y_true.cpu().byte()
    for i,n in enumerate(idxs):
        fig,axes  = plt.subplots(nrows=1,ncols=3,figsize=figsize,constrained_layout=False)
        fig.tight_layout()
        fig.patch.set_facecolor('white')

        filename = os.path.join(save_dir,"img_"+str(n.item())+".png")

        true_mask = draw_segmentation_masks(img_tens[i],true_mask_tens[i].bool(),alpha=alpha)
        tens = [("Img",img_tens[i]),("GT Mask",true_mask),("GT",true_mask_tens[i])]
  
        fig.suptitle(f'Img {n}: {fname}')

        for i,ax in enumerate(axes):
            ax.plot()            
            title,t = tens[i]
            img = F_vis.to_pil_image(t)
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_title(f'{title}')
        fig.savefig(filename)
        plt.close()