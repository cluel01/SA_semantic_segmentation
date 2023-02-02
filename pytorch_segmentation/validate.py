import torch
import pandas as pd
import numpy as np
from .utils.plotting import save_prediction_plots
from .evaluate import evaluate

def validate(model,dl,save_dir,device,n_images=100,deeplab=False,patch_size=[3,256,256],filter_true_empty=True,metric="iou",mode="best"):
    running_score = pd.DataFrame([{"acc":0.0,"iou":0.0,"dice":0.0}])
    image_tens = torch.empty([n_images]+patch_size).byte()
    gt_tens = torch.empty([n_images]+patch_size[1:]).byte()
    pred_tens = torch.empty([n_images]+patch_size[1:]).byte()
    if mode == "best":
        total_scores = pd.DataFrame(np.zeros(n_images,dtype=np.float64),columns=[metric])
    elif mode == "worst":
        total_scores = pd.DataFrame(np.ones(n_images,dtype=np.float64),columns=[metric])
    idx_tens = torch.empty(n_images).long()

    n = 0
    model.eval()
    for batch in dl:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
  
        with torch.no_grad():
            outputs = model(x)
            if deeplab:
                outputs = outputs["out"]

        # stats - whatever is the phase
        with torch.no_grad():
            y_pred  = outputs.argmax(dim=1)
            #score = evaluate(outputs.cpu(), y.cpu())
            score,n_images = evaluate(y_pred, y,reduction=False,aggregate=False,filter_true_empty=filter_true_empty) #on GPU

        idxs = score.index.values
        for i in idxs:
            update = False
            s = score.loc[i][metric]
            if mode == "best":
                if s > total_scores.min().iloc[0]:
                    old_idx = total_scores[metric].argmin()
                    update = True
            elif mode == "worst":
                if s < total_scores.max().iloc[0]:
                    old_idx = total_scores[metric].argmax()
                    update = True

            if update:
                #Update
                image_tens[old_idx] = (x[i].cpu() * 255).byte()#.permute(1,2,0)
                gt_tens[old_idx] = y[i].cpu().byte()
                pred_tens[old_idx] =  y_pred[i].cpu().byte()
                total_scores[metric].loc[old_idx] = s.astype("float64").item()
                idx_tens[old_idx] = i
                
        running_score  += score.mean()*len(idxs)
        n += len(idxs)

    if mode == "best":
        sort_idxs = np.argsort(total_scores[metric].values)[::-1].copy()
    elif mode == "worst":
        sort_idxs = np.argsort(total_scores[metric].values)

    save_prediction_plots(image_tens[sort_idxs],gt_tens[sort_idxs],pred_tens[sort_idxs],
                            sort_idxs,total_scores.reindex(sort_idxs),save_dir)

    epoch_score = (running_score / n).iloc[0]
    print(f"Scores: {epoch_score.to_string(dtype=False)}")
    return epoch_score