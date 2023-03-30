import torch
import pandas as pd
import numpy as np

def evaluate(outputs,y_true,reduction=True,aggregate=True,filter_true_empty=False):
    if reduction:
        y_pred = outputs.argmax(dim=1)
    else:
        y_pred = outputs

    y_pred = y_pred.reshape(y_pred.size(0),-1)
    y_true = y_true.reshape(y_true.size(0),-1)

    acc =  (torch.sum(y_pred == y_true,axis=1)/y_pred.shape[1]).cpu()
    iou = iou_score(y_pred,y_true).cpu()
    dice = dice_coeff(y_pred,y_true).cpu()
    rmse = rmse_tree_coverage(y_pred,y_true).cpu()

    if filter_true_empty:
        idxs = iou != 1.0
        acc = acc[idxs]
        iou = iou[idxs]
        dice = dice[idxs]
        rmse = rmse[idxs]
        idx = torch.arange(len(y_pred))[idxs].numpy()
    else:
        idx = torch.arange(len(y_pred)).numpy()
    
    if aggregate:
        acc = acc.mean().item()
        iou = iou.mean().item()
        dice = dice.mean().item()
        rmse = rmse.mean().item()
        score = pd.DataFrame([{"acc":acc,"iou":iou,"dice":dice,"rmse_cov":rmse}]).iloc[0]
    else:
        score =  pd.DataFrame({"acc":acc,"iou":iou,"dice":dice,"rmse_cov":rmse},index=idx)

    return score,len(idx)


def iou_score(y_pred,target,absent_score=1.0):
    numerator = torch.sum(y_pred * target,dim=1)  # TP
    denominator = torch.sum(y_pred + target,dim=1) - numerator  # 2TP + FP + FN - TP
    iou = (numerator) / (denominator)
    iou[denominator == 0] = absent_score
    return iou

def dice_coeff(y_pred, target,absent_score=1.0):
    # F1 = TP / (TP + 0.5 (FP + FN)) = 2TP / (2TP + FP + FN)
    numerator = 2 * torch.sum(y_pred * target,dim=1)  # 2TP
    denominator = torch.sum(y_pred + target,dim=1)  # 2TP + FP + FN
    dice = (numerator) / (denominator)
    dice[denominator == 0] = absent_score
    return dice

def rmse_tree_coverage(y_pred,target):
    # Tree coverage = TP / (TP + FN)
    pred_cov = y_pred.sum(1) / y_pred.shape[-1]
    true_cov = target.sum(1) / target.shape[-1]
    rmse = torch.sqrt((pred_cov - true_cov)**2)
    return rmse