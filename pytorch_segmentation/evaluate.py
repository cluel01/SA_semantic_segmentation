import torch
import pandas as pd

def evaluate(outputs,y_true,reduction=True):
    if reduction:
        y_pred = outputs.argmax(dim=1)
    else:
        y_pred = outputs
    acc =  (torch.sum(y_pred == y_true)/torch.numel(y_pred)).item()
    iou = iou_score(y_pred,y_true).item()
    dice = dice_coeff(y_pred,y_true).item()
    scores =  pd.DataFrame([{"acc":acc,"iou":iou,"dice":dice}])
    return scores


def iou_score(y_pred,target,absent_score=0.0):
    numerator = torch.sum(y_pred * target)  # TP
    denominator = torch.sum(y_pred + target) - numerator  # 2TP + FP + FN - TP
    iou = (numerator) / (denominator)
    iou[denominator == 0] = absent_score
    return iou

def dice_coeff(y_pred, target):
    # F1 = TP / (TP + 0.5 (FP + FN)) = 2TP / (2TP + FP + FN)
    numerator = 2 * torch.sum(y_pred * target)  # 2TP
    denominator = torch.sum(y_pred + target)  # 2TP + FP + FN
    if (denominator == 0):
        return torch.tensor(0.).to(y_pred.device)
    return (numerator) / (denominator)
