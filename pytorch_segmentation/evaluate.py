import torch
import pandas as pd

def evaluate(outputs,y_true,reduction=True):
    if reduction:
        y_pred = outputs.argmax(dim=1)
    else:
        y_pred = outputs

    y_pred = y_pred.reshape(y_pred.size(0),-1)
    y_true = y_true.reshape(y_true.size(0),-1)

    acc =  (torch.sum(y_pred == y_true)/torch.numel(y_pred)).item()
    iou = iou_score(y_pred,y_true).item()
    dice = dice_coeff(y_pred,y_true).item()
    scores =  pd.DataFrame([{"acc":acc,"iou":iou,"dice":dice}])

    return scores


def iou_score(y_pred,target,absent_score=1.0):
    numerator = torch.sum(y_pred * target,dim=1)  # TP
    denominator = torch.sum(y_pred + target,dim=1) - numerator  # 2TP + FP + FN - TP
    iou = (numerator) / (denominator)
    iou[denominator == 0] = absent_score
    return iou.mean()

def dice_coeff(y_pred, target,absent_score=1.0):
    # F1 = TP / (TP + 0.5 (FP + FN)) = 2TP / (2TP + FP + FN)
    numerator = 2 * torch.sum(y_pred * target,dim=1)  # 2TP
    denominator = torch.sum(y_pred + target,dim=1)  # 2TP + FP + FN
    dice = (numerator) / (denominator)
    dice[denominator == 0] = absent_score
    return dice.mean()
