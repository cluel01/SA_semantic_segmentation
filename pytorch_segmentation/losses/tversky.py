from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._one_hot import one_hot
# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha: float, beta: float) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        fps = torch.sum(input_soft * (1. - target_one_hot), dims)
        fns = torch.sum((1. - input_soft) * target_one_hot, dims)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        return torch.mean(1. - tversky_loss)


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,smooth=1, alpha=0.7, beta=0.3, gamma=4/3):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky

######################
# functional interface
######################

def tversky_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        beta: float) -> torch.Tensor:
    r"""Function that computes Tversky loss.

    See :class:`~torchgeometry.losses.TverskyLoss` for details.
    """
    return TverskyLoss(alpha, beta)(input, target)



