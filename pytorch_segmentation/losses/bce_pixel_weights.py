import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyWithPixelWeights(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        # input = input.flatten()
        # target = target.flatten()

        # input = input[target != 255]
        # target = target[target != 255]

        ce = F.cross_entropy(input, target.clip(0,1),
                                                  None,
                                                  reduction='none')

        weight_arr = ce.clone()
        weight_arr[:] = 1
        # for i in range(len(self.weights)):
        #     weight_arr[target == i] += self.weights[i]

        weight_arr[target == 1] += self.weights[0]
        weight_arr[target == 2] += self.weights[1]

        ce = ce * weight_arr
        return torch.mean(ce)