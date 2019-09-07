import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math


class WeightedBCELoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightedBCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        # _assert_no_grad(target)
        target = target.float()
        # input = input[0][0]
        beta = 1 - torch.mean(target)
        # input = F.softmax(input, dim=1)
        input = input[:, 0, :, :]
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = 1 - beta + (2 * beta - 1) * target

        return F.binary_cross_entropy(input, target, weights, reduction='mean')


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        target = target.float()
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class AllLoss(nn.Module):
    def __init__(self):
        super(AllLoss, self).__init__()
        self.dice = DiceLoss()
        self.BCE = WeightedBCELoss()

    def forward(self, input, target):
        loss1 = self.dice(input, target)
        loss2 = self.BCE(input, target)

        return loss1 + 10 * loss2


myloss = {
    'weighted_bce': WeightedBCELoss,
    'dice': DiceLoss,
    'all': AllLoss,
}