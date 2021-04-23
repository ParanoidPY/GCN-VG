import torch
import torch.nn as nn
from .loss import BinaryDiceLoss, BCEFocalLoss

class Loss(nn.Module):
    def __init__(self,args):
        super(Loss, self).__init__()

        self.loss_fun = []
        self.weights = []

        if args.loss_function == 'dice':
            self.loss_fun.append(BinaryDiceLoss())
            self.weights.append(1)

        elif args.loss_function == 'bce':
            self.loss_fun.append(nn.BCELoss())
            self.weights.append(1)

        elif args.loss_function == 'bce_focal':
            self.loss_fun.append(BCEFocalLoss())
            self.weights.append(1)

        elif args.loss_function == 'bce+dice':
            self.loss_fun.append(BinaryDiceLoss())
            self.weights.append(1)
            self.loss_fun.append(nn.BCELoss())
            self.weights.append(0.2)

    def forward(self,w,label):
        losses = 0
        for idx,loss_f in enumerate(self.loss_fun):
            losses = losses + float(self.weights[idx]) * loss_f(w,label)

        return losses