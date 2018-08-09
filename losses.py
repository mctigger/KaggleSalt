import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss
import loss_lovasz

class SoftDiceBCELoss(nn.Module):
    def __init__(self):
        super(SoftDiceBCELoss, self).__init__()

        self.dice = SoftDiceLoss()
        self.bce = BCELoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + self.dice(prediction, target)


class SoftDiceBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SoftDiceBCEWithLogitsLoss, self).__init__()

        self.dice = SoftDiceWithLogitsLoss()
        self.bce = BCEWithLogitsLoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + self.dice(prediction, target)


class LovaszBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(LovaszBCEWithLogitsLoss, self).__init__()

        self.bce = BCEWithLogitsLoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + loss_lovasz.lovasz_hinge(prediction, target, per_image=True)


class SoftDicePerImageBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SoftDicePerImageBCEWithLogitsLoss, self).__init__()

        self.dice = SoftDicePerImageWithLogitsLoss()
        self.bce = BCEWithLogitsLoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + self.dice(prediction, target)


class SoftDiceWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SoftDiceWithLogitsLoss, self).__init__()
        self.dice = SoftDiceLoss()

    def forward(self, prediction, target):
        return self.dice(F.sigmoid(prediction), target)


class SoftDicePerImageWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SoftDicePerImageWithLogitsLoss, self).__init__()
        self.dice = SoftDicePerImageLoss()

    def forward(self, prediction, target):
        return self.dice(F.sigmoid(prediction), target)


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, target):
        smooth = 1.0

        iflat = prediction.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class SoftDicePerImageLoss(nn.Module):
    def __init__(self):
        super(SoftDicePerImageLoss, self).__init__()

    def forward(self, prediction, target):
        smooth = 1.0
        batch_size = prediction.size(0)

        iflat = prediction.view(batch_size, -1)
        tflat = target.view(batch_size, -1)
        intersection = (iflat * tflat).sum(dim=1)

        a = 1 - ((2. * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth))

        print(a)

        return torch.mean(a)


class DiceScore(nn.Module):
    def __init__(self, threshold=0.5):
        super(DiceScore, self).__init__()
        self.threshold = threshold

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        predicts = (probs.view(num, -1) > self.threshold).float()
        labels = labels.view(num, -1)
        intersection = (predicts * labels)
        score = 2. * (intersection.sum(1)) / (predicts.sum(1) + labels.sum(1))
        return score.mean()