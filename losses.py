import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss


class SoftDiceBCELoss(nn.Module):
    def __init__(self):
        super(SoftDiceBCELoss, self).__init__()

        self.dice = SoftDiceLoss()
        self.bce = BCELoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + self.dice(prediction, target)


# from https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, target):
        smooth = 1.0

        iflat = prediction.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


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