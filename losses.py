import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss
import loss_lovasz


class mAPLoss(nn.Module):
    def __init__(self):
        super(mAPLoss, self).__init__()

        self.lovasz = LovaszWithLogitsLoss()

    def forward(self, prediction, target):
        thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        iou_loss = self.lovasz(prediction, target)
        iou = 1 - iou_loss

        loss = 0
        for t in thresholds:
            loss += torch.max(t - iou * t, iou.new_tensor(0)) * iou_loss

        loss += 5 * torch.max(0.5 - iou * 0.5, iou.new_tensor(0)) * iou_loss

        return loss / 10



class mAPLoss2(nn.Module):
    def __init__(self):
        super(mAPLoss2, self).__init__()

        self.lovasz = SoftDiceWithLogitsLoss()

    def forward(self, prediction, target):
        thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        iou_loss = self.lovasz(prediction, target)
        iou = 1 - iou_loss

        loss = 0
        for t in thresholds:
            loss += torch.max(t - iou * t, iou.new_tensor(0)) * iou_loss

        loss += 5 * torch.max(0.5 - iou * 0.5, iou.new_tensor(0)) * iou_loss

        return loss / 10



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


class LovaszWithLogitsLoss(nn.Module):
    def __init__(self):
        super(LovaszWithLogitsLoss, self).__init__()

    def forward(self, prediction, target):
        return loss_lovasz.lovasz_hinge(prediction, target, per_image=True)


class SmoothLovaszWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SmoothLovaszWithLogitsLoss, self).__init__()

    def forward(self, prediction, target):
        return loss_lovasz.lovasz_hinge_smooth(prediction, target, per_image=True)


class ELULovaszWithLogitsLoss(nn.Module):
    def __init__(self, size_average=True):
        super(ELULovaszWithLogitsLoss, self).__init__()

        self.size_average = size_average

    def forward(self, prediction, target):
        return loss_lovasz.lovasz_hinge_elu(prediction, target, per_image=self.size_average)


class LovaszBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight_bce=1, weight_lovasz=1):
        super(LovaszBCEWithLogitsLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_lovasz = weight_lovasz

        self.bce = BCEWithLogitsLoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) * self.weight_bce + loss_lovasz.lovasz_hinge(prediction, target, per_image=True) * self.weight_lovasz


class LovaszFocalWithLogitsLoss(nn.Module):
    def __init__(self, weight_focal=1, weight_lovasz=1):
        super(LovaszFocalWithLogitsLoss, self).__init__()
        self.weight_focal= weight_focal
        self.weight_lovasz = weight_lovasz

        self.focal = FocalLoss2d()

    def forward(self, prediction, target):
        return self.focal(prediction, target) * self.weight_focal + loss_lovasz.lovasz_hinge(prediction, target, per_image=True) * self.weight_lovasz


class ELULovaszBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight_bce=1, weight_lovasz=1):
        super(ELULovaszBCEWithLogitsLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_lovasz = weight_lovasz

        self.bce = BCEWithLogitsLoss()
        self.lovasz = ELULovaszWithLogitsLoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) * self.weight_bce + self.lovasz(prediction, target) * self.weight_lovasz


class ELULovaszFocalWithLogitsLoss(nn.Module):
    def __init__(self, weight_focal=1, weight_lovasz=1, size_average=True):
        super(ELULovaszFocalWithLogitsLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_lovasz = weight_lovasz

        self.focal = FocalLoss2d(size_average)
        self.lovasz = ELULovaszWithLogitsLoss(size_average)

    def forward(self, prediction, target):
        return self.focal(prediction, target) * self.weight_focal + self.lovasz(prediction, target) * self.weight_lovasz


class SmoothLovaszBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SmoothLovaszBCEWithLogitsLoss, self).__init__()

        self.bce = BCEWithLogitsLoss()
        self.lovasz = SmoothLovaszWithLogitsLoss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + self.lovasz(prediction, target)


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
        return self.dice(torch.sigmoid(prediction), target)


class SoftDicePerImageWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SoftDicePerImageWithLogitsLoss, self).__init__()
        self.dice = SoftDicePerImageLoss()

    def forward(self, prediction, target):
        return self.dice(torch.sigmoid(prediction), target)


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

        return torch.mean(a)


class DiceScore(nn.Module):
    def __init__(self, threshold=0.5):
        super(DiceScore, self).__init__()
        self.threshold = threshold

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        predicts = (probs.view(num, -1) > self.threshold).float()
        labels = labels.view(num, -1)
        intersection = (predicts * labels)
        score = 2. * (intersection.sum(1)) / (predicts.sum(1) + labels.sum(1))
        return score.mean()


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = torch.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        batch_loss = - class_weight * (torch.pow((1 - prob), self.gamma)) * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss
