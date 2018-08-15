import torch
from torch.nn import functional as F


def identity(img):
    return img


def flip_forward(img):
    return torch.flip(img, [3])


def flip_backward(mask):
    return torch.flip(mask, [3])


def resize_forward(img, size):
    return F.interpolate(img, size=size)


def resize_backward(img, size):
    return F.interpolate(img, size=size)