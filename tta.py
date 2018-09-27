from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F


class TTA(ABC):
    @abstractmethod
    def transform_forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def transform_backward(self, x):
        raise NotImplementedError


class Identity(TTA):
    def transform_forward(self, x):
        return x

    def transform_backward(self, x):
        return x


class Pipeline(TTA):
    def __init__(self, ttas):
        self.ttas = ttas

    def transform_forward(self, x):
        for tta in self.ttas:
            x = tta.transform_forward(x)

        return x

    def transform_backward(self, x):
        for tta in reversed(self.ttas):
            x = tta.transform_backward(x)

        return x


class Flip(TTA):
    def transform_forward(self, x):
        return torch.flip(x, [3])

    def transform_backward(self, x):
        return torch.flip(x, [3])


class Resize(TTA):
    def __init__(self, size, original_size=101):
        self.size = size
        self.original_size = original_size

    def transform_forward(self, x):
        return F.interpolate(x, size=self.size, mode='bilinear')

    def transform_backward(self, x):
        return F.interpolate(x, size=self.original_size, mode='bilinear')


class Pad(TTA):
    def __init__(self, pad):
        self.pad = pad

    def transform_forward(self, x):
        return F.pad(x, self.pad, mode='reflect')

    def transform_backward(self, x):
        pad = self.pad
        b, c, h, w = x.size()

        return x[:, :, pad[2]:h-pad[3], pad[0]:w-pad[1]]


class Translate(TTA):
    def __init__(self, y=0):
        self.y = y

    def transform_forward(self, x):
        if self.y >= 0:
            top = self.y
            bottom = 0
        else:
            top = 0
            bottom = self.y

        x = x[:, :, top:x.size(2) + bottom, :]
        x = F.pad(x, (0, 0, top, -bottom), mode='reflect')

        return x

    def transform_backward(self, x):

        return x
