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
    def __init__(self, size):
        self.size = size

    def transform_forward(self, x):
        return F.interpolate(x, size=self.size)

    def transform_backward(self, x):
        return F.interpolate(x, size=self.size)


class Pad(TTA):
    def __init__(self, pad):
        self.pad = pad

    def transform_forward(self, x):
        return F.pad(x, self.pad, mode='reflect')

    def transform_backward(self, x):
        pad = self.pad
        b, c, h, w = x.size()

        return x[:, :, pad[2]:h-pad[3], pad[0]:w-pad[1]]
