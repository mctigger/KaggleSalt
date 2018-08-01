from collections import defaultdict

import numpy as np


class AverageMeter:
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, key, value):
        self.data[key].append(value)

    def get(self, key):
        return np.mean(self.data[key])

    def get_all(self):
        data = {}
        for key, value in self.data.items():
            data[key] = np.mean(value)

        return data