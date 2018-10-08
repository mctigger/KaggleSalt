import os
import math
import csv
from multiprocessing import Pool

import torch
import numpy as np
import pandas as pd
import scipy.misc
from sklearn.model_selection import KFold
from skimage.io import imread
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
from torch.distributions.beta import Beta
from tqdm import tqdm

import meters
import metrics


pd.options.display.float_format = '{:,.4f}'.format


def get_train_samples():
    return np.array([path[:-4] for path in os.listdir('./data/train/images')])


def get_aux_samples():
    return np.array([path[:-4] for path in os.listdir('./data/auxiliary_small/images')])


def get_mosaic_pairs():
    return np.array([(path[:-4].split('_')[2], path[:-4].split('_')[3]) for path in os.listdir('./data/mosaic_pairs/images')])


def get_mosaic_samples():
    return np.array([path[:-4] for path in os.listdir('./data/mosaic/images')])


def get_test_samples():
    return np.array([path[:-4] for path in os.listdir('./data/test/images')])


def get_train_samples_sorted():
    path_folds = './cv-5fold-splits.pth'
    if os.path.exists(path_folds):
        print("Loading existing cv-split!")
        return torch.load(path_folds)

    masks_sum = []

    for id in tqdm(get_train_samples(), ascii=True, desc='Sorting train samples'):
        path = os.path.join('./data/train/masks', id + '.png')
        img = imread(path)
        masks_sum.append((id, np.sum(img)))

    sorted_masks = sorted(masks_sum, key=lambda x: x[1])
    ids = [id for id, mask_sum in sorted_masks]

    torch.save(ids, path_folds)

    return ids


def k_fold():
    rs = KFold(n_splits=5, shuffle=True, random_state=0)
    samples = get_train_samples()

    for i, (train_index, val_index) in enumerate(rs.split(samples)):
        yield samples[train_index], samples[val_index]


def test_k_fold():
    rs = KFold(n_splits=5, shuffle=True, random_state=0)
    samples = get_test_samples()

    for i, (train_index, val_index) in enumerate(rs.split(samples)):
        yield samples[val_index]


def mask_stratified_k_fold(n_splits=5):
    sorted_samples = np.array(get_train_samples_sorted())

    for i in range(n_splits):
        val = sorted_samples[i::n_splits]
        train = list(set(sorted_samples) - set(val))

        yield train, val


def imshow(image_tensor):
    scipy.misc.imshow(image_tensor.numpy())


def run_length_encode(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


class Submission:
    def __init__(self, name):
        self.name = name
        self.samples = {}
        self.pool = Pool()

    def add_sample(self, mask, id):
        self.samples[id] = run_length_encode(mask)

    def add_samples(self, masks, ids):
        for rle, id in zip(self.pool.map(run_length_encode, masks), ids):
            self.samples[id] = rle

    def save(self):
        with open(os.path.join('./submissions', self.name), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['id', 'rle_mask'])
            writer.writerow({'id': 'id', 'rle_mask': 'rle_mask'})

            for id, rle in self.samples.items():
                writer.writerow({'id': id, 'rle_mask': rle})


class TestPredictions:
    def __init__(self, name, mode='test', dir='./predictions'):
        self.name = name
        self.mode = mode
        self.predictions = {}
        self.dir = dir

    def add_sample(self, prediction, id):
        self.predictions[id] = prediction

    def add_samples(self, masks, ids):
        for prediction, id in zip(masks, ids):
            self.predictions[id] = prediction

    def add_predictions(self, predictions):
        for prediction, id in predictions:
            self.predictions[id] = prediction

    def save(self):
        np.savez(os.path.join(self.dir, self.mode, self.name), **self.predictions)

    def load_raw(self):
        predictions = np.load(os.path.join(self.dir, self.mode, self.name) + '.npz')
        return predictions

    def load(self):
        predictions = np.load(os.path.join(self.dir, self.mode, self.name) + '.npz')
        self.predictions = {id: predictions[id] for id in tqdm(predictions, leave=False, ascii=True, desc='Loading predictions for {}'.format(self.name))}
        return self.predictions


class OptimalThreshold:
    def __init__(self):
        self.meter = meters.AverageMeter()

    def update(self, predictions, targets):
        optimal_map_score = 0
        optimal_t = 0
        for i in range(1, 10):
            t = 1 / i
            map_score = metrics.mAP(predictions > t, targets.byte())

            if map_score > optimal_map_score:
                optimal_map_score = map_score
                optimal_t = t

        self.meter.add('threshold', optimal_t)

        return optimal_map_score, optimal_t


class HardNegativeMiningLoader:
    def __init__(self):
        super(HardNegativeMiningLoader, self).__init__()
        self.min_loss = math.inf
        self.images = None
        self.masks = None

    def update(self, images, masks, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.images = images
            self.masks = masks

    def get(self):
        return self.images, self.masks


class EpochLogger:
    def __init__(self, name):
        self.name = name
        self.stats = []

    def add_epoch(self, stats):
        self.stats.append(stats)

    def save(self):
        df = pd.DataFrame(self.stats)
        df.to_csv(
            os.path.join('./logs', 'epochs', self.name),
            index_label='epoch',
            float_format='%.4f',
            sep=' '
        )


class ExperimentLogger:
    def __init__(self, name, mode='train'):
        self.name = name
        self.stats = {}
        self.mode = mode

    def set_split(self, i, stats):
        self.stats[i] = stats

    def create_df(self):
        path_csv = os.path.join('./logs', 'experiments')

        average_meter = meters.AverageMeter()
        for i, stats in self.stats.items():
            for k, v in stats.items():
                average_meter.add(k, v)

        df = pd.DataFrame(average_meter.get_all(), index=[self.name])
        df = df.rename_axis("experiment")
        if os.path.exists(path_csv):
            old_df = pd.read_csv(path_csv, delim_whitespace=True, index_col=0)
            if self.mode == 'val':
                old_df = old_df[['val_iou', 'val_mAP']]

            df = pd.concat([df, old_df])

            df = df[~df.index.duplicated(keep='first')]

        df = df.sort_values(by='val_mAP', ascending=False)

        return df

    def print(self):
        df = self.create_df()
        print(df)

    def save(self):
        path_csv = os.path.join('./logs', 'experiments')
        df = self.create_df()
        df.to_csv(
            path_csv,
            index_label='experiment',
            float_format='%.4f',
            sep=' '
        )


class DictLR(_LRScheduler):
    def __init__(self, optimizer, steps):
        self.steps = steps
        super(DictLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        learning_rate = None

        for e, lr in self.steps.items():
            if e <= epoch:
                learning_rate = lr

        return [learning_rate for base_lr in self.base_lrs]


class MixUp(nn.Module):
    def __init__(self, alpha):
        super(MixUp, self).__init__()
        self.beta = Beta(alpha, alpha)

    def forward(self, x, y):
        idx_1 = torch.randperm(x.size(0))
        idx_2 = torch.randperm(x.size(0))

        lam = self.beta.sample_n(x.size(0))
        lam_x = lam.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(x)
        lam_y = lam.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(y)
        x = lam_x * x[idx_1] + (1 - lam_x) * x[idx_2]
        y = lam_y * y[idx_1] + (1 - lam_y) * y[idx_2]

        return x, y


def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration/(2  * stepsize))
    x = np.abs(iteration/stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
    return lr


def get_triangular_lr_2(iteration, stepsize, base_lr, max_lr):
    cycle = np.floor(1 + iteration / (2 * stepsize))
    x = np.abs(iteration / stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / float(1.5 ** (cycle - 1))
    return lr


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, stepsize, steps):
        self.stepsize = stepsize
        self.steps = steps
        super(CyclicLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch

        base_lr, max_lr = 0, 0
        for e, (b_lr, m_lr) in self.steps.items():
            if e <= epoch:
                base_lr = b_lr
                max_lr = m_lr

        learning_rate = get_triangular_lr(epoch, self.stepsize, base_lr, max_lr)

        return [learning_rate for base_lr in self.base_lrs]


class MultCyclicLR(_LRScheduler):
    def __init__(self, optimizer, stepsize, steps):
        self.stepsize = stepsize
        self.steps = steps
        super(MultCyclicLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch

        base_lr, max_lr = 0, 0
        for e, (b_lr, m_lr) in self.steps.items():
            if e <= epoch:
                base_lr = b_lr
                max_lr = m_lr

        factor = get_triangular_lr(epoch, self.stepsize, base_lr, max_lr)

        return [base_lr*factor for base_lr in self.base_lrs]


class CyclicLR2(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, stepsize):
        self.stepsize = stepsize
        self.base_lr = base_lr
        self.max_lr = max_lr
        super(CyclicLR2, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        learning_rate = get_triangular_lr_2(epoch, self.stepsize, self.base_lr, self.max_lr)

        return [learning_rate for base_lr in self.base_lrs]


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_epoch, power=0.9, steps=None):
        self.max_epoch = max_epoch
        self.power = power
        self.steps = steps

        super(PolyLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch

        base_lr = 0
        epoch_step = 0
        for e, b_lr in self.steps.items():
            if e <= epoch:
                base_lr = b_lr
                epoch_step = e

        return [base_lr * (1 - (epoch - epoch_step) / self.max_epoch) ** self.power for _ in self.base_lrs]


transformations_options = {
    'mode': 'reflect',
    'cval': 0,
    'order': 1,
    'clip': False,
    'preserve_range': True
}


def create_distance_tensor(size):
    horizontal = torch.cat([torch.arange(size // 2, 0, -1), torch.arange(1, size // 2 + 2)]) \
        .unsqueeze(0) \
        .expand(size, size) \
        .float()

    vertical = torch.cat([torch.arange(size // 2, 0, -1), torch.arange(1, size // 2 + 2)]) \
        .unsqueeze(1) \
        .expand(size, size) \
        .float()

    normalized_distance = vertical + horizontal
    normalized_distance = torch.exp(normalized_distance / 100 + 0.5)

    return normalized_distance
