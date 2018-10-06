from os.path import join

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from skimage.io import imread
from skimage import img_as_float


class ImageDataset(Dataset):
    def __init__(self, samples, path, transforms, transforms_image=None, test=False, semisupervised_marker=False):
        self.samples = samples
        self.path = path
        self.transforms = transforms
        self.transforms_image = transforms_image
        self.test = test
        self.semisupervised_marker = semisupervised_marker

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]

        image = img_as_float(imread(join(self.path, 'images', id) + '.png'))

        t = next(self.transforms)

        image = t(image)

        if self.transforms_image:
            t_image = next(self.transforms_image)
            image = t_image(image)

        image = ToTensor()(image).float()

        if self.test:
            return image, id

        mask = img_as_float(imread(join(self.path, 'masks', id) + '.png'))
        mask = t(mask)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

        if self.semisupervised_marker:
            return image, mask, 0

        return image, mask


class StackingDataset(Dataset):
    def __init__(self, samples, path, transforms, predictions=[], transforms_image=None, test=False):
        self.samples = samples
        self.path = path
        self.predictions = predictions
        self.transforms = transforms
        self.transforms_image = transforms_image
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]

        model_predictions = [p[id] for p in self.predictions]
        image = np.concatenate(model_predictions, axis=0)
        image = np.moveaxis(image, 0, -1)

        t = next(self.transforms)

        image = t(image)

        if self.transforms_image:
            t_image = next(self.transforms_image)
            image = t_image(image)

        image = ToTensor()(image).float()

        if self.test:
            return image, id
        else:
            mask = img_as_float(imread(join(self.path, 'masks', id) + '.png'))
            mask = t(mask)
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

            return image, mask


class AnalysisDataset(Dataset):
    def __init__(self, samples, path, transforms, predictions):
        self.samples = samples
        self.path = path
        self.predictions = predictions
        self.transforms = transforms

    def get_by_id(self, id):
        image = self.predictions[id]

        t = next(self.transforms)

        image = t(image)

        mask = img_as_float(imread(join(self.path, 'masks', id) + '.png'))
        mask = t(mask)

        return image, mask, id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]

        return self.get_by_id(id)


class ImageDatasetRemoveSmall(Dataset):
    def __init__(self, samples, path, transforms, transforms_image=None, test=False):
        self.samples = samples
        self.path = path
        self.transforms = transforms
        self.transforms_image = transforms_image
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]

        image = img_as_float(imread(join(self.path, 'images', id) + '.png'))

        t = next(self.transforms)

        image = t(image)

        if self.transforms_image:
            t_image = next(self.transforms_image)
            image = t_image(image)

        image = ToTensor()(image).float()

        if self.test:
            return image, id
        else:
            mask = img_as_float(imread(join(self.path, 'masks', id) + '.png'))
            mask = t(mask)
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

            if torch.sum(mask.view(-1)) < 2424795:
                mask = 0

            return image, mask


class SemiSupervisedImageDataset(Dataset):
    def __init__(self, samples, path, transforms, size=None, momentum=0, test_predictions=None, semisupervised_marker=False):
        if size is None:
            size = len(samples)

        self.samples = samples
        self.path = path
        self.transforms = transforms
        self.size = size
        self.momentum = momentum
        self.mask_predictions = {}
        self.semisupervised_marker = semisupervised_marker

        if test_predictions:
            self.mask_predictions = test_predictions

    def set_masks(self, test_predictions):
        for predictions, ids in test_predictions:
            for p, id in zip(predictions, ids):
                if id in self.mask_predictions:
                    self.mask_predictions[id] = (1 - self.momentum) * self.mask_predictions[id] + self.momentum * p
                else:
                    self.mask_predictions[id] = p

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        id = self.samples[index]

        image = img_as_float(imread(join(self.path, 'images', id) + '.png'))

        t = next(self.transforms)

        image = t(image)
        image = ToTensor()(image).float()

        mask = self.mask_predictions[id] > 0.5
        mask = t(mask)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

        if self.semisupervised_marker:
            return image, mask, 1

        return image, mask


class RandomSubsetDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        real_index = np.random.randint(0, len(self.dataset))

        return self.dataset[real_index]


class DataAnalysisDataset(Dataset):
    def __init__(self, samples, path):
        super(DataAnalysisDataset, self).__init__()

        self.samples = samples
        self.path = path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]
        image = img_as_float(imread(join(self.path, 'images', id) + '.png'))
        mask = img_as_float(imread(join(self.path, 'masks', id) + '.png'))

        return image, mask, id


class TestDataAnalysisDataset(Dataset):
    def __init__(self, samples, path):
        super(TestDataAnalysisDataset, self).__init__()

        self.samples = samples
        self.path = path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]
        image = img_as_float(imread(join(self.path, 'images', id) + '.png'))

        return image, id
