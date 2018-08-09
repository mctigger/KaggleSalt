from os.path import join

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from skimage.io import imread
from skimage import img_as_float


class ImageDataset(Dataset):
    def __init__(self, samples, path, transforms, test=False):
        self.samples = samples
        self.path = path
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]

        image = imread(join(self.path, 'images', id) + '.png')

        t = next(self.transforms)

        image = t(image)
        image = ToTensor()(image).float()

        if self.test:
            return image, id
        else:
            mask = img_as_float(imread(join(self.path, 'masks', id) + '.png'))
            mask = t(mask)
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

            return image, mask


class SemiSupervisedImageDataset(Dataset):
    def __init__(self, samples, path, transforms, size):
        self.samples = samples
        self.path = path
        self.transforms = transforms
        self.size = size
        self.masks = {}

    def set_masks(self, test_predictions):
        for predictions, ids in test_predictions:
            for p, id in zip(predictions, ids):
                self.masks[id] = p

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        id = self.samples[index]

        image = imread(join(self.path, 'images', id) + '.png')

        t = next(self.transforms)

        image = t(image)
        image = ToTensor()(image).float()

        mask = self.masks[id]
        mask = t(mask)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

        return image, mask


# Imgaug datasets
class ImageAugTrainDataset(Dataset):
    def __init__(self, samples, path, transforms, test=False):
        super(ImageAugTrainDataset, self).__init__()
        self.samples = samples
        self.path = path
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]

        image = imread(join(self.path, 'images', id) + '.png')
        mask = img_as_float(imread(join(self.path, 'masks', id) + '.png'))

        print(image.dtype, mask.dytpe, mask.shape)

        image, mask = self.transforms.augment_images([image, mask])

        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
        image = ToTensor()(image).float()

        return image, mask


class ImageAugTestDataset(Dataset):
    def __init__(self, samples, path, transforms, test=False):
        super(ImageAugTestDataset, self).__init__()
        self.samples = samples
        self.path = path
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        id = self.samples[index]

        image = imread(join(self.path, 'images', id) + '.png')

        image = self.transforms.augment_images([image])[0]
        image = ToTensor()(image).float()

        return image, id