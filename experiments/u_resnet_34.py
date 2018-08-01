import os
import math
import pathlib

import torch
from torch.nn import DataParallel, BCELoss
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet
from tqdm import tqdm
from sklearn.model_selection import KFold

from ela import transformations, generator, random

from nets.u_resnet import UResNet, BottleneckUResNet
from metrics import iou, mean_iou, mAP
import datasets
import utils
import meters
import losses

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class Model:
    def __init__(self, name, split):
        self.split = split
        self.path = os.path.join('./checkpoints', name + '-split_{}'.format(i))
        self.net = UResNet(resnet.resnet34(pretrained=True), layers=[3, 4, 6, 3])

    def save(self):
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(self.path, 'model'))

    def load(self):
        state_dict = torch.load(os.path.join(self.path, 'model'))
        self.net.load_state_dict(state_dict)

    def update_pbar(self, masks_predictions, masks_targets, pbar, average_meter, pbar_description):
        average_meter.add('SoftDiceLoss', losses.SoftDiceLoss()(masks_predictions, masks_targets).item())
        average_meter.add('BCELoss', BCELoss()(masks_predictions, masks_targets).item())
        average_meter.add('iou', iou(masks_predictions > 0.5, masks_targets.byte()))
        average_meter.add('mAP', mAP(masks_predictions > 0.5, masks_targets.byte()))

        pbar.set_description(
            pbar_description + ''.join(
                [' {}:{:6.4f}'.format(k, v) for k, v in average_meter.get_all().items()]
            )
        )

        pbar.update()

    def train(self, samples_train, samples_val):
        net = DataParallel(self.net).cuda()
        optimizer = SGD(net.parameters(), lr=1e-2, weight_decay=1e-4, momentum=0.9, nesterov=True)
        lr_scheduler = StepLR(optimizer, 25, 0.1)
        criterion = losses.SoftDiceBCELoss()
        epochs = 100

        transforms_train = generator.TransformationsGenerator([
            random.RandomFlipLr(),
            transformations.Resize((128, 128)),
        ])

        transforms_val = generator.TransformationsGenerator([
            transformations.Resize((128, 128)),
        ])

        train_dataset = datasets.ImageDataset(samples_train, './data/train', transforms_train)
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=10,
            batch_size=64,
            shuffle=True
        )

        val_dataset = datasets.ImageDataset(samples_val, './data/train', transforms_val)
        val_dataloader = DataLoader(
            val_dataset,
            num_workers=10,
            batch_size=128
        )

        best_val_mAP = 0
        best_stats = None
        epoch_logger = utils.EpochLogger(name + '-split_{}'.format(self.split))

        # Training
        for e in range(epochs):
            lr_scheduler.step()

            with tqdm(total=len(train_dataloader), leave=False) as pbar, torch.enable_grad():
                net.train()
                average_meter = meters.AverageMeter()

                for images, masks_targets in train_dataloader:
                    masks_targets = masks_targets.to(gpu)
                    masks_predictions = net(images)

                    loss = criterion(F.adaptive_avg_pool2d(masks_predictions, (101, 101)),
                                     F.adaptive_avg_pool2d(masks_targets, (101, 101)))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    average_meter.add('loss', loss.item())
                    self.update_pbar(masks_predictions, masks_targets, pbar, average_meter, 'Training epoch {}'.format(e))

            with tqdm(total=len(val_dataloader), leave=True) as pbar, torch.no_grad():
                net.eval()
                average_meter = meters.AverageMeter()

                for images, masks_targets in val_dataloader:
                    masks_targets = masks_targets.to(gpu)
                    masks_predictions = net(images)

                    loss = criterion(F.adaptive_avg_pool2d(masks_predictions, (101, 101)),
                                     F.adaptive_avg_pool2d(masks_targets, (101, 101)))

                    average_meter.add('loss', loss.item())
                    self.update_pbar(masks_predictions, masks_targets, pbar, average_meter, 'Validation epoch {}'.format(e))

                epoch_logger.add_epoch(average_meter.get_all())
                if average_meter.get('mAP') > best_val_mAP:
                    best_val_mAP = average_meter.get('mAP')
                    best_stats = average_meter.get_all()
                    self.save()

        # Post training
        epoch_logger.save()

        return best_stats

    def test(self, samples_test):
        net = DataParallel(self.net).cuda()

        transforms = generator.TransformationsGenerator([
            transformations.Resize((128, 128))
        ])

        test_dataset = datasets.ImageDataset(samples_test, './data/test', transforms, test=True)
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=10,
            batch_size=32
        )

        with tqdm(total=len(test_dataloader), leave=True) as pbar, torch.no_grad():
            net.eval()

            for images, ids in test_dataloader:
                images = images.to(gpu)
                masks_predictions = net(images)
                masks_predictions = F.adaptive_avg_pool2d(masks_predictions, (101, 101))

                pbar.set_description('Creating test predictions...')
                pbar.update()

                yield masks_predictions.cpu().squeeze().numpy(), ids


if __name__ == "__main__":
    file_name = os.path.basename(__file__).split('.')[0]
    name = str(file_name)

    rs = KFold(n_splits=5, shuffle=True, random_state=0)
    samples = utils.get_train_samples()
    experiment_logger = utils.ExperimentLogger(name)

    for i, (train_index, val_index) in enumerate(rs.split(samples)):
        model = Model(name, i)
        validation_stats = model.train(samples[train_index], samples[val_index])
        experiment_logger.set_split(i, validation_stats)

        model.load()
        test_predictions = utils.TestPredictions(name + '-split_{}'.format(i))

        samples_test = utils.get_test_samples()
        for predictions, ids in model.test(samples_test):
            test_predictions.add_samples(predictions, ids)

        test_predictions.save()

        """
        submission = utils.Submission(name)

        samples_test = utils.get_test_samples()
        for predictions, ids in model.test(samples_test):
            submission.add_samples(predictions > 0.5, ids)

        submission.save()
        """

    experiment_logger.save()