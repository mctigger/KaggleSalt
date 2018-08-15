import os
import pathlib

import torch
from torch.nn import DataParallel, BCEWithLogitsLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet
from tqdm import tqdm

from ela import transformations, generator, random

from nets.u_resnet import BottleneckUResNet
from metrics import iou, mAP
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
        self.net = BottleneckUResNet(resnet.resnet50(pretrained=True), layers=[3, 4, 6, 3])

    def save(self):
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(self.path, 'model'))

    def load(self):
        state_dict = torch.load(os.path.join(self.path, 'model'))
        self.net.load_state_dict(state_dict)

    def update_pbar(self, masks_predictions, masks_targets, pbar, average_meter, pbar_description):
        average_meter.add('SoftDiceLoss', losses.SoftDiceWithLogitsLoss()(masks_predictions, masks_targets).item())
        average_meter.add('BCELoss', BCEWithLogitsLoss()(masks_predictions, masks_targets).item())
        average_meter.add('iou', iou(masks_predictions > 0.5, masks_targets.byte()))
        average_meter.add('mAP', mAP(masks_predictions > 0.5, masks_targets.byte()))

        pbar.set_description(
            pbar_description + ''.join(
                [' {}:{:6.4f}'.format(k, v) for k, v in average_meter.get_all().items()]
            )
        )

        pbar.update()

    def train(self, samples_train, samples_val):
        test_samples = utils.get_test_samples()
        net = DataParallel(self.net).cuda()
        optimizer = Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
        cyclic_lr_scheduler = utils.CyclicLR(optimizer, 1e-4, 1e-6, 3)
        lr_scheduler = StepLR(optimizer, 25, 0.1)

        criterion = losses.SoftDiceBCEWithLogitsLoss()
        epochs = 200

        transforms_train = generator.TransformationsGenerator([
            random.RandomFlipLr(),
            random.RandomAffine(
                image_size=101,
                translation=lambda rs: (rs.randint(-20, 20), rs.randint(-20, 20)),
                scale=lambda rs: (rs.uniform(0.9, 1.1), rs.uniform(0.9, 1.1)),
                **utils.transformations_options
            ),
            transformations.Resize((128, 128), **utils.transformations_options),
        ])

        transforms_val = generator.TransformationsGenerator([
            transformations.Resize((128, 128), **utils.transformations_options),
        ])

        train_dataset = datasets.ImageDataset(samples_train, './data/train', transforms_train)
        test_predictions = utils.TestPredictions('refine_next_50_bn_augmentations_lovasz_long_training').load()
        pseudo_dataset = datasets.SemiSupervisedImageDataset(
            test_samples,
            './data/test',
            transforms_train,
            size=int(len(train_dataset) * 1 / 2),
            test_predictions=test_predictions,
            momentum=0.25
        )

        loader = DataLoader(
            ConcatDataset([pseudo_dataset, train_dataset]),
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

            average_meter_train = meters.AverageMeter()
            average_meter_val = meters.AverageMeter()

            if e > 120:
                pseudo_dataset = datasets.SemiSupervisedImageDataset(
                    test_samples,
                    './data/test',
                    transforms_train,
                    size=int(len(train_dataset) * ((201 - e) / 80)),
                    test_predictions=test_predictions,
                    momentum=0.25
                )
                
                pseudo_dataset.set_masks(self.test(test_samples))
                loader = DataLoader(
                    ConcatDataset([pseudo_dataset, train_dataset]),
                    num_workers=10,
                    batch_size=64,
                    shuffle=True
                )

            with tqdm(total=len(loader), leave=False) as pbar, torch.enable_grad():
                net.train()

                for images, masks_targets in loader:
                    masks_targets = masks_targets.to(gpu)
                    masks_predictions = net(images)

                    loss = criterion(F.adaptive_avg_pool2d(masks_predictions, (101, 101)),
                                     F.adaptive_avg_pool2d(masks_targets, (101, 101)))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    average_meter_train.add('loss', loss.item())

                    self.update_pbar(masks_predictions, masks_targets, pbar, average_meter_train, 'Training epoch {}'.format(e))

            with tqdm(total=len(val_dataloader), leave=True) as pbar, torch.no_grad():
                net.eval()

                for images, masks_targets in val_dataloader:
                    masks_targets = masks_targets.to(gpu)
                    masks_predictions = net(images)

                    loss = criterion(F.adaptive_avg_pool2d(masks_predictions, (101, 101)),
                                     F.adaptive_avg_pool2d(masks_targets, (101, 101)))

                    average_meter_val.add('loss', loss.item())
                    self.update_pbar(masks_predictions, masks_targets, pbar, average_meter_val, 'Validation epoch {}'.format(e))

            train_stats = {'train_' + k: v for k, v in average_meter_train.get_all().items()}
            val_stats = {'val_' + k: v for k, v in average_meter_val.get_all().items()}
            stats = {**train_stats, **val_stats}

            epoch_logger.add_epoch(stats)
            if average_meter_val.get('mAP') > best_val_mAP:
                best_val_mAP = average_meter_val.get('mAP')
                best_stats = stats
                self.save()

        # Post training
        epoch_logger.save()

        return best_stats

    def test(self, samples_test):
        net = DataParallel(self.net).cuda()

        transforms = generator.TransformationsGenerator([
            transformations.Resize((128, 128), **utils.transformations_options)
        ])

        test_dataset = datasets.ImageDataset(samples_test, './data/test', transforms, test=True)
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=10,
            batch_size=32
        )

        with tqdm(total=len(test_dataloader), leave=False) as pbar, torch.no_grad():
            net.eval()

            for images, ids in test_dataloader:
                images = images.to(gpu)
                masks_predictions = net(images)
                masks_predictions = F.sigmoid(F.adaptive_avg_pool2d(masks_predictions, (101, 101)))

                pbar.set_description('Creating test predictions...')
                pbar.update()

                yield masks_predictions.cpu().squeeze().numpy(), ids


if __name__ == "__main__":
    file_name = os.path.basename(__file__).split('.')[0]
    name = str(file_name)

    experiment_logger = utils.ExperimentLogger(name)

    for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold()):
        model = Model(name, i)
        validation_stats = model.train(samples_train, samples_val)
        experiment_logger.set_split(i, validation_stats)

        model.load()
        test_predictions = utils.TestPredictions(name + '-split_{}'.format(i))

        samples_test = utils.get_test_samples()
        for predictions, ids in model.test(samples_test):
            test_predictions.add_samples(predictions, ids)

        test_predictions.save()

    experiment_logger.save()
