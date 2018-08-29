import os
import pathlib

import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet
from tqdm import tqdm

from ela import transformations, generator, random

from nets.refine_net_bn import RefineNet, ResNetBase
from metrics import iou, mAP
import datasets
import utils
import meters
import losses
import tta

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class Model:
    def __init__(self, name, split):
        self.name = name
        self.split = split
        self.path = os.path.join('./checkpoints', name + '-split_{}'.format(split))
        self.net = RefineNet(ResNetBase(
            resnet.resnet50(pretrained=True)),
            num_features=128
        )
        self.tta = [
            tta.Pipeline([tta.Pad((13, 14, 13, 14))]),
            tta.Pipeline([tta.Pad((13, 14, 13, 14)), tta.Flip()])
        ]

        self.criterion = losses.LovaszBCEWithLogitsLoss()

    def save(self):
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(self.path, 'model'))

    def load(self):
        state_dict = torch.load(os.path.join(self.path, 'model'))
        self.net.load_state_dict(state_dict)

    def update_pbar(self, masks_predictions, masks_targets, pbar, average_meter, pbar_description):
        average_meter.add('iou', iou(masks_predictions > 0.5, masks_targets.byte()))
        average_meter.add('mAP', mAP(masks_predictions > 0.5, masks_targets.byte()))

        pbar.set_description(
            pbar_description + ''.join(
                [' {}:{:6.4f}'.format(k, v) for k, v in average_meter.get_all().items()]
            )
        )

        pbar.update()

    def predict(self, net, images):
        tta_masks = []
        for tta in self.tta:
            masks_predictions = net(tta.transform_forward(images))
            masks_predictions = torch.sigmoid(tta.transform_backward(masks_predictions))
            tta_masks.append(masks_predictions)

        tta_masks = torch.stack(tta_masks, dim=0)
        masks_predictions = torch.mean(tta_masks, dim=0)

        return masks_predictions

    def fit(self, samples_train, samples_val):
        net = DataParallel(self.net)

        optimizer = Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
        lr_scheduler = utils.CyclicLR(optimizer, 5, {
            0: (1e-4, 1e-6),
            100: (0.5e-4, 1e-6),
            160: (1e-5, 1e-6),
        })

        epochs = 200

        best_val_mAP = 0
        best_stats = None

        # Logs stats for each epoch and saves them as .csv at the end
        epoch_logger = utils.EpochLogger(self.name + '-split_{}'.format(self.split))

        # Training
        for e in range(epochs):
            lr_scheduler.step(e)

            stats_train = self.train(net, samples_train, optimizer, e)
            stats_val = self.validate(net, samples_val, e)

            stats = {**stats_train, **stats_val}

            epoch_logger.add_epoch(stats)
            current_mAP = stats_val['val_mAP']
            if current_mAP > best_val_mAP:
                best_val_mAP = current_mAP
                best_stats = stats
                self.save()

        # Post training
        epoch_logger.save()

        return best_stats

    def train(self, net, samples, optimizer, e):
        transforms = generator.TransformationsGenerator([
            random.RandomFlipLr(),
            random.RandomAffine(
                image_size=101,
                translation=lambda rs: (rs.randint(-20, 20), rs.randint(-20, 20)),
                scale=lambda rs: (rs.uniform(0.85, 1.15), 1),
                **utils.transformations_options
            ),
            transformations.Padding(((13, 14), (13, 14), (0, 0)))
        ])

        dataset = datasets.ImageDataset(samples, './data/train', transforms)
        dataloader = DataLoader(
            dataset,
            num_workers=10,
            batch_size=32,
            shuffle=True
        )

        average_meter_train = meters.AverageMeter()

        with tqdm(total=len(dataloader), leave=False) as pbar, torch.enable_grad():
            net.train()

            for images, masks_targets in dataloader:
                masks_targets = masks_targets.to(gpu)
                masks_predictions = net(images)

                loss = self.criterion(masks_predictions, masks_targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                average_meter_train.add('loss', loss.item())
                self.update_pbar(
                    torch.sigmoid(masks_predictions),
                    masks_targets,
                    pbar,
                    average_meter_train,
                    'Training epoch {}'.format(e)
                )

        train_stats = {'train_' + k: v for k, v in average_meter_train.get_all().items()}
        return train_stats

    def validate(self, net, samples, e):
        transforms = generator.TransformationsGenerator([])
        dataset = datasets.ImageDataset(samples, './data/train', transforms)
        dataloader = DataLoader(
            dataset,
            num_workers=10,
            batch_size=64
        )

        average_meter_val = meters.AverageMeter()

        with tqdm(total=len(dataloader), leave=True) as pbar, torch.no_grad():
            net.eval()

            for images, masks_targets in dataloader:
                masks_targets = masks_targets.to(gpu)
                masks_predictions = self.predict(net, images)

                self.update_pbar(
                    masks_predictions,
                    masks_targets,
                    pbar,
                    average_meter_val,
                    'Validation epoch {}'.format(e)
                )

        val_stats = {'val_' + k: v for k, v in average_meter_val.get_all().items()}
        return val_stats

    def test(self, samples_test, dir_test='./data/test'):
        net = DataParallel(self.net)

        transforms = generator.TransformationsGenerator([])

        test_dataset = datasets.ImageDataset(samples_test, dir_test, transforms, test=True)
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=10,
            batch_size=64
        )

        with tqdm(total=len(test_dataloader), leave=True) as pbar, torch.no_grad():
            net.eval()

            for images, ids in test_dataloader:
                images = images.to(gpu)
                masks_predictions = self.predict(net, images)

                pbar.set_description('Creating test predictions...')
                pbar.update()

                masks_predictions = masks_predictions.cpu().squeeze().numpy()

                for p, id in zip(masks_predictions, ids):
                    yield p, id


def main():
    file_name = os.path.basename(__file__).split('.')[0]
    name = str(file_name)

    experiment_logger = utils.ExperimentLogger(name)

    for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold()):
        model = Model(name, i)
        stats = model.fit(samples_train, samples_val)
        experiment_logger.set_split(i, stats)

        # Load the best performing checkpoint
        model.load()

        # Do a final validation
        model.validate(DataParallel(model.net), samples_val, -1)

        # Predict the test data
        test_predictions = utils.TestPredictions(name + '-split_{}'.format(i), mode='test')
        test_predictions.add_predictions(model.test(utils.get_test_samples()))
        test_predictions.save()

    experiment_logger.save()

if __name__ == "__main__":
    main()