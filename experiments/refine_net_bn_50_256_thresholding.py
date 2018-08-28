import os
import pathlib

import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.optim import Adam, SGD
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


resize = transformations.Resize((128, 128), **utils.transformations_options)


class Model:
    def __init__(self, name, split):
        self.split = split
        self.path = os.path.join('./checkpoints', name + '-split_{}'.format(i))
        self.net = RefineNet(ResNetBase(
            resnet.resnet50(pretrained=True)),
            num_features=256
        )

        state_dict = torch.load(os.path.join(os.path.join('./checkpoints', 'refine_net_bn_50_256' + '-split_{}'.format(i)), 'model'))
        self.net.load_state_dict(state_dict)

        self.tta = [
            (tta.identity, tta.identity)
        ]

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
        for tta_forward, tta_backward in self.tta:
            masks_predictions = net(tta_forward(images))
            tta_masks.append(tta_backward(masks_predictions))

        tta_masks = torch.stack(tta_masks, dim=0)
        masks_predictions = torch.mean(tta_masks, dim=0)

        return masks_predictions

    def train(self, samples_train, samples_val):
        net = DataParallel(self.net).cuda()

        transforms_val = generator.TransformationsGenerator([
            resize,
        ])

        val_dataset = datasets.ImageDataset(samples_val, './data/train', transforms_val)
        val_dataloader = DataLoader(
            val_dataset,
            num_workers=10,
            batch_size=64
        )


        criterion = losses.LovaszBCEWithLogitsLoss()

        best_val_mAP = 0
        best_stats = None

        epoch_logger = utils.EpochLogger(name + '-split_{}'.format(self.split))

        average_meter_val = meters.AverageMeter()

        with tqdm(total=len(val_dataloader), leave=True) as pbar, torch.no_grad():
            net.eval()

            for images, masks_targets in val_dataloader:
                masks_targets = masks_targets.to(gpu)
                masks_predictions = self.predict(net, images)

                masks_predictions = F.adaptive_avg_pool2d(masks_predictions, (101, 101))
                masks_targets = F.adaptive_avg_pool2d(masks_targets, (101, 101))

                loss = criterion(masks_predictions, masks_targets)

                average_meter_val.add('loss', loss.item())
                self.update_pbar(
                    masks_predictions,
                    masks_targets,
                    pbar,
                    average_meter_val,
                    'Validation epoch {}'.format(0)
                )

        train_stats = {'train_' + k: v for k, v in average_meter_val.get_all().items()}
        val_stats = {'val_' + k: v for k, v in average_meter_val.get_all().items()}
        stats = {**train_stats, **val_stats}

        epoch_logger.add_epoch(stats)
        if average_meter_val.get('mAP') > best_val_mAP:
            best_stats = stats
            self.save()

        return best_stats

    def test(self, samples_test, dir_test='./data/test'):
        net = DataParallel(self.net).cuda()

        transforms = generator.TransformationsGenerator([
            resize
        ])

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
                masks_predictions = (F.adaptive_avg_pool2d(masks_predictions, (101, 101)) > 0.5).float()

                pbar.set_description('Creating test predictions...')
                pbar.update()

                masks_predictions = masks_predictions.cpu().squeeze().numpy()

                for p, id in zip(masks_predictions, ids):
                    yield p, id


file_name = os.path.basename(__file__).split('.')[0]
name = str(file_name)


if __name__ == "__main__":
    experiment_logger = utils.ExperimentLogger(name)

    for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold()):
        model = Model(name, i)
        stats = model.train(samples_train, samples_val)
        experiment_logger.set_split(i, stats)

        # Load the best performing checkpoint
        model.load()

        # Predict the test data
        test_predictions = utils.TestPredictions(name + '-split_{}'.format(i), mode='test')
        test_predictions.add_predictions(model.test(utils.get_test_samples()))
        test_predictions.save()

    experiment_logger.save()