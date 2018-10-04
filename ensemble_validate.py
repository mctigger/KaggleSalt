import numpy as np
from tqdm import tqdm
import torch

from ela import generator

import utils
import metrics
import datasets


def ensemble_mean(p, threshold=0.5):
    return np.mean(np.mean(p, axis=0) > threshold, axis=0)


def ensemble_vote(p):
    return np.mean((p > 0.5).reshape(-1, *p.shape[2:]), axis=0)


def ensemble_mean_mean(p):
    return np.mean(sigmoid(p.reshape(-1, *p.shape[2:])), axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


experiments = [
    'nopoolrefinenet_seresnext50_ndadam_scse_block_resize',
]

test_predictions_experiment = []

for name in experiments:
    test_predictions = utils.TestPredictions('{}'.format(name), mode='val')
    test_predictions_experiment.append(test_predictions.load_raw())

train_samples = utils.get_train_samples()


transforms = generator.TransformationsGenerator([])
dataset = datasets.AnalysisDataset(train_samples, './data/train', transforms, utils.TestPredictions('{}'.format(name), mode='val').load())

split_map = []
for train, val in utils.mask_stratified_k_fold():
    predictions = []
    masks = []
    with tqdm(total=len(val), leave=False) as pbar:
        for id in val:
            _, mask, _ = dataset.get_by_id(id)
            test_prediction = np.concatenate([predictions[id] for predictions in test_predictions_experiment], axis=0)
            prediction = torch.FloatTensor(test_prediction)
            mask = torch.FloatTensor(mask)

            predictions.append(prediction)
            masks.append(mask)

    predictions = torch.stack(predictions, dim=0)
    masks = torch.stack(masks, dim=0)

    print(predictions.size(), masks.size())

    predictions = torch.sigmoid(predictions)
    predictions = (torch.mean(predictions, dim=1) > 0.5).float()

    map = metrics.mAP(predictions, masks)
    split_map.append(map)

    break

print(np.mean(split_map), split_map)