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
    'nopoolrefinenet_dpn92_hypercolumn'
]

test_predictions_experiment = []

for name in experiments:
    test_predictions = utils.TestPredictions('{}'.format(name), mode='val')
    test_predictions_experiment.append(test_predictions.load_raw())

train_samples = utils.get_train_samples()


transforms = generator.TransformationsGenerator([])
dataset = datasets.AnalysisDataset(train_samples, './data/train', transforms, utils.TestPredictions('{}'.format(name), mode='val').load())

maps = []
for image, mask, id in tqdm(dataset):
    test_prediction = np.stack([predictions[id] for predictions in test_predictions_experiment], axis=0)

    prediction = torch.FloatTensor(test_prediction)
    prediction = torch.sigmoid(prediction)
    prediction = torch.mean(torch.mean(prediction, dim=0), dim=0)
    mask = torch.FloatTensor(mask)

    map = metrics.mAP(prediction, mask)
    print(map)
    maps.append(map)

print(torch.mean(torch.stack(maps, dim=0), dim=0))