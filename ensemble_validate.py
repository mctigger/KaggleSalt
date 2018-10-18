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
    'nopoolrefinenet_seresnext101_dual_hypercolumn_aux_data_poly_lr_pseudo_labels_ensemble',
    'nopoolrefinenet_dpn98_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_seresnext101_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_senet154_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_dpn107_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels_ensemble',
    'nopoolrefinenet_seresnext50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_seresnext50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels_ensemble',
    'nopoolrefinenet_seresnet152_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
]
output = 'ensemble-top-10-val'

test_predictions_experiment = []

for name in experiments:
    test_predictions = utils.TestPredictions('{}'.format(name), mode='val')
    test_predictions_experiment.append(test_predictions.load_raw())

train_samples = utils.get_train_samples()


transforms = generator.TransformationsGenerator([])
dataset = datasets.AnalysisDataset(train_samples, './data/train', transforms, utils.TestPredictions('{}'.format(name), mode='val').load())

split_map = []
val = utils.get_train_samples()
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

predictions = torch.stack(predictions, dim=0).cuda()
masks = torch.stack(masks, dim=0).cuda()

predictions = torch.sigmoid(predictions)
predictions = torch.mean(predictions, dim=1)

test_predictions = utils.TestPredictions(output, mode='val')
test_predictions.add_predictions(zip(predictions.cpu().numpy(), train_samples))
test_predictions.save()

predictions = (predictions > 0.5).float()

print(predictions.size())

map = metrics.mAP(predictions, masks)
split_map.append(map)

print(np.mean(split_map), split_map)