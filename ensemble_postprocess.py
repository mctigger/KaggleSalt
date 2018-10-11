import pickle

import numpy as np
from tqdm import tqdm
import torch

from ela import generator

import utils
import metrics
import datasets


experiments = [
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
]

test_predictions_experiment = []

for name in experiments:
    test_predictions = utils.TestPredictions('{}'.format(name), mode='val')
    test_predictions_experiment.append(test_predictions.load_raw())

samples_train = utils.get_train_samples()
transforms = generator.TransformationsGenerator([])
dataset = datasets.AnalysisDataset(samples_train, './data/train', transforms, utils.TestPredictions('{}'.format(name), mode='val').load())

train = {}
for id in tqdm(samples_train):
    _, mask, _ = dataset.get_by_id(id)
    test_prediction = np.concatenate([predictions[id] for predictions in test_predictions_experiment], axis=0)
    prediction = torch.mean(torch.sigmoid(torch.FloatTensor(test_prediction)), dim=0)
    mask = torch.FloatTensor(mask)

    train[id] = (prediction, mask)


def strip_nan(e):
    if e != e:
        return ''

    return e[:-4]


with open('./data/8_neighbours_mosaics.pkl', "rb") as f:
    neighbors = pickle.load(f)
    neighbors = {k[:-4]: [strip_nan(e) for e in v] for k, v in neighbors.items()}

samples_test = utils.get_test_samples()
masks_test = utils.TestPredictions('{}-split_{}'.format(name, 0)).load()


for sample in samples_train:
    if sample in neighbors:
        sample_neighbors = neighbors[sample]

        masks_neighbors = []
        for n in sample_neighbors:
            if n in samples_train:
                mask_n = train[n][0]

            if n in samples_test:
                mask_n = torch.FloatTensor(masks_test[n])

            masks_neighbors.append(mask_n)

        if len(masks_neighbors) == 0:
            continue

        num_neighbors = len(masks_neighbors)
        masks_neighbors = torch.cat(masks_neighbors, dim=0)
        mean_neighbors = torch.mean(masks_neighbors)

        mask_sample, mask = train[sample]
        if mean_neighbors > 0.8 and num_neighbors > 4 and torch.mean(mask_sample) < 0.01:
            mask_sample[:, :] = 1
            train[sample] = mask_sample, mask
            print("AS")


predictions = [prediction for id, (prediction, mask) in train.items()]
predictions = torch.stack(predictions, dim=0).cuda()
masks = [mask for id, (prediction, mask) in train.items()]
masks = torch.stack(masks, dim=0).cuda()

predictions = (predictions > 0.5).float()

map = metrics.mAP(predictions, masks)

print('mAP', map)
