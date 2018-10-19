import numpy as np
from tqdm import tqdm
import torch

from scipy import optimize
from ela import generator
from hyperopt import fmin, tpe, hp

import utils
import metrics
import datasets


experiments = [
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_seresnext101_ndadam_scse_block_pseudo_labels'
]

output = False

test_predictions_experiment = []

for name in experiments:
    test_predictions = utils.TestPredictions('{}'.format(name), mode='val')
    test_predictions_experiment.append(test_predictions.load_raw())

train_samples = utils.get_train_samples()


transforms = generator.TransformationsGenerator([])
dataset = datasets.AnalysisDataset(train_samples, './data/train', transforms, utils.TestPredictions('{}'.format(name), mode='val').load())

val = utils.get_train_samples()
predictions = []
masks = []

for id in tqdm(val, leave=False, ascii=True):
    _, mask, _ = dataset.get_by_id(id)
    prediction = torch.stack([torch.mean(torch.sigmoid(torch.FloatTensor(predictions[id])), dim=0) for predictions in
                              test_predictions_experiment], dim=0)
    mask = torch.FloatTensor(mask)

    predictions.append(prediction)
    masks.append(mask)

predictions_in = torch.stack(predictions, dim=0).cuda()
masks = torch.stack(masks, dim=0).cuda()


def run_evaluation(weights):
    weights_sum = np.sum(weights)

    predictions = predictions_in * (torch.FloatTensor(weights) / float(weights_sum)).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(2).expand_as(predictions_in)
    predictions = torch.sum(predictions, dim=1)
    predictions = (predictions > 0.5).float()

    map = metrics.mAP(predictions, masks)

    print(map)

    return 1 - map


mAP_mean = run_evaluation([1 / len(experiments)]*len(experiments))
print('Uniform weight mAP: ', mAP_mean)

best = fmin(
    fn=run_evaluation(),
    space= hp.choice('classifier_type', [

    ]),
    algo=tpe.suggest,
    max_evals=100
)

print(best)