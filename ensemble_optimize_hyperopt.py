import numpy as np
from tqdm import tqdm
import torch

from ela import generator
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import utils
import metrics
import datasets


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
    'nopoolrefinenet_seresnet152_dual_hypercolumn_aux_data_poly_lr_pseudo_labels_ensemble',
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
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

    print(map.item())

    return {'loss': 1 - map.item(), 'status': STATUS_OK}


weights_uniform = [1 / len(experiments)]*len(experiments)
mAP_mean = run_evaluation(weights_uniform)
print('Uniform weight mAP: ', mAP_mean)

space_weights = [hp.uniform('w{}'.format(i), 0, 1) for i in range(len(experiments))]
trials = Trials()
best = fmin(
    fn=run_evaluation,
    space=space_weights,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials
)

print(best)
print(list(best.values()))
print(run_evaluation(list(best.values())))