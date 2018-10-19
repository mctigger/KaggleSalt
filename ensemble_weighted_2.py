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
    'nopoolrefinenet_dpn98_dual_hypercolumn_poly_lr_aux_data_pseudo_labels_ensemble',
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


ensemble_name = 'ensemble-top-12-test-weighted-2'

test_predictions_experiment = []

for name in experiments:
    test_predictions = utils.TestPredictions('{}'.format(name), mode='test')
    test_predictions_experiment.append(test_predictions.load_raw())

samples_test = utils.get_test_samples()


transforms = generator.TransformationsGenerator([])
predictions = []

for id in tqdm(samples_test, leave=False, ascii=True):
    prediction = torch.stack([torch.mean(torch.sigmoid(torch.FloatTensor(predictions[id])), dim=0) for predictions in
                              test_predictions_experiment], dim=0)

    predictions.append(prediction)

predictions_in = torch.stack(predictions, dim=0).cuda()


def run(weights):
    weights_sum = np.sum(weights)

    predictions = predictions_in * (torch.FloatTensor(weights) / float(weights_sum)).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(2).expand_as(predictions_in)
    predictions = torch.sum(predictions, dim=1)
    predictions = (predictions > 0.5).float()

    test_predictions = utils.TestPredictions(ensemble_name, mode='test')
    test_predictions.add_predictions(zip(predictions.cpu().numpy(), samples_test))
    test_predictions.save()


weights = [0.20825407, 0.75518098, 0.73788061, 0.12476653, 0.12492937, 0.77131858, 0.42370022, 0.62843662, 0.08514515, 0.61333554, 0.57577217, 0.80513217]
run(weights)