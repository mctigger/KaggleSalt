from pydoc import locate

import numpy as np
from tqdm import tqdm

import utils


def ensemble_mean(p):
    return np.mean(p, axis=0)


def ensemble_vote(p):
    return np.mean((p > 0.5).reshape(-1, *p.shape[2:]), axis=0)


def ensemble_mean_mean(p):
    return np.mean((p).reshape(-1, *p.shape[2:]), axis=0)


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

ensemble_name = 'ensemble-top-12-test-weighted'
weights = [0.20825407, 0.75518098, 0.73788061, 0.12476653, 0.12492937, 0.77131858, 0.42370022, 0.62843662, 0.08514515, 0.61333554, 0.57577217, 0.80513217]

test_predictions_experiment = []

for name in experiments:
    test_predictions_split = []
    for i in range(5):
        test_predictions = utils.TestPredictions('{}-split_{}'.format(name, i))
        test_predictions_split.append(test_predictions.load_raw())
    test_predictions_experiment.append(test_predictions_split)

test_samples = utils.get_test_samples()

predictions_mean = []
for id in tqdm(test_samples, ascii=True):
    # p = n_models x h x w
    p = []
    for i, test_predictions_split in enumerate(test_predictions_experiment):
        print(i)
        test_predictions_split = np.stack([predictions[id]*w for predictions, w in zip(test_predictions_split, weights)], axis=0)
        p.append(test_predictions_split)

    p = np.concatenate(p, axis=0)

    prediction_ensemble = ensemble_mean(p)
    predictions_mean.append((prediction_ensemble, id))

# Save ensembled predictions (for example for pseudo-labeling)
ensemble_predictions = utils.TestPredictions(ensemble_name)
ensemble_predictions.add_predictions(predictions_mean)
ensemble_predictions.save()

# Threshold for submission
predictions_thresholded = [p > 0.5 for p, id in predictions_mean]

submission = utils.Submission(ensemble_name)
submission.add_samples(predictions_thresholded, test_samples)
submission.save()

