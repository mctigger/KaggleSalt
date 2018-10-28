import numpy as np
from tqdm import tqdm

import utils


experiments = [
    'nopoolrefinenet_dpn98_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_seresnext101_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_dpn107_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_seresnext50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_seresnet152_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
]

ensemble_name = 'ensemble'


for i in range(5):
    print('Processing fold {}'.format(i))
    test_predictions_experiment = []
    for name in experiments:
        test_predictions_split = []
        test_predictions = utils.TestPredictions('{}-split_{}'.format(name, i))
        test_predictions_split.append(test_predictions.load_raw())
        test_predictions_experiment.append(test_predictions_split)

    test_samples = utils.get_test_samples()

    predictions_mean = []
    for id in tqdm(test_samples, ascii=True):
        # p = n_models x h x w
        p = []
        for test_predictions_split in test_predictions_experiment:
            test_predictions_split = np.stack([predictions[id] for predictions in test_predictions_split], axis=0)
            p.append(test_predictions_split)

        p = np.concatenate(p, axis=0)

        prediction_ensemble = np.mean(p, axis=0)
        predictions_mean.append((prediction_ensemble, id))

    # Save ensembled predictions (for example for pseudo-labeling)
    ensemble_predictions = utils.TestPredictions(ensemble_name + '-' + str(i))
    ensemble_predictions.add_predictions(predictions_mean)
    ensemble_predictions.save()

    # Threshold for submission
    predictions_thresholded = [p > 0.5 for p, id in predictions_mean]

    submission = utils.Submission(ensemble_name + '-' + str(i))
    submission.add_samples(predictions_thresholded, test_samples)
    submission.save()

