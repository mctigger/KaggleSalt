import numpy as np
from tqdm import tqdm

import utils


experiments = [
    'ensemble'
]

ensemble_name = 'test'


for i in range(5):
    test_predictions_experiment = []
    for name in experiments:
        test_predictions_split = []
        test_predictions = utils.TestPredictions('{}-{}'.format(name, i))
        test_predictions_split.append(test_predictions.load_raw())
        test_predictions_experiment.append(test_predictions_split)

    test_samples = utils.get_test_samples()

    predictions_mean = []
    for id in tqdm(test_samples, ascii=True):
        # p = n_models x h x w
        p = []
        for i, test_predictions_split in enumerate(test_predictions_experiment):
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

