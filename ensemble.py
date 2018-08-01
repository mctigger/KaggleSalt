import numpy as np
from tqdm import tqdm

import utils


name = 'u_resnet_18'

predictions_splits = []
for i in range(4, 5):
    test_predictions = utils.TestPredictions('{}-split_{}'.format(name, i))
    predictions_splits.append(test_predictions.load())

test_samples = utils.get_test_samples()

predictions_mean = []
for id in tqdm(test_samples):
    predictions_id = [predictions[id] for predictions in predictions_splits]
    prediction_mean = np.mean(np.stack(predictions_id, axis=0), axis=0)
    predictions_mean.append(prediction_mean > 0.5)

submission = utils.Submission(name)
submission.add_samples(predictions_mean, test_samples)

submission.save()