import numpy as np
from tqdm import tqdm

import utils


name = 'u_resnet_50_pseudo_labeling_3_augmentations'

predictions_splits = []
for i in range(0, 5):
    test_predictions = utils.TestPredictions('{}-split_{}'.format(name, i))
    predictions_splits.append(test_predictions.load())

test_samples = utils.get_test_samples()

predictions_mean = []
for id in tqdm(test_samples):
    predictions_id = [predictions[id] for predictions in predictions_splits]
    prediction_mean = np.mean(np.stack(predictions_id, axis=0), axis=0)
    predictions_mean.append((id, prediction_mean))

# Save ensembled predictions (for example for pseudo-labeling)
ensemble_predictions = utils.TestPredictions(name)

for id, p_mean in predictions_mean:
    ensemble_predictions.add_sample(p_mean, id)

ensemble_predictions.save()

# Threshold for submission
predictions_thresholded = [p > 0.5 for id, p in predictions_mean]

submission = utils.Submission(name)
submission.add_samples(predictions_thresholded, test_samples)

submission.save()
