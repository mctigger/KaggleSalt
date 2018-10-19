import numpy as np

import utils

predictions = utils.TestPredictions('ensemble-top-12-test').load()
predictions_weighted = utils.TestPredictions('ensemble-top-12-test-weighted').load()


diffs = []
for key in predictions_weighted:
    weighted_p = predictions_weighted[key]
    p = predictions[key]

    diff = np.mean(np.abs(weighted_p - p))
    diffs.append(diff)

print(diffs)
print(np.mean(diffs))