from os.path import join
import pickle

from pydoc import locate

import numpy as np
from tqdm import tqdm

from skimage.io import imread
from skimage import img_as_float
from ela import generator

import utils


def ensemble_mean(p, threshold=0.5):
    return np.mean(p, axis=0) > threshold


def ensemble_vote(p):
    return np.mean((p > 0.5).reshape(-1, *p.shape[2:]), axis=0)


def ensemble_mean_mean(p):
    return np.mean((p).reshape(-1, *p.shape[2:]), axis=0)


def strip_nan(e):
    if e != e:
        return None

    return e[:-4]

# LIST OF EXPERIMENTS TO INCLUDE
experiments = [
    'nopoolrefinenet_seresnet50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
]

# NAME OF SUBMISSION
ensemble_name = 'nopoolrefinenet_seresnet50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels + post-processing'

test_predictions_experiment = []

for name in experiments:
    test_predictions_split = []
    n_splits = locate('experiments.' + name + '.n_splits')
    for i in range(0, 5):
        test_predictions = utils.TestPredictions('{}-split_{}'.format(name, i))
        test_predictions_split.append(test_predictions.load_raw())
    test_predictions_experiment.append(test_predictions_split)

samples_train = utils.get_train_samples()
samples_test = utils.get_test_samples()


predictions_mean = []
for id in tqdm(samples_test, ascii=True):
    # p = n_models x h x w
    p = []
    for i, test_predictions_split in enumerate(test_predictions_experiment):
        test_predictions_split = np.stack([predictions[id] for predictions in test_predictions_split], axis=0)
        p.append(test_predictions_split)

    p = np.concatenate(p, axis=0)

    prediction_ensemble = ensemble_mean(p)
    predictions_mean.append((prediction_ensemble, id))

# Save ensembled predictions (for example for pseudo-labeling)
ensemble_predictions = utils.TestPredictions(ensemble_name)
ensemble_predictions.add_predictions(predictions_mean)


def get_mask(id):
    if id in samples_train:
        mask_n = img_as_float(imread(join('./data/train', 'masks', id) + '.png'))

    if id in samples_test:
        mask_n = ensemble_predictions[id].astype(np.float32)

    return mask_n


# PUT CORRECT PATH TO NEIGHBORS FILE HERE
with open('./data/8_neighbours_mosaics.pkl', "rb") as f:
    neighbors = pickle.load(f)
    neighbors = {k[:-4]: [strip_nan(e) for e in v] for k, v in neighbors.items()}

test_postprocessed = {test_id: get_mask(test_id) for test_id in samples_test}
for sample in tqdm(samples_test, ascii=True):
    if sample in neighbors:
        sample_neighbors = neighbors[sample]
        mask = test_postprocessed[sample]

        # HERE IS SOME SAMPLE CODE. PUT YOUR POST_PROCESSING HERE
        n_top, n_left, n_right, n_bottom = sample_neighbors[1], sample_neighbors[3], sample_neighbors[4], \
                                           sample_neighbors[6]
        if n_top and n_left and n_right and n_bottom:
            top = get_mask(n_top)
            left = get_mask(n_left)
            right = get_mask(n_right)
            bottom = get_mask(n_bottom)

            t = 0.5
            if np.mean(np.abs(top[-1, :] - bottom[0, :])) < t and np.mean(top[-1, :] > 0.5) > 0.1 and np.mean(
                    bottom[0, :] > 0.5) > 0.1:
                if n_top in samples_train:
                    mask[:, :] = np.clip(mask[:, :] + top[-1, :], 0, 1)
                if n_bottom in samples_train:
                    mask[:, :] = np.clip(mask[:, :] + bottom[0, :], 0, 1)

            t = 0.2
            if np.mean(np.abs(top[-1, :] - bottom[0, :])) < t and np.mean(top[-1, :] > 0.5) > 0.1 and np.mean(
                    bottom[0, :] > 0.5) > 0.1:
                mask[:, :] = np.clip(mask[:, :] + top[-1, :], 0, 1)
                mask[:, :] = np.clip(mask[:, :] + bottom[0, :], 0, 1)

        # END SAMPLE CODE


predictions = [prediction for id, prediction in test_postprocessed.items()]
predictions = np.stack(predictions, axis=0)
predictions = (predictions > 0.5).astype(np.float32)

# Threshold for submission
submission = utils.Submission(ensemble_name)
submission.add_samples(predictions, samples_test)
submission.save()

