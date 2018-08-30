import argparse
from pydoc import locate

from torch.nn import DataParallel

import utils
import tta

parser = argparse.ArgumentParser(description='Predict validation for a experiment.')
parser.add_argument('name', help='Use one of the experiment names here excluding the .py ending.')
parser.add_argument('--top-n', default=10, type=int)
args = parser.parse_args()

name = args.name

tta = [
    tta.Pipeline([tta.Pad((13, 14, 13, 14))]),
    tta.Pipeline([tta.Pad((13, 14, 13, 14)), tta.Flip()]),
    tta.Pipeline([tta.Pad((17, 10, 17, 10))]),
    tta.Pipeline([tta.Pad((10, 17, 17, 10))]),
    tta.Pipeline([tta.Pad((17, 10, 10, 17))]),
    tta.Pipeline([tta.Pad((10, 17, 10, 17))]),
    tta.Pipeline([tta.Pad((17, 10, 17, 10)), tta.Flip()]),
    tta.Pipeline([tta.Pad((10, 17, 17, 10)), tta.Flip()]),
    tta.Pipeline([tta.Pad((17, 10, 10, 17)), tta.Flip()]),
    tta.Pipeline([tta.Pad((10, 17, 10, 17)), tta.Flip()]),
]

test_predictions = utils.TestPredictions(name, mode='val')
for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold()):
    # Get the model architecture
    Model = locate('experiments.' + name + '.Model')
    model = Model(name, i)

    # Load the best performing checkpoint
    model.load()

    # Set model tta
    model.tta = tta

    # Predict the test data
    test_predictions.add_predictions(model.test(samples_val, dir_test='./data/train', predict=model.predict_raw))


test_predictions.save()
