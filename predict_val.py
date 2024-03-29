import argparse
from pydoc import locate

import utils
import settings

parser = argparse.ArgumentParser(description='Predict validation for a experiment.')
parser.add_argument('name', help='Use one of the experiment names here excluding the .py ending.')
args = parser.parse_args()

name = args.name

test_predictions = utils.TestPredictions(name, mode='val')
for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold(5)):
    # Get the model architecture
    Model = locate('experiments.' + name + '.Model')
    model = Model(name, i)

    # Load the best performing checkpoint
    model.load()

    # Predict the test data
    test_predictions.add_predictions(model.test(samples_val, dir_test=settings.train, predict=model.predict_raw))

test_predictions.save()
