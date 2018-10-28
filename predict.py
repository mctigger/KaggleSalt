import argparse
from pydoc import locate

import utils

parser = argparse.ArgumentParser(description='Validate a experiment with different test time augmentations.')
parser.add_argument('name', help='Use one of the experiment names here excluding the .py ending.')
parser.add_argument('test_set', help='Specify the path to the new test_set')
parser.add_argument('output_dir', help='Specify the path to the output dir for the test-predictions.')
args = parser.parse_args()

name = args.name
test_set = args.test_set
output_dir = args.output_dir

experiment_logger = utils.ExperimentLogger(name, mode='val')

for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold()):
    # Get the model architecture
    Model = locate('experiments.' + name + '.Model')
    model = Model(name, i)

    # Load the best performing checkpoint
    model.load()

    # Predict the test data
    test_predictions = utils.TestPredictions(name + '-split_{}'.format(i), mode=output_dir)
    test_predictions.add_predictions(model.test(utils.get_test_samples(test_set)))
    test_predictions.save()