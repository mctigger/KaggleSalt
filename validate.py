import argparse
from pydoc import locate

from torch.nn import DataParallel

import utils

parser = argparse.ArgumentParser(description='Validate a experiment with different test time augmentations.')
parser.add_argument('name', help='Use one of the experiment names here excluding the .py ending.')
args = parser.parse_args()

name = args.name

experiment_logger = utils.ExperimentLogger(name, mode='val')

for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold()):
    # Get the model architecture
    Model = locate('experiments.' + name + '.Model')
    model = Model(name, i)

    # Load the best performing checkpoint
    model.load()

    # Validate
    stats_train = model.validate(DataParallel(model.net).cuda(), samples_train, -1)
    stats_val = model.validate(DataParallel(model.net).cuda(), samples_val, -1)
    stats = {**stats_train, **stats_val}
    experiment_logger.set_split(i, stats)

experiment_logger.print()
experiment_logger.save()