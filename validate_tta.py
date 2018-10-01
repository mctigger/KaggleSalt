import argparse
from pydoc import locate

from torch.nn import DataParallel

import utils
import tta

parser = argparse.ArgumentParser(description='Validate a experiment with different test time augmentations.')
parser.add_argument('name', help='Use one of the experiment names here excluding the .py ending.')
parser.add_argument('--top-n', default=10, type=int)
args = parser.parse_args()

name = args.name

experiment_logger = utils.ExperimentLogger(name, mode='val')

tta = [
    tta.Pipeline([tta.Pad((13, 14, 13, 14))]),
    tta.Pipeline([tta.Pad((13, 14, 13, 14)), tta.Flip()])
]

for i, (samples_train, samples_val) in enumerate(utils.mask_stratified_k_fold()):
    # Get the model architecture
    Model = locate('experiments.' + name + '.Model')
    model = Model(name, i)

    # Load the best performing checkpoint
    model.load()

    # Set model tta
    model.tta = tta

    # Validate
    stats = model.validate(DataParallel(model.net).cuda(), samples_val, -1)
    experiment_logger.set_split(i, stats)

experiment_logger.print()
experiment_logger.save()