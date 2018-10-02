import argparse
import pandas as pd

import utils

parser = argparse.ArgumentParser(description='Validate a experiment with different test time augmentations.')
parser.add_argument('name', help='Use one of the experiment names here excluding the .py ending.')
args = parser.parse_args()

name = args.name

experiment_logger = utils.ExperimentLogger(name)

for i in range(5):

    df_split = pd.read_csv('logs/epochs/{}-split_{}'.format(name, i), delim_whitespace=True, index_col=0)
    split = df_split.as_matrix(columns=['train_loss', 'train_iou', 'train_mAP', 'val_iou', 'val_mAP'])
    best = split[split[:, 3].argsort()][-1]
    best = {
        'train_loss': best[0],
        'train_iou': best[1],
        'train_mAP': best[2],
        'val_iou': best[3],
        'val_mAP': best[4],
    }
    # Validate
    experiment_logger.set_split(i, best)

experiment_logger.print()
experiment_logger.save()
