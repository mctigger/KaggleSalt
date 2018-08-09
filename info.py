import os

import pandas as pd


path_csv = os.path.join('./logs', 'experiments')
df = pd.read_csv(path_csv, sep=' ', index_col=0)

print(df[['train_mAP', 'val_mAP']])