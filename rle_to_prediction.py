import pandas as pd
import numpy as np

from tqdm import tqdm

import utils



def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101).T


for ensemble_i in tqdm(range(5), ascii=True):
    subm = pd.read_csv('./submissions/ensemble-{}'.format(ensemble_i)).fillna('')
    subm['mask'] = subm['rle_mask'].apply(rle_decode)

    predictions = utils.TestPredictions('ensemble-{}'.format(ensemble_i))

    for i in tqdm(range(18000), ascii=True):
        id = subm['id'][i]
        mask = subm['mask'][i]

        predictions.add_sample(mask, id)

    predictions.save()
