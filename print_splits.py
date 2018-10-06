import utils

folds = utils.mask_stratified_k_fold(5)

for train, val in folds:
    print(val[:10])