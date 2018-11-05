Hello!

Below you can find a outline of how to reproduce our solution for the TGS Salt Identification competition.
If you run into any trouble with the setup/code or have any questions please contact us at tim.joseph@mctigger.com

# ARCHIVE CONTENTS (TODO)

# HARDWARE: (The following specs were used to create the original solution)
3x:
Ubuntu 16.04 LTS (512 GB boot disk)
16GB RAM
2x NVIDIA GTX 1080TI 11GB

1x
Ubuntu 16.04 LTS (512 GB boot disk)
256GB RAM
4x NVIDIA Tesla P100

# SOFTWARE (python packages are detailed separately in `environment.yml`):
Anaconda 5.3, Python 3.6, NVIDIA drivers 410.66

(CUDA and cuDNN are not listed here since these are part of the Pytorch conda package, see `environment.yml`)

# DATA SETUP 
Download and extract competition data into `./data`.

# DATA PROCESSING
Download small_mask auxilary data into `./data/auxiliary_small` like explained here https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65500 or use their official repository to generate it: https://github.com/neptune-ml/open-solution-salt-identification

# General model building
test prediction = Model predictions for the test data. Contains a predicted mask for each test sample with values between 0 and 1.

submission = File which contains a mask for each test samples with values which are either 0 or 1.

WARNING: Everytime a script generates a new file it will overwrite existing files of the same name without warning. (Usually not a problem though.)

## Creating directories
First create all directories specified in `directory_structure.txt` if not yet existing.

## Training models and creating predictions
Run experiments with `CUDA_VISIBLE_DEVICES=0 python -m experiments.{your_experiment}` (for example to run "nopoolrefinenet_dpn92.py" use `CUDA_VISIBLE_DEVICES=0 python -m experiments.nopoolrefinenet_dpn92`. 
Each experiment contains one neural network architecture which then will be trained.
This will run 5-folds per experiment and create test predictions for each fold. Expect this to run between 5h/fold to 12h/fold.

## Creating an ensemble test prediction and submission
In `ensemble.py` an ensemble_name and all experiments must be specified that should be included into an ensemble. When you run `python ensemble.py` all the test predictions from the different specified experiments are averaged. Two files are created:
1. `{ensemble_name}.npz` in `./predictions/test/` which contains the averaged test predictions.
2. `{ensemble_name}` in `./submissions/` which is a submission file containing the masks from 1., but thresholded at `0.5`.

## Creating fold-ensembles and fold-submissions
For some experiments it is necessary to create some ensembled test predictions which only include predictions from specific folds of experiments.
For this reason you can use `python ensemble_foldwise.py` which essentially works just like `ensemble.py`, but creates test predictions and submissions for each fold.
The submissions are not of further use, but can be submitted to the leaderboard for some insight into their performance.
The created test predictions are saved into `./predictions/test/` as `{ensemble_name}-{i}.npz` where `i` is between 0 and 4.

## Using a different test set
Using a different test set is easy. Just use `predict.py` as follows: `python predict.py {your_experiment} {path_to_test_set_images} {output_dir}`.
Test predictions will be created for the specified experiment by using the images contained in the test set iamges directory and saved into `./predictions/{output_dir}`.
To create a submission just use `ensemble.py` like specified above, but change `input_dir` to the specified `output_dir` in `predict.py`.

# Reproducing our submission
This is a step by step guide on how to reproduce our final leaderboard submission.

## 1. Supervised only ensemble
First step is to create an ensemble of some models that are training in a purely supervised way.
So first train these models as shown before and then create an ensemble.

```
experiments = [
    'nopoolrefinenet_dpn92_hypercolumn',
    'nopoolrefinenet_dpn92',
    'nopoolrefinenet_dpn98',
    'nopoolrefinenet_dpn107',
    'nopoolrefinenet_seresnext101_ndadam_scse_block_padding',
    'nopoolrefinenet_seresnext50_ndadam_scse_block_padding'
]

ensemble_name = 'ensemble'
```

Run `ensemble_foldwise.py` with this config to generate predictions which are then loaded in subsequent models.

## 2. First-round semi-supervised ensemble
Now we just repeat this process, but with new models which are dependent on the ensemble we created before and additionally depend on the small mask dataset.
For the specified models change the parameter `self.test_predictions=...` to `self.test_predictions = utils.TestPredictions('ensemble-{}'.format(split)).load()`
So first train these models and then run `ensemble_foldwise.py` with the given config.

```
experiments = [
    'nopoolrefinenet_dpn98_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_seresnext101_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_dpn107_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_seresnext50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_seresnet152_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
]

ensemble_name = 'ensemble-top-6-test'
```

## 3. Creating the final ensemble
Now we just repeat this process, but with new models which are dependent on the ensemble we created before.
For the specified models change the parameter `self.test_predictions=...` to `self.test_predictions = utils.TestPredictions('ensemble-top-6-test-{}'.format(split)).load()`
So first train these models and then run `ensemble.py` with the given config.

```
experiments = [
    'nopoolrefinenet_seresnext101_dual_hypercolumn_aux_data_poly_lr_pseudo_labels_ensemble',
    'nopoolrefinenet_dpn98_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_dpn98_dual_hypercolumn_poly_lr_aux_data_pseudo_labels_ensemble',
    'nopoolrefinenet_seresnext101_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_senet154_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_dpn107_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels_ensemble',
    'nopoolrefinenet_seresnext50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_seresnext50_dual_hypercolumn_aux_data_poly_lr_pseudo_labels_ensemble',
    'nopoolrefinenet_seresnet152_dual_hypercolumn_aux_data_poly_lr_pseudo_labels',
    'nopoolrefinenet_seresnet152_dual_hypercolumn_aux_data_poly_lr_pseudo_labels_ensemble',
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
]

ensemble_name = 'ensemble-top-12-test'
```

In `./submissions` there should be a file called `ensemble-top-12-test` now which can be submitted to the leaderboard.


## 4. Postprocessing 

To start working on postprocessing you need to build mosaics from train and test images. This can be done using `salt-jigsaw-puzzle.ipynb` notebook - just create `mosaics` folder in the root and run all cells in the notebook. After that you will see saved mosaics in the `mosaics` folder.

The second part is running all cells in `postprocessing.ipynb` notebook - it takes as input mosaics and predictions for validation and test (the files could be changed in variables `test_preds` and `val_pred`). In the end of this notebook you can see Kaggle submit file generation. 

This postprocessing was pretty useful on Public LB, but does almost nothing in Private LB.


# Using a simple model
Using a single simple model is done similiarly to creating our final submission.

## 1. Training the model and creating predictions
For a simple single model we recommend using `experiments/nopoolrefinenet_dpn92.py`. Now for training put your data into the train folder just like specified in the directoy structure.
To run the most simple model (so no cross-validation) only run the first iteration of the loop at the bottom of the experiment file. This is equivalent to running only the first fold of 5-fold cross-validation.
You can even lower the number of epochs to run for faster training. It should only give slightly worse results the running the full training. 

Here is an example configuration which only takes 1/4 of the original training duration:
```
lr_scheduler = utils.CyclicLR(optimizer, 5, {
    0: (1e-4, 1e-6),
    25: (0.5e-4, 1e-6),
    40: (1e-5, 1e-6),
})

epochs = 50
```

Put your test data into the specified test-directory before training. The predictions for this data will created after training automatically.
This can be changed in the loop at the bottom at the file. Changing this is trivial and should only need a look at the code as explanation.

Now run `CUDA_VISIBLE_DEVICES=0 python -m experiments.nopoolrefinenet_dpn92` to start training.

## 2. Creating a submission
In `ensemble.py` only include the simple model like this:

```
experiments = [
    'nopoolrefinenet_dpn92'
]
```

Also specify the ensemble name.

Change the first loop from 5 iterations to 1:
```
for i in range(1):
```

Run `python ensemble.py`.

Now a valid submission should have been created in `./submissions`.