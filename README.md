Hello!

Below you can find a outline of how to reproduce our solution for the TGS Salt Identification competition.
If you run into any trouble with the setup/code or have any questions please contact us at tim.joseph@mctigger.com

#ARCHIVE CONTENTS (TODO)

#HARDWARE: (The following specs were used to create the original solution)
3x:
Ubuntu 16.04 LTS (512 GB boot disk)
16GB RAM
2x NVIDIA GTX 1080TI 11GB

1x
Ubuntu 16.04 LTS (512 GB boot disk)
256GB RAM
4x NVIDIA Tesla P100

#SOFTWARE (python packages are detailed separately in `environment.yml`):
Anaconda 5.3, Python 3.6, NVIDIA drivers 410.66

(CUDA and cuDNN are not listed here since these are part of the Pytorch conda package, see `environment.yml`)

#DATA SETUP 
Download and extract competition data into `./data`.

#DATA PROCESSING

#MODEL BUILD
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
In `ensemble.py` an ensemble_name and all experiments must be specified that should be included into an ensemble. When you run `python ensemble.py` all the test predictions from the different specified experiments are average. Two files are created:
1. `{ensemble_name}.npz` in `./predictions/test/` which contains the averaged test predictions.
2. `{ensemble_name}` in `./submissions/` which is a submission file containing the masks from 1., but thresholded at `0.5`.

## Creating a fold-ensembles and fold-submissions
For some experiments it is necessary to create some ensembled test predictions which only include predictions from specific folds of experiments.
For this reason you can use `python ensemble_foldwise.py` which essentially works just like `ensemble.py`, but creates test predictions and submissions for each fold.
The submissions are not of further use, but can be submitted to the leaderboard for some insight into their performance.
The created test predictions are saved into `./predictions/test/` as `{ensemble_name}-{i}.npz` where `i` is between 0 and 4.

## Using a different test set
Using a different test set is easy. Just use `predict.py` as follows: `python predict.py {your_experiment} {path_to_test_set_images} {output_dir}`.
Test predictions will be created for the specified experiment by using the images contained in the test set iamges directory and saved into `./predictions/{output_dir}`.
To create a submission just use `ensemble.py` like specified above, but change `input_dir` to the specified `output_dir` in `predict.py`.


