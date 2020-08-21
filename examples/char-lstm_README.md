# Simple run

Download the training zip from https://download.pytorch.org/tutorial/data.zip and then extract it to a local directory. In this example, we assume that the data path is /home/data_folder/names/*.txt

Run with dp:
```
python char-lstm-classification.py --device cuda:0 --learning-rate=2.0 --n-hidden=128 --delta=8e-5 --batch-size=800 --sigma=1.0 --training-path=/home/data_folder/names/*.txt --max-per-sample-grad-norm=1.5
```
Expected multi-class accuracy : 50.74 training time : 31 m 22s epsilon : 11.81  best_alpha : 2.7

Run without dp:
```
python char-lstm-classification.py --device cuda:0 --learning-rate=2.0 --n-hidden=128 --delta=8e-5 --batch-size=800 --sigma=1.0 --training-path=/home/data_folder/names/*.txt --max-per-sample-grad-norm=1.5 --disable-dp
```
Expected multi-class accuracy : 51.43, training time : 28 m 12s

# Overview

This example demonstrates how to run a privacy-preserving LSTM (with DP) on a name classification task. Currently we support only one-layer, one-directional LSTMs without dropout. This closely follows the example provided in https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html (with sections of duplicated code) and is structured into dataset preparation, model definition and evaluation.

The example below also incorporates the following modifications to the original tutorial :
(1) Batch support in training with padding to decrease training time, with a padding token #.
(2) The dataset of names is split into a training and a validation set with no overlap.
(3) A simple character RNN is replaced with a batched LSTM model

# Hyper-parameter Tuning

This tutorial utilizes several sets of hyper-parameters, both for the model (eg. learning rate, number of neurons, embedding size) as well as for the optimizer with DP (such as delta and sigma). We measure performance on the validation set and select the model with the best multi-class balanced accuracy on the validation set. The parameter ranges we have considered are given below. We run a random grid search to select the best hyper-parameters for our model.
We also have a maximum sequence length of 15, a training/validation split of 80/20 and run our model training for 1000 iterations.
```
batch_size : [200, 400, 800],
n_hidden : [128, 256, 512],
max_seq_length : [15],
learning_rate : [1.0, 1.5, 2.0],
iterations : [1000],
train_eval_split : [0.8],
sigma : [0.125, 0.25, 0.5, 1.0],
max_per_sample_grad_norm : [0.5, 0.75, 1.0, 1.5],
delta : [2e-5, 4e-5, 8e-5]
```

# Comparison with baselines

We compare performances of a baseline LSTM with and without privacy with a simple RNN as provided in the tutorial. The character RNN is re-implemented and performance measured on the same validation set. Table below lists the baselines and the best model obtained with DP obtains slightly less performance than the non-DP LSTM model with little loss of privacy (corresponding to an epsilon = 11.796 and best alpha = 2.7).

The metric is balanced multi-class accuracy (as defined in sklearn.metrics.balanced_accuracy_score from the scikit-learn library), which we utilize to account for imabalnce among categories and prevent any one frequent category (such as English/Russian) from dominating the metrics.

| Approach                                | Accuracy |
| ----------------------------------------|:--------:|
| RNN                                     | 46.40%   |
| Char-LSTM (no-DP)                       | 51.43%   |
| Char-LSTM (DP)                          | 50.74%   |
