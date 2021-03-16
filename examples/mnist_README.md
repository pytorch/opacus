# First run
To run a basic training script without differential privacy:
```shell
python mnist.py --device=cpu --disable-dp --n=20 --lr=.1 -sr=0.004
```
The first time the script runs, it attempts to download the MNIST dataset from http://yann.lecun.com and place it in `../mnist/MNIST/raw`. If you prefer a different location or your execution environment does not have access to the outside world, download and unpack the dataset yourself and pass the location as `--data-root=custom_dir_name`. The script will expect to find under `custom_dir_name/MNIST/processed` two files: `test.pt` (7.9 MB) and `training.pt` (47.5 MB).

If the run is successful, you will something similar to this (your exact accuracy and performance numbers will vary):
```
100%|██████████| 240/240 [00:14<00:00, 16.58it/s]
Train Epoch: 1   Loss: 0.536689
100%|██████████| 240/240 [00:14<00:00, 16.48it/s]
Train Epoch: 2   Loss: 0.102070
...
100%|██████████| 240/240 [00:14<00:00, 17.02it/s]
Train Epoch: 10   Loss: 0.025479
100%|██████████| 10/10 [00:01<00:00,  6.64it/s]

Test set: Average loss: 0.0000, Accuracy: 9893/10000 (98.93%)
```

To train a differentially private model, run the following command:
```shell
python mnist.py --device=cpu -n=15 --lr=.25 --sigma=1.3 -c=1.5 -sr=0.004
```
If the run is successful, expect to see
```
100%|██████████| 240/240 [00:22<00:00, 10.48it/s]
Train Epoch: 1    Loss: 0.912457 (ε = 0.71, δ = 1e-05) for α = 18.0
...
100%|██████████| 240/240 [00:22<00:00, 10.79it/s]
Train Epoch: 15   Loss: 0.404850 (ε = 1.16, δ = 1e-05) for α = 17.0
100%|██████████| 10/10 [00:01<00:00,  6.76it/s]

Test set: Average loss: 0.0004, Accuracy: 9486/10000 (94.86%)
```

# Sample parameter sets

**Baseline: no differential privacy**

Command: `--disable-dp --n=20 --lr=.1 -sr=0.004`

Result: accuracy averaged over 10 runs 98.94% ± 0.32%

**(6.86, 10<sup>-5</sup>)-DP**

Command: `-n=45 --lr=.25 --sigma=.7 -c=1.5 -sr=0.004`

Result: accuracy averaged over 10 runs 97.09% ± 0.17%

**(2.91, 10<sup>-5</sup>)-DP**

Command: `-n 60 --lr=.15 --sigma=1.1 -c=1.0 -sr=0.004`

Result: accuracy averaged over 10 runs 96.78% ± 0.21%

**(1.16, 10<sup>-5</sup>)-DP**

Command: `-n=15 --lr=.25 --sigma=1.3 -c=1.5 -sr=0.004`

Result: accuracy averaged over 10 runs 94.63% ± 0.34%
