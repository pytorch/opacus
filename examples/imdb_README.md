# Simple run
run with dp:
```
python imdb.py --device cuda:0

```
for with no dp:
```
python imdb.py --device cuda:0 --disable-dp
```

Default parameters are tweaked for optimal performance with DP

# Expeted results
We use the same architecture and dataset as described in https://arxiv.org/pdf/1911.11607.pdf.

Reference implementation for the paper (tensorflow privacy): https://github.com/tensorflow/privacy/blob/master/research/GDP_2019/imdb_tutorial.py

Unless specified otherwise, we set `batch_size=64` for a faster iteration (opacus has higher requirements in memory compared to tensorflow-privacy). We didn't use virtual batches for this experiment, and therefowe were limited to the given batch size.
All metrics are calculated on test data after training for 20 epochs

| Approach                                | Accuracy |
| ----------------------------------------|:--------:|
| Claimed in paper (no DP)                | 84%      |
| tf_privacy  (no DP)                     | 84%      |
| opacus (no DP)                          | 84%      |
| Claimed in paper (with DP)              | 84%      |
| tf_privacy  (with DP)                   | 80%      |
| tf_privacy  (with DP, bsz=512)          | 84%      |
| opacus (Adam, lr = 0.02)                | 74%      |
| opacus (Adam, lr = 0.02, 100 epochs)    | 78%      |
| opacus (Adagrad, lr = 0.02)             | 59%      |
| opacus (Adagrad, lr = 0.1)              | 65%      |
| opacus (SGD, lr = 0.1)                  | 63%      |
