# Benchmarks

A set of micro-benchmarks that measure elapsed runtime and allocated CUDA memory for both [basic modules](https://github.com/pytorch/opacus/tree/main/opacus/grad_sample) and [more complex layers](https://github.com/pytorch/opacus/tree/main/opacus/layers), as well as their respective [torch.nn](https://pytorch.org/docs/stable/nn.html) counterparts.

Requires PyTorch version >= 1.10.0.

## Contents

- [run_benchmarks.py](run_benchmarks.py) runs a list of benchmarks based on the given config file and writes out the results as pickle files to `results/raw` (by default)

- [benchmark_layer.py](benchmark_layer.py) benchmarks a single layer at the given batch size for both runtime and memory

- [config.json](config.json) contains an example JSON config for running benchmarks

- [layers.py](layers.py) implements each layer with its corresponding inputs/outputs, moving each component to device, and collecting respective memory statistics

- [utils.py](utils.py) implements helper functions, e.g. for saving results as pickle files


## Benchmarks

For each layer and batch size in [config.json](config.json), [run_benchmarks.py](run_benchmarks.py) will do the following:
```
Do this num_runs times:
    Init layer, one batch of random input, one batch of random "labels"
    Move each component to GPU and collect its allocated CUDA memory (if applicable)

    Start timer

    Do this num_repeats times:
        preds = layer(input)
        loss = criterion(preds, labels)
        loss.backward()

    Stop timer
    
    Return elapsed time / num_repeats and memory statistics
```

## Layers

All layers follow the corresponding `torch.nn` module's interface with the same default values (if not specified in [config.json](config.json)).

A note on `input_shape` in [config.json](config.json): parameters that are shared between the model and the input are listed separately. Therefore, the actual input shape will vary:

- Linear: `(batch_size, *input_shape, in_features)`

- Convolutional: `(batch_size, in_channels, *input_shape)`

- LayerNorm:
    - Input: `(batch_size, *input_shape)`
    - Normalized shape: `(input_shape[-D:])`

- InstanceNorm: `(batch_size, num_features, *input_shape)`

- GroupNorm: `(batch_size, num_channels, *input_shape)`

- Embedding: `(batch_size, *input_shape)`

- MultiheadAttention:
    - If not batch_first: `(targ_seq_len, batch_size, embed_dim)`
    - Else: `(batch_size, targ_seq_len, embed_dim)`

- RNN, GRU, LSTM:
    - If not batch_first: `(seq_len, batch_size, input_size)`
    - Else: `(batch_size, seq_len, input_size)`


## Usage

`run_benchmarks.py` without additional arguments will replicate the results in the [technical report introducing Opacus](https://arxiv.org/abs/2109.12298), which you can cite as follows:
```
@article{opacus,
  title={Opacus: {U}ser-Friendly Differential Privacy Library in {PyTorch}},
  author={Ashkan Yousefpour and Igor Shilov and Alexandre Sablayrolles and Davide Testuggine and Karthik Prasad and Mani Malek and John Nguyen and Sayan Ghosh and Akash Bharadwaj and Jessica Zhao and Graham Cormode and Ilya Mironov},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

If saving results, ensure that the `results/raw` (or otherwise specified) root directory exists.

```
usage: run_benchmarks.py [-h]
                         [--layers {linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} [{linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} ...]]
                         [--batch_sizes BATCH_SIZES [BATCH_SIZES ...]] [--num_runs NUM_RUNS]
                         [--num_repeats NUM_REPEATS] [--forward_only] [--random_seed RANDOM_SEED]
                         [-c CONFIG_FILE] [--cont] [--root ROOT] [--suffix SUFFIX] [--no_save]
                         [-v]

optional arguments:
  -h, --help            show this help message and exit
  --layers {linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} [{linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} ...]
  --batch_sizes BATCH_SIZES [BATCH_SIZES ...]
  --num_runs NUM_RUNS   number of benchmarking runs
  --num_repeats NUM_REPEATS
                        how many forward/backward passes per run
  --forward_only        only run forward passes
  --random_seed RANDOM_SEED
                        random seed for the first run of each layer, subsequent runs increase the
                        random seed by 1
  -c CONFIG_FILE, --config_file CONFIG_FILE
  --cont                only run missing experiments
  --root ROOT           path to directory that benchmark results should be saved in
  --suffix SUFFIX       suffix to append to each result file's name
  --no_save
  -v, --verbose
```

## Tests

```python -m pytest tests/```

Tests for CUDA memory benchmarking will be skipped on CPU.
