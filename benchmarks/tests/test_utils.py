# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import shutil
from typing import Any, Dict, List, Tuple

import pytest
import torch
from helpers import get_n_byte_tensor, skipifnocuda
from utils import get_layer_set, get_path, reset_peak_memory_stats, save_results


@pytest.mark.parametrize(
    "layer_set, layers",
    [
        # Pytorch layers are named layer (no DP) or gsm_layer (DP)
        ("linear", ["linear", "gsm_linear"]),
        ("conv", ["conv", "gsm_conv"]),
        # Opacus layers are named dplayer (no DP) or gsm_dplayer (DP)
        ("mha", ["mha", "dpmha", "gsm_dpmha"]),
        # RNN-based models share the same interface
        ("rnn_base", ["rnn", "dprnn", "gsm_dprnn"]),
        ("rnn_base", ["lstm", "dplstm", "gsm_dplstm"]),
    ],
)
def test_get_layer_set(layer_set: str, layers: List[str]) -> None:
    """Tests assignment of individual layers to the layer set.

    Args:
        layer_set: layer set (e.g. linear, rnn_base)
        layers: non-exhaustive list of layers that belong to the layer_set
    """
    assert all(get_layer_set(layer) == layer_set for layer in layers)


@skipifnocuda
@pytest.mark.parametrize(
    "prev_max_memory, allocated_memory",
    [
        # prev_max_memory = allocated_memory = 0 --> (0, 0)
        (0, 0),
        # prev_max_memory = allocated_memory > 0 --> (prev_max_memory, prev_max_memory)
        (1, 1),
        # prev_max_memory > allocated_memory = 0 --> (prev_max_memory, 0)
        (1, 0),
        # prev_max_memory > allocated_memory > 0 --> (prev_max_memory, allocated_memory)
        (2, 1),
    ],
)
def test_reset_peak_memory_stats(prev_max_memory: int, allocated_memory: int) -> None:
    """Tests resetting of peak memory stats.

    Notes: Only the relative and not the absolute sizes of prev_max_memory and
    allocated_memory are relevant.

    Args:
        prev_max_memory: current maximum memory stat to simulate
        allocated_memory: current allocated memory to simulate
    """
    device = torch.device("cuda:0")

    # keep x, delete y
    x = get_n_byte_tensor(allocated_memory, device=device)
    y = get_n_byte_tensor(prev_max_memory - allocated_memory, device=device)
    del y

    # get the true allocated memory (CUDA memory is allocated in blocks)
    prev_max_memory = torch.cuda.max_memory_allocated(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    assert prev_max_memory >= allocated_memory
    assert reset_peak_memory_stats(device) == (prev_max_memory, allocated_memory)

    # clean up
    del x
    torch.cuda.reset_peak_memory_stats(device)
    assert torch.cuda.max_memory_allocated(device) == 0
    assert torch.cuda.memory_allocated(device) == 0


@pytest.mark.parametrize(
    "config, path",
    [
        (
            {"layer": "linear", "batch_size": 64, "num_runs": 10, "num_repeats": 100},
            "./results/raw/linear_bs_64_runs_10_repeats_100_seed_None.pkl",
        ),
        (
            {
                "layer": "gsm_rnn",
                "batch_size": 128,
                "num_runs": 5,
                "num_repeats": 20,
                "random_seed": 13964,
                "forward_only": True,
            },
            "./results/raw/gsm_rnn_bs_128_runs_5_repeats_20_seed_13964_forward_only.pkl",
        ),
        (
            {
                "layer": "dpmha",
                "batch_size": 16,
                "num_runs": 20,
                "num_repeats": 50,
                "random_seed": 88362,
                "root": "./results/tmp/",
                "suffix": "no_3",
            },
            "./results/tmp/dpmha_bs_16_runs_20_repeats_50_seed_88362_no_3.pkl",
        ),
    ],
)
def test_get_path(config: Dict[str, Any], path: str) -> None:
    """Tests result pickle path generation.

    Args:
        config: arguments to pass to get_path
        path: corresponding path
    """
    assert path == get_path(**config)


@pytest.fixture(scope="function")
def pickle_data_and_config(
    config: Dict[str, Any], root: str, suffix: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    # setup test directory and save results to pickle file
    os.mkdir(root)
    save_results(**config, results=[], config={}, root=root, suffix=suffix)

    # return pickle data and the config
    with open(get_path(**config, root=root, suffix=suffix), "rb") as f:
        yield pickle.load(f), config

    # remove directory
    shutil.rmtree(root)


@pytest.mark.parametrize(
    "config, root, suffix",
    [
        (
            {
                "layer": "linear",
                "batch_size": 64,
                "num_runs": 10,
                "num_repeats": 100,
                "random_seed": 13964,
            },
            "tests/tmp/",
            "",
        ),
        (
            {
                "layer": "dpmha",
                "batch_size": 16,
                "num_runs": 20,
                "num_repeats": 50,
                "random_seed": 88362,
                "forward_only": True,
            },
            "tests/tmp1/",
            "no_3",
        ),
    ],
)
def test_save_results(
    pickle_data_and_config: Tuple[Dict[str, Any], Dict[str, Any]]
) -> None:
    """Tests saving benchmark results.

    Args:
        pickle_data_and_config: tuple consisting of the pickle data as a dict and the
            original data as a dict, as given in the config
    """
    pickle_data, config = pickle_data_and_config
    for key, value in config.items():
        assert pickle_data[key] == value
