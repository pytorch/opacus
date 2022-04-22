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

import pickle
from collections import namedtuple
from typing import Any, Dict, List, Optional

import torch
from layers import LayerType


Memory = namedtuple("Memory", "prev_max_mem, cur_mem")


def reset_peak_memory_stats(device: torch.device) -> Memory:
    """Safely resets CUDA peak memory statistics of device if it is
    a CUDA device.

    Notes: ``torch.cuda.reset_peak_memory_stats(device)`` will error
    if no CUDA memory has been allocated to the device.

    Args:
        device: A torch.device

    Returns:
        max_memory_allocated before resetting the statistics and
        memory_allocated, both in bytes
    """
    prev_max_memory = torch.cuda.max_memory_allocated(device)
    memory_allocated = torch.cuda.memory_allocated(device)

    if prev_max_memory != memory_allocated and prev_max_memory > 0:
        # raises RuntimeError if no previous allocation occurred
        torch.cuda.reset_peak_memory_stats(device)
        assert torch.cuda.max_memory_allocated(device) == memory_allocated

    return Memory(prev_max_memory, memory_allocated)


def get_layer_set(layer: str) -> str:
    """Layers in the same layer set share a config.

    Args:
        layer: Full name of the layer. This will be the PyTorch or Opacus
        name of the layer in lower case (e.g. linear, rnn, dprnn), prefixed with
        gsm_ (e.g. gsm_linear, gsm_dprnn) if DP is enabled. MultiheadAttention
        is abbreviated to mha.

    Returns:
        The name of the layer set, where a set of layers are defined as layers
        that share the same __init__ signature.

    Notes:
        All RNN-based models share a config.

    """
    layer_set = layer.replace("gsm_dp", "").replace("gsm_", "").replace("dp", "")

    # all RNN-based model use the same config
    if layer_set in ["rnn", "gru", "lstm"]:
        layer_set = "rnn_base"

    return layer_set


def get_path(
    layer: LayerType,
    batch_size: int,
    num_runs: int,
    num_repeats: int,
    random_seed: Optional[int] = None,
    forward_only: bool = False,
    root: str = "./results/raw/",
    suffix: str = "",
) -> str:
    """Gets the path to the file where the corresponding results are located.
    File is presumed to be a pickle file.

    Args:
        layer: full layer name
        batch_size: batch size
        num_runs: number of runs per benchmark
        num_repeats: how many benchmarks were run
        random_seed: the initial random seed
        forward_only: whether backward passes were skipped
        root: directory to write results to
        suffix: optional string to append to file name

    Returns:
        Path to results pickle file
    """
    pickle_name = f"{layer}_bs_{batch_size}_runs_{num_runs}_repeats_{num_repeats}_seed_{random_seed}"
    if forward_only:
        pickle_name += "_forward_only"

    if len(suffix) and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    return f"{root}{pickle_name}{suffix}.pkl"


def save_results(
    layer: LayerType,
    batch_size: int,
    num_runs: int,
    num_repeats: int,
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    random_seed: Optional[int] = None,
    forward_only: bool = False,
    root: str = "./results/raw/",
    suffix: str = "",
) -> None:
    """Saves the corresponding results as a pickle file.

    Args:
        layer: full layer name
        batch_size: batch size
        num_runs: number of runs per benchmark
        num_repeats: how many benchmarks were run
        runtimes: list of runtimes of length num_repeats
        memory: list of memory stats of length num_repeats
        config: layer config
        random_seed: the initial random seed
        forward_only: whether backward passes were skipped
        root: directory to write results to
        suffix: optional string to append to file name
    """
    path = get_path(
        layer=layer,
        batch_size=batch_size,
        num_runs=num_runs,
        num_repeats=num_repeats,
        random_seed=random_seed,
        forward_only=forward_only,
        root=root,
        suffix=suffix,
    )

    with open(path, "wb") as handle:
        pickle.dump(
            {
                "layer": layer,
                "batch_size": batch_size,
                "num_runs": num_runs,
                "num_repeats": num_repeats,
                "random_seed": random_seed,
                "forward_only": forward_only,
                "results": results,
                "config": config,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
