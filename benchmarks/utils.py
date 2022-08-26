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

import glob
import pickle
from collections import namedtuple
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
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


def generate_report(path_to_results: str, save_path: str, format: str) -> None:
    """Generate a report from the benchamrks outcome.
    The output is a file whic contains the runtime and memory of each layer.
    If multiple layer variants were run (pytorch nn, DP, or GSM).
    Then we will compare the performance of both DP and GSM to pytorch.nn.

    Args:
        path_to_results: the path that `run_benchmarks.py` has saved results to.
        save_path: path to save the output.
        format: output format : csv or pkl.
    """
    path_to_results = (
        path_to_results if path_to_results[-1] != "/" else path_to_results[:-1]
    )
    files = glob.glob(f"{path_to_results}/*")

    if len(files) == 0:
        raise Exception(f"There were no result files in the path {path_to_results}")

    raw_results = []
    for result_file in files:
        with open(result_file, "rb") as handle:
            raw_results.append(pickle.load(handle))

    results_dict = []
    for raw in raw_results:
        runtime = np.mean([i["runtime"] for i in raw["results"]])
        memory = np.mean([i["memory_stats"]["max_memory"] for i in raw["results"]])
        result = {
            "layer": raw["layer"],
            "batch_size": raw["batch_size"],
            "num_runs": raw["num_runs"],
            "num_repeats": raw["num_repeats"],
            "forward_only": raw["forward_only"],
            "runtime": runtime,
            "memory": memory,
        }
        results_dict.append(result)

    results = pd.DataFrame(results_dict)
    results["variant"] = "control"
    results["variant"][results["layer"].str.startswith("gsm")] = "gsm"
    results["variant"][results["layer"].str.startswith("dp")] = "dp"
    results["base_layer"] = results["layer"].str.replace("(gsm_)|(dp)", "")

    pivot = results.pivot_table(
        index=["batch_size", "num_runs", "num_repeats", "forward_only", "base_layer"],
        columns=["variant"],
        values=["runtime", "memory"],
    )

    def add_ratio(df, metric, variant):
        if variant in df.columns.get_level_values("variant"):
            df[(metric, f"{variant}/control")] = (
                df.loc[:, (metric, variant)] / df.loc[:, (metric, "control")]
            )
        else:
            df[(metric, f"{variant}/control")] = np.nan

    if "control" in results["variant"].tolist():
        add_ratio(pivot, "runtime", "dp")
        add_ratio(pivot, "memory", "dp")
        add_ratio(pivot, "runtime", "gsm")
        add_ratio(pivot, "memory", "gsm")
        pivot.columns = pivot.columns.set_names("value", level=1)

    output = pivot.sort_index(axis=1).sort_values(
        ["batch_size", "num_runs", "num_repeats", "forward_only"]
    )
    if format == "csv":
        output.to_csv(save_path)
    else:
        output.to_pickle(save_path)
