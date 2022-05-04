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

import math
import time

import pytest
import torch
import torch.nn as nn
from benchmark_layer import run_layer_benchmark
from helpers import get_actual_memory_allocated, get_n_byte_tensor, skipifnocuda
from layers import Layer
from utils import reset_peak_memory_stats


class FakeModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._runtime = kwargs.get("runtime", 0)
        self._pass_memory = kwargs.get("pass_memory", 0)
        self._parameters = {
            "fake_param": get_n_byte_tensor(kwargs.get("layer_memory", 0))
        }


class FakeLayer(Layer):
    """Fake layer to test runtime and memory benchmarking.

    Kwargs:
        runtime: duration for one forward or backward pass
        pass_memory: memory used during a forward or backward pass
        layer_memory: memory taken up by the module
    """

    def __init__(self, **kwargs) -> None:
        self._runtime = kwargs.get("runtime", 0)
        self._pass_memory = kwargs.get("pass_memory", 0)
        self._input_tensor = get_n_byte_tensor(0)
        self._module = FakeModule(**kwargs)
        self._labels = get_n_byte_tensor(0)

    def forward_only(self) -> torch.Tensor:
        """Wait for self.duration and allocate self.max_memory bytes"""
        time.sleep(self._runtime)
        tensor = get_n_byte_tensor(
            self._pass_memory,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        return tensor

    def forward_backward(self) -> None:
        """2x runtime and memory of forward_only"""
        _ = self.forward_only()
        _ = self.forward_only()


@pytest.mark.parametrize("duration", [0, 0.005, 0.01, 0.05])
@pytest.mark.parametrize("num_repeats", [10, 20, 50])
@pytest.mark.parametrize("forward_only", [False, True])
def test_runtime_benchmark(
    duration: float, num_repeats: int, forward_only: bool
) -> None:
    """Tests runtime benchmarks on a dummy layer.

    Args:
        duration: duration (seconds) to test runtime benchmarks for
        num_repeats: number of repeats for each benchmark
        forward_only: whether to benchmark forward passes or forward/backward passes
    """
    runtime, memory_stats = run_layer_benchmark(
        num_repeats=num_repeats,
        forward_only=forward_only,
        create_layer=FakeLayer,
        runtime=duration,
    )
    # account for small variations
    assert abs(runtime - ((2 - forward_only) * duration)) < 0.01

    # check that no memory allocation took place
    assert memory_stats["layer"] == 0 and memory_stats["max_memory"] == 0


@skipifnocuda
@pytest.mark.parametrize("pass_bytes", [0, 1, 100, 500, 4096])
@pytest.mark.parametrize("layer_bytes", [0, 1, 256, 1024, 2000])
@pytest.mark.parametrize("num_repeats", [10, 20, 50, 100])
@pytest.mark.parametrize("forward_only", [False, True])
def test_memory_benchmark(
    pass_bytes: int, layer_bytes: int, num_repeats: int, forward_only: bool
) -> None:
    """Tests CUDA memory benchmarks on a dummy layer.

    Args:
        pass_bytes: number of bytes allocated during a forward or forward/backward pass
        layer_bytes: number of bytes allocated for the layer
        num_repeats: number of repeats for each benchmark
        forward_only: whether to benchmark forward passes or forward/backward passes
    """
    device = torch.device("cuda:0")

    true_pass_memory = get_actual_memory_allocated(pass_bytes, device=device)
    true_layer_memory = get_actual_memory_allocated(layer_bytes, device=device)

    # reset memory stats and ensure there is no memory leakage
    assert reset_peak_memory_stats(device).cur_mem == 0

    runtime, memory_stats = run_layer_benchmark(
        num_repeats=num_repeats,
        forward_only=forward_only,
        create_layer=FakeLayer,
        layer_memory=layer_bytes,
        pass_memory=pass_bytes,
    )

    assert memory_stats["layer"] == true_layer_memory
    assert (
        memory_stats["max_memory"]
        == true_layer_memory + (2 - forward_only) * true_pass_memory
    )

    # reset memory stats and ensure there is no memory leakage
    assert reset_peak_memory_stats(device).cur_mem == 0


@skipifnocuda
@pytest.mark.parametrize("pass_bytes", [0, 1, 100, 500, 4096])
@pytest.mark.parametrize("layer_bytes", [0, 1, 256, 1024, 2000])
@pytest.mark.parametrize("num_repeats", [10, 20, 50, 100])
@pytest.mark.parametrize("forward_only", [False, True])
def test_memory_benchmark_strict(
    pass_bytes: int, layer_bytes: int, num_repeats: int, forward_only: bool
) -> None:
    """Tests CUDA memory benchmarks on a dummy layer by predicting each measurement
    using the CUDA memory block size.

    Notes:
        During the experiments included in the paper, CUDA memory is allocated in
        blocks, where block sizes vary across kernels. New CUDA memory is allocated
        for each new tensor. This test will fail under a different allocation
        scheme.

    Args:
        pass_bytes: number of bytes allocated during a forward or forward/backward pass
        layer_bytes: number of bytes allocated for the layer
        num_repeats: number of repeats for each benchmark
        forward_only: whether to benchmark forward passes or forward/backward passes
    """
    device = torch.device("cuda:0")

    # find the block size by creating a tensor of size 1 byte
    tiny_tensor = get_n_byte_tensor(1, device=device)
    BLOCK_SIZE = torch.cuda.max_memory_allocated(device)
    del tiny_tensor

    num_blocks_pass = math.ceil(pass_bytes / BLOCK_SIZE)
    num_blocks_layer = math.ceil(layer_bytes / BLOCK_SIZE)

    # reset memory stats and ensure there is no memory leakage
    assert reset_peak_memory_stats(device).cur_mem == 0

    runtime, memory_stats = run_layer_benchmark(
        num_repeats=num_repeats,
        forward_only=forward_only,
        create_layer=FakeLayer,
        layer_memory=layer_bytes,
        pass_memory=pass_bytes,
    )
    assert memory_stats["layer"] == num_blocks_layer * BLOCK_SIZE
    assert (
        memory_stats["max_memory"]
        == (num_blocks_layer + (2 - forward_only) * num_blocks_pass) * BLOCK_SIZE
    )

    # reset memory stats and ensure there is no memory leakage
    assert reset_peak_memory_stats(device).cur_mem == 0
