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

from collections import Counter
from typing import List

import torch
import torch.nn as nn
from torch.optim import Optimizer


def params(optimizer: Optimizer) -> List[nn.Parameter]:
    """
    Return all parameters controlled by the optimizer
    Args:
        optimizer: optimizer

    Returns:
        Flat list of parameters from all ``param_groups``
    """
    ret = []
    for param_group in optimizer.param_groups:
        ret += [p for p in param_group["params"] if p.requires_grad]
    return ret


def check_grad_sample_batch_size_consistency(
    params: List[torch.Tensor], grad_samples: List[torch.Tensor]
) -> None:
    """
    Validates that all per-sample gradient tensors have a consistent batch size (first dimension).

    Args:
        params: List of parameters
        grad_samples: List of per-sample gradient tensors corresponding to params

    Raises:
        ValueError: If different parameters have different batch sizes (e.g. due to batch-broadcasted
            or unexpanded inputs in embedding layers).
    """
    if not grad_samples:
        return

    batch_sizes = [len(g) for g in grad_samples]
    if len(set(batch_sizes)) <= 1:
        return

    counts = Counter(batch_sizes)
    expected_batch_size = counts.most_common(1)[0][0]
    mismatches = []
    for p, g in zip(params, grad_samples):
        if len(g) != expected_batch_size:
            param_name = (
                getattr(p, "_param_name", None)
                or getattr(p, "name", None)
                or f"tensor with shape {tuple(p.shape)}"
            )
            mismatches.append(
                f"Parameter ({param_name}) has batch dimension {len(g)}, expected {expected_batch_size}"
            )

    mismatch_str = "; ".join(mismatches)
    raise ValueError(
        f"Inconsistent per-sample gradient batch dimensions across parameters: {mismatch_str}. "
        "This often occurs when a layer receives batch-broadcasted or unexpanded inputs (e.g., position "
        "embeddings with batch size 1 in models like GPT-2). For differential privacy, each trainable "
        "parameter must receive inputs expanded along the batch dimension (e.g., position_ids.expand(batch_size, -1)), "
        "or be frozen with requires_grad=False if it should not be trained privately."
    )
