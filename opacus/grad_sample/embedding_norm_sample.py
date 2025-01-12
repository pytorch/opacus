#!/usr/bin/env python3
# Copyright 2024, The Opacus authors.
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

"""Utility for computing gradient norm for the embedding layer.

Based on the algorithm from the paper:
https://proceedings.neurips.cc/paper_files/paper/2023/file/a45d344b28179c8da7646bc38ff50ad8-Paper-Conference.pdf.
"""
from typing import Dict, List

import torch
from torch import nn


def compute_embedding_norm_sample(
    layer: nn.Embedding,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """Computes per sample gradient norms for ``nn.Embedding`` layer.

    Args:
      layer: Layer
      activations: Activations
      backprops: Backpropagations

    Returns:
      A dictionary of parameter gradients

    NOTE: Here is an example input, and the expected intermediate values. This
    is proivided to help in understanding the algorithm:
    Inputs:
      layer:  Embedding(3, 1) # (vocab_size, embedding_dim)
      activations:  [tensor([[1, 1],
          [2, 0],
          [2, 0]])]
      backprops:  tensor([[[0.2], [0.2]],
          [[0.3], [0.1]],
          [[0.3], [0.1]]])
      backprops.shape:  torch.Size([3, 2, 1])

    Intermediate values:
      input_ids:  tensor([[1, 1],
          [2, 0],
          [2, 0]])
      input_ids.shape:  torch.Size([3, 2])
      grad_values:  tensor([[0.2000],
          [0.2000],
          [0.3000],
          [0.1000],
          [0.3000],
          [0.1000]])
      grad_values.shape:  torch.Size([6, 1])
      nrows:  3
      ncols:  2
      row_indices:  tensor([[0],
          [0],
          [1],
          [1],
          [2],
          [2]])
      flattened_indices:  tensor([[1],
          [1],
          [2],
          [0],
          [2],
          [0]])
      paired_indices:  tensor([[0, 1],
          [0, 1],
          [1, 2],
          [1, 0],
          [2, 2],
          [2, 0]])
      unique_paired_indices:  tensor([[0, 1],
          [1, 0],
          [1, 2],
          [2, 0],
          [2, 2]])
      new_index_positions:  tensor([0, 0, 2, 1, 4, 3])
      num_unique_paired_indices:  5
      summed_gradients:  tensor([[0.4000],
          [0.1000],
          [0.3000],
          [0.1000],
          [0.3000]])
      sqr_gradient_sum:  tensor([0.1600, 0.0100, 0.0900, 0.0100, 0.0900])
      unique_batch_ids:  tensor([0, 1, 1, 2, 2])
      result:  tensor([0.1600, 0.1000, 0.1000])
      result_sqrt:  tensor([0.4000, 0.3162, 0.3162])
    """
    device = activations[0].device
    input_ids = activations[0].to(device)
    grad_values = backprops.to(device)

    # Reshape input_ids preserving the batch size as the first dimension
    input_ids = input_ids.reshape(input_ids.shape[0], -1)

    # Reshape grad_values preserving the embedding dimension as the last dimension
    grad_values = grad_values.reshape(-1, grad_values.size(-1))

    # Create 1D tensor of row indices
    nrows = input_ids.size(0)
    ncols = input_ids.size(1)
    row_indices = (
        torch.repeat_interleave(torch.arange(nrows).to(device), ncols)
        .unsqueeze(-1)
        .to(device)
    )

    # Pair the input IDs with the row indices
    flattened_indices = input_ids.view(-1, 1)
    paired_indices = torch.cat([row_indices, flattened_indices], dim=1).to(device)

    # Get unique paired indices and new index positions for aggregation
    unique_paired_indices, new_index_positions = torch.unique(
        paired_indices, dim=0, return_inverse=True, sorted=True
    )

    # Sum gradients over new index positions and compute squared gradient norms
    num_unique_paired_indices = unique_paired_indices.size(0)
    summed_gradients = torch.zeros(
        num_unique_paired_indices, grad_values.size(-1), device=device
    )
    summed_gradients = summed_gradients.index_add(
        0, new_index_positions.to(device), grad_values
    )
    sqr_gradient_sum = torch.sum(summed_gradients**2, dim=1)

    # Scatter add the squared sums back to their respective rows
    result = torch.zeros(nrows, device=device)
    unique_batch_ids = unique_paired_indices[:, 0].to(device)
    result.scatter_add_(0, unique_batch_ids, sqr_gradient_sum)

    # Compute the square root for the final result (norm)
    result_sqrt = torch.sqrt(result)
    return {layer.weight: result_sqrt}
