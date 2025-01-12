#!/usr/bin/env python3
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

import unittest

import torch
import torch.nn as nn
from opacus.grad_sample import embedding_norm_sample


class TestComputeEmbeddingNormSample(unittest.TestCase):

    def test_compute_embedding_norm_sample(self):
        # Define the embedding layer
        embedding_dim = 1
        vocab_size = 3
        embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # Manually set weights for the embedding layer for testing
        embedding_layer.weight = nn.Parameter(
            torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
        )

        # Example input ids (activations). Shape: [3, 2]
        input_ids = torch.tensor([[1, 1], [2, 0], [2, 0]], dtype=torch.long)

        # Example backprops. Shape: [3, 2, 1]
        backprops = torch.tensor(
            [[[0.2], [0.2]], [[0.3], [0.1]], [[0.3], [0.1]]], dtype=torch.float32
        )

        # Wrap input_ids in a list as expected by the norm sample function
        activations = [input_ids]

        # Call the function under test
        result = embedding_norm_sample.compute_embedding_norm_sample(
            embedding_layer, activations, backprops
        )

        # Expected norms
        expected_norms = torch.tensor([0.4000, 0.3162, 0.3162], dtype=torch.float32)

        # Extract the result for the embedding layer weight parameter
        computed_norms = result[embedding_layer.weight]

        # Verify the computed norms match the expected norms
        torch.testing.assert_close(computed_norms, expected_norms, atol=1e-4, rtol=1e-4)

    def test_compute_embedding_norm_sample_with_non_one_embedding_dim(self):
        # Define the embedding layer
        embedding_dim = 2
        vocab_size = 3
        embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # Manually set weights for the embedding layer for testing
        embedding_layer.weight = nn.Parameter(
            torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)
        )

        # Example input ids (activations). Shape: [6, 1, 1].
        input_ids = torch.tensor(
            [[[1]], [[1]], [[2]], [[0]], [[2]], [[0]]], dtype=torch.long
        )

        # Example backprops per input id, with embedding_dim=2.
        # Shape: [6, 1, 1, 2]
        backprops = torch.tensor(
            [
                [[[0.2, 0.2]]],
                [[[0.2, 0.2]]],
                [[[0.3, 0.3]]],
                [[[0.1, 0.1]]],
                [[[0.3, 0.3]]],
                [[[0.1, 0.1]]],
            ],
            dtype=torch.float32,
        )

        # Wrap input_ids in a list as expected by the grad norm function
        activations = [input_ids]

        # Call the function under test
        result = embedding_norm_sample.compute_embedding_norm_sample(
            embedding_layer, activations, backprops
        )

        # Expected output based on the example
        expected_norms = torch.tensor(
            [0.2828, 0.2828, 0.4243, 0.1414, 0.4243, 0.1414], dtype=torch.float32
        )

        # Extract the result for the embedding layer weight parameter
        computed_norms = result[embedding_layer.weight]

        # Verify the computed norms match the expected norms
        torch.testing.assert_close(computed_norms, expected_norms, atol=1e-4, rtol=1e-4)

    def test_compute_embedding_norm_sample_with_extra_activations_per_example(self):
        # Define the embedding layer
        embedding_dim = 1
        vocab_size = 10
        embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # Manually set weights for the embedding layer for testing
        embedding_layer.weight = nn.Parameter(
            torch.tensor(
                [
                    [0.1],
                    [0.2],
                    [0.3],
                    [0.4],
                    [0.5],
                    [0.6],
                    [0.7],
                    [0.8],
                    [0.9],
                    [1.0],
                ],
                dtype=torch.float32,
            )
        )

        # Example input ids with 6 activations per sample, shape: [5, 6, 1]
        input_ids = torch.tensor(
            [
                [[0], [0], [0], [0], [0], [0]],
                [[1], [0], [0], [0], [0], [0]],
                [[2], [3], [4], [5], [6], [7]],
                [[4], [3], [0], [0], [0], [0]],
                [[8], [7], [9], [6], [5], [0]],
            ],
            dtype=torch.long,
        )

        # Example gradients per input id, with embedding_dim=1.
        # Shape: [5, 6, 1, 1]
        backprops = torch.tensor(
            [
                [
                    [[0.0025]],
                    [[0.0025]],
                    [[0.0025]],
                    [[0.0025]],
                    [[0.0025]],
                    [[0.0025]],
                ],
                [
                    [[-0.0014]],
                    [[-0.0014]],
                    [[-0.0014]],
                    [[-0.0014]],
                    [[-0.0014]],
                    [[-0.0014]],
                ],
                [
                    [[-0.0002]],
                    [[-0.0002]],
                    [[-0.0002]],
                    [[-0.0002]],
                    [[-0.0002]],
                    [[-0.0002]],
                ],
                [
                    [[0.0019]],
                    [[0.0019]],
                    [[0.0019]],
                    [[0.0019]],
                    [[0.0019]],
                    [[0.0019]],
                ],
                [
                    [[-0.0016]],
                    [[-0.0016]],
                    [[-0.0016]],
                    [[-0.0016]],
                    [[-0.0016]],
                    [[-0.0016]],
                ],
            ],
            dtype=torch.float32,
        )

        # Wrap input_ids in a list as expected by the function
        activations = [input_ids]

        # Call the function we want to test
        result = embedding_norm_sample.compute_embedding_norm_sample(
            embedding_layer, activations, backprops
        )

        # Expected output based on the example
        expected_norms = torch.tensor(
            [0.0150, 0.0071, 0.0005, 0.0081, 0.0039], dtype=torch.float32
        )
        computed_norms = result[embedding_layer.weight]

        # Verify the computed norms match the expected norms
        torch.testing.assert_close(computed_norms, expected_norms, atol=1e-4, rtol=1e-4)
