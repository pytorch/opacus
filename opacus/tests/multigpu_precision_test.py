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
from typing import Dict, Optional, Type

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import MixedPrecisionTest and models from mixed_precision_test.py
from opacus.tests.mixed_precision_test import (
    MixedPrecisionTest,
    SimpleLinearReluModel,
    SimpleLinearModel,
    Conv1DModel,
    Conv2DModel,
    Conv3DModel,
    RNNModel,
    AttentionModel,
    EmbeddingModel,
    ComplexModel,
)

# Import setup and cleanup functions from multigpu_gradcheck.py
from opacus.tests.multigpu_gradcheck import setup, cleanup


def run_mixed_precision_distributed_test(
    rank,
    world_size,
    model_class,
    batch_size,
    dtype,
    grad_sample_mode,
    weight,
    model_kwargs=None,
):
    """
    Run a mixed precision distributed test on a specific rank.

    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        model_class: The model class to instantiate
        batch_size: Batch size for training
        dtype: Data type for mixed precision (torch.float16 or torch.bfloat16)
        grad_sample_mode: Gradient sample mode ("hooks", "functorch", "ew", or "ghost")
        weight: Shared tensor to store model weights for verification
        model_kwargs: Keyword arguments for model instantiation
    """
    if model_kwargs is None:
        model_kwargs = {}

    torch.manual_seed(world_size)
    setup(rank, world_size)

    # Create a test instance to access MixedPrecisionTest methods
    test_instance = MixedPrecisionTest()
    test_instance.device = torch.device(f"cuda:{rank}")
    test_instance.batch_size = batch_size
    test_instance.num_batches = 2
    test_instance.input_dim = 4
    test_instance.hidden_dim = 8
    test_instance.output_dim = 4
    test_instance.seq_len = 4

    # Create model
    model = model_class(**model_kwargs)

    # Determine model type for data creation
    model_type = "linear"
    if isinstance(model, SimpleLinearReluModel):
        model_type = "simple_linear"
    elif isinstance(model, SimpleLinearModel):
        model_type = "linear"
    elif isinstance(model, Conv1DModel):
        model_type = "conv1d"
    elif isinstance(model, Conv2DModel):
        model_type = "conv2d"
    elif isinstance(model, Conv3DModel):
        model_type = "conv3d"
    elif isinstance(model, RNNModel):
        model_type = "rnn"
    elif isinstance(model, AttentionModel):
        model_type = "attention"
    elif isinstance(model, EmbeddingModel):
        model_type = "embedding"
    elif isinstance(model, ComplexModel):
        model_type = "complex"

    # Create data using MixedPrecisionTest method
    dataloader, _, _ = test_instance._create_random_data(model_type)

    # Create distributed sampler
    dataset = dataloader.dataset
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Get training components using MixedPrecisionTest method
    model, optimizer, criterion, _, _ = test_instance._get_training_components(
        model, data_loader, grad_sample_mode, torch.float32
    )

    # Wrap model in DPDDP
    model = DPDDP(model)

    # Train for one epoch with mixed precision
    for x_batch, y_batch in data_loader:
        optimizer.zero_grad()

        # Use autocast for mixed precision
        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = model(x_batch)

            # Verify outputs are in the expected precision
            assert (
                outputs.dtype == dtype
            ), f"Expected output dtype {dtype}, got {outputs.dtype}"

            loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()

        # Verify gradients are in FP32 for mixed precision
        for p in model.parameters():
            if p.grad is not None:
                assert (
                    p.grad.dtype == torch.float32
                ), f"Expected grad dtype torch.float32, got {p.grad.dtype}"

        optimizer.step()
        break  # One batch is enough for testing

    # Store model weights for verification
    if hasattr(model.module, "net1"):
        weight.copy_(model.module.net1.weight.data.cpu())

    # Clean up
    cleanup()


def run_low_precision_distributed_test(
    rank,
    world_size,
    model_class,
    batch_size,
    dtype,
    grad_sample_mode,
    weight,
    model_kwargs=None,
):
    """
    Run a low precision distributed test on a specific rank.

    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        model_class: The model class to instantiate
        batch_size: Batch size for training
        dtype: Data type for low precision (torch.float16 or torch.bfloat16)
        grad_sample_mode: Gradient sample mode ("hooks", "functorch", "ew", or "ghost")
        weight: Shared tensor to store model weights for verification
        model_kwargs: Keyword arguments for model instantiation
    """
    if model_kwargs is None:
        model_kwargs = {}

    torch.manual_seed(world_size)
    setup(rank, world_size)

    # Create a test instance to access MixedPrecisionTest methods
    test_instance = MixedPrecisionTest()
    test_instance.device = torch.device(f"cuda:{rank}")
    test_instance.batch_size = batch_size
    test_instance.num_batches = 2
    test_instance.input_dim = 4
    test_instance.hidden_dim = 8
    test_instance.output_dim = 4
    test_instance.seq_len = 4

    # Create model and convert to low precision
    model = model_class(**model_kwargs).to(dtype)

    # Determine model type for data creation
    model_type = "linear"
    if isinstance(model, SimpleLinearReluModel):
        model_type = "simple_linear"
    elif isinstance(model, SimpleLinearModel):
        model_type = "linear"
    elif isinstance(model, Conv1DModel):
        model_type = "conv1d"
    elif isinstance(model, Conv2DModel):
        model_type = "conv2d"
    elif isinstance(model, Conv3DModel):
        model_type = "conv3d"
    elif isinstance(model, RNNModel):
        model_type = "rnn"
    elif isinstance(model, AttentionModel):
        model_type = "attention"
    elif isinstance(model, EmbeddingModel):
        model_type = "embedding"
    elif isinstance(model, ComplexModel):
        model_type = "complex"

    # Create data using MixedPrecisionTest method
    _, x, y = test_instance._create_random_data(model_type)

    # Convert input to low precision if it's a floating point tensor
    if x.is_floating_point():
        x = x.to(dtype)

    # Create dataset with low precision data
    dataset = torch.utils.data.TensorDataset(x, y)

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Get training components using MixedPrecisionTest method
    model, optimizer, criterion, _, _ = test_instance._get_training_components(
        model, data_loader, grad_sample_mode, dtype
    )

    # Wrap model in DPDDP
    model = DPDDP(model)

    # Train for one epoch with low precision
    for x_batch, y_batch in data_loader:
        optimizer.zero_grad()

        outputs = model(x_batch)

        # Verify outputs are in the expected precision
        assert (
            outputs.dtype == dtype
        ), f"Expected output dtype {dtype}, got {outputs.dtype}"

        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()

        # Verify gradients are in the same low precision
        for p in model.parameters():
            if p.grad is not None:
                assert (
                    p.grad.dtype == dtype
                ), f"Expected grad dtype {dtype}, got {p.grad.dtype}"

        optimizer.step()
        break  # One batch is enough for testing

    # Store model weights for verification
    if hasattr(model.module, "net1"):
        weight.copy_(model.module.net1.weight.data.cpu())

    # Clean up
    cleanup()


def run_distributed_test(test_fn, world_size, *args):
    """
    Helper function to run a distributed test across multiple processes.

    Args:
        test_fn: The test function to run
        world_size: Number of processes to spawn
        args: Arguments to pass to the test function
    """
    mp.spawn(
        test_fn,
        args=(world_size, *args),
        nprocs=world_size,
        join=True,
    )


class MultiGPUPrecisionTest(MixedPrecisionTest):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def setUp(self):
        self.batch_size = 4
        self.world_size = min(2, torch.cuda.device_count())  # Use at most 2 GPUs

        # Check if bfloat16 is supported
        self.bf16_supported = hasattr(torch, "bfloat16")

        # Model parameters
        self.input_dim = 4
        self.hidden_dim = 8
        self.output_dim = 4

    def _test_precision_training(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Optional[Dict] = None,
    ):
        """Test training a model with different precision types in a distributed setting"""
        if model_kwargs is None:
            model_kwargs = {}

        # Create model instance for testing
        model = model_class(**model_kwargs)

        # Test with low precision (FP16)
        for grad_sample_mode in ["hooks", "functorch", "ew", "ghost"]:
            # Skip unsupported combinations
            if grad_sample_mode == "ew" and isinstance(
                model, (SimpleLinearModel, EmbeddingModel, ComplexModel)
            ):
                continue

            # Test FP16
            weight_fp16 = torch.zeros_like(next(model.parameters()))
            print(
                f"Testing {model_class.__name__} with FP16 and grad sample mode {grad_sample_mode}"
            )
            run_distributed_test(
                run_low_precision_distributed_test,
                self.world_size,
                model_class(**model_kwargs),
                self.batch_size,
                torch.float16,
                grad_sample_mode,
                weight_fp16,
            )

            # Test BF16 if supported
            if self.bf16_supported:
                weight_bf16 = torch.zeros_like(next(model.parameters()))
                print(
                    f"Testing {model_class.__name__} with BF16 and grad sample mode {grad_sample_mode}"
                )
                run_distributed_test(
                    run_low_precision_distributed_test,
                    self.world_size,
                    model_class(**model_kwargs),
                    self.batch_size,
                    torch.bfloat16,
                    grad_sample_mode,
                    weight_bf16,
                )

        # Test mixed precision (only for hooks and ghost modes)
        for grad_sample_mode in ["hooks", "ghost"]:
            # Test mixed FP32 + FP16
            weight_mixed_fp16 = torch.zeros_like(next(model.parameters()))
            print(
                f"Testing {model_class.__name__} with mixed FP32+FP16 and grad sample mode {grad_sample_mode}"
            )
            run_distributed_test(
                run_mixed_precision_distributed_test,
                self.world_size,
                model_class(**model_kwargs),
                self.batch_size,
                torch.float16,
                grad_sample_mode,
                weight_mixed_fp16,
            )

            # Test mixed FP32 + BF16 if supported
            if self.bf16_supported:
                weight_mixed_bf16 = torch.zeros_like(next(model.parameters()))
                print(
                    f"Testing {model_class.__name__} with mixed FP32+BF16 and grad sample mode {grad_sample_mode}"
                )
                run_distributed_test(
                    run_mixed_precision_distributed_test,
                    self.world_size,
                    model_class(**model_kwargs),
                    self.batch_size,
                    torch.bfloat16,
                    grad_sample_mode,
                    weight_mixed_bf16,
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_simple_linear_model_precision(self):
        """Test SimpleLinearModel with different precision settings"""
        self._test_precision_training(
            model_class=SimpleLinearModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_simple_linear_relu_model_precision(self):
        """Test SimpleLinearReluModel with different precision settings"""
        self._test_precision_training(
            model_class=SimpleLinearReluModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_conv1d_model_precision(self):
        """Test Conv1DModel with different precision settings"""
        self._test_precision_training(
            model_class=Conv1DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_conv2d_model_precision(self):
        """Test Conv2DModel with different precision settings"""
        self._test_precision_training(
            model_class=Conv2DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_conv3d_model_precision(self):
        """Test Conv3DModel with different precision settings"""
        self._test_precision_training(
            model_class=Conv3DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_rnn_model_precision(self):
        """Test RNNModel with different precision settings"""
        self._test_precision_training(
            model_class=RNNModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_attention_model_precision(self):
        """Test AttentionModel with different precision settings"""
        self._test_precision_training(
            model_class=AttentionModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_heads": 4,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_embedding_model_precision(self):
        """Test EmbeddingModel with different precision settings"""
        self._test_precision_training(
            model_class=EmbeddingModel,
            model_kwargs={
                "vocab_size": 100,
                "embedding_dim": self.hidden_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 GPUs, skipping test"
    )
    def test_complex_model_precision(self):
        """Test ComplexModel with different precision settings"""
        self._test_precision_training(
            model_class=ComplexModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "seq_len": 4,
            },
        )


if __name__ == "__main__":
    unittest.main()
