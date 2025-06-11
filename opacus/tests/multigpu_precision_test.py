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

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from .mixed_precision_test import (
    EmbeddingModel,
    SimpleLinearReluModel,
    create_random_data,
)


def setup(rank, world_size):
    if sys.platform == "win32":
        raise ValueError("Windows platform is not supported for this test")
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


def cleanup():
    dist.destroy_process_group()


def _get_training_components(
    model_class,
    model_kwargs,
    device,
    grad_sample_mode,
):
    input_dim = model_kwargs.get("input_dim", 4)
    output_dim = model_kwargs.get("output_dim", 4)
    seq_len = model_kwargs.get("seq_len", 4)
    batch_size = 2
    num_batches = 2

    dataloader, _ = create_random_data(
        model_class,
        batch_size=batch_size,
        input_dim=input_dim,
        output_dim=output_dim,
        num_batches=num_batches,
        seq_len=seq_len,
        device=device,
    )

    model = model_class(**model_kwargs).to(device)
    model = model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    model = DPDDP(model)

    if grad_sample_mode in ["hooks", "functorch", "ew"]:
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=1,
            max_grad_norm=1,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=False,
        )
    elif grad_sample_mode == "ghost":
        model, optimizer, criterion, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            criterion=criterion,
            max_grad_norm=1,
            noise_multiplier=1,
            grad_sample_mode="ghost",
            poisson_sampling=False,
        )

    return model, optimizer, criterion, dataloader, privacy_engine, output_dim


def run_mixed_precision_test(
    rank,
    world_size,
    model_class,
    model_kwargs,
    dtype,
    grad_sample_mode,
):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model, optimizer, criterion, dataloader, privacy_engine, output_dim = (
        _get_training_components(model_class, model_kwargs, device, grad_sample_mode)
    )

    # Model weights should be in high precision (fp32)
    model = model.to(torch.float32)
    for p in model.parameters():
        assert p.dtype == torch.float32

    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = model(x)
            assert outputs.dtype == dtype

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()

        # The gradients should have been cast up to high precision (fp32)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.dtype == torch.float32

        optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    assert epsilon > 0

    model.eval()
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=dtype):
        output = model(x)
    assert output.shape[1] == output_dim

    cleanup()


def run_low_precision_test(
    rank,
    world_size,
    model_class,
    model_kwargs,
    dtype,
    grad_sample_mode,
):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model, optimizer, criterion, dataloader, privacy_engine, output_dim = (
        _get_training_components(model_class, model_kwargs, device, grad_sample_mode)
    )

    # Model weights should be in low precision
    model = model.to(dtype)
    for p in model.parameters():
        assert p.dtype == dtype

    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad()

        if x.is_floating_point():  # For embedding layers, keep input as int
            x = x.to(dtype)
        outputs = model(x)
        assert outputs.dtype == dtype

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.dtype == dtype

        if grad_sample_mode == "ghost":
            per_sample_norms = model.get_norm_sample()
            assert per_sample_norms.dtype == dtype

        optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    assert epsilon > 0

    model.eval()
    optimizer.zero_grad()
    output = model(x)
    assert output.shape[1] == output_dim

    cleanup()


def run_distributed_test(
    model_class,
    model_kwargs,
    dtype,
    grad_sample_mode,
    mixed_precision,
    world_size=2,
):
    if mixed_precision:
        mp.spawn(
            run_mixed_precision_test,
            args=(world_size, model_class, model_kwargs, dtype, grad_sample_mode),
            nprocs=world_size,
            join=True,
        )
    else:
        mp.spawn(
            run_low_precision_test,
            args=(world_size, model_class, model_kwargs, dtype, grad_sample_mode),
            nprocs=world_size,
            join=True,
        )


class MultiGPUPrecisionTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def setUp(self):
        self.input_dim = 4
        self.hidden_dim = 16
        self.output_dim = 4
        self.seq_len = 4
        self.bf16_supported = hasattr(torch, "bfloat16")

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_precision_training(
        self,
    ):

        model_kwargs_map = {
            SimpleLinearReluModel: {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
            EmbeddingModel: {
                "vocab_size": 100,
                "embedding_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        }

        # test models sequentially since running tests in parallel fails for DDP
        for model_class in [SimpleLinearReluModel, EmbeddingModel]:

            model_kwargs = model_kwargs_map[model_class]
            dtype = torch.bfloat16 if self.bf16_supported else torch.float16

            # test low precision training
            for grad_sample_mode in ["ew", "functorch", "hooks", "ghost"]:
                if grad_sample_mode == "ew" and model_class == EmbeddingModel:
                    continue
                run_distributed_test(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    dtype=dtype,
                    grad_sample_mode=grad_sample_mode,
                    mixed_precision=False,
                )

            # test mixed precision training
            for grad_sample_mode in ["functorch", "hooks", "ghost"]:
                run_distributed_test(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    dtype=dtype,
                    grad_sample_mode=grad_sample_mode,
                    mixed_precision=True,
                )


if __name__ == "__main__":
    unittest.main()
