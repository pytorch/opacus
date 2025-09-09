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

import copy
import logging
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus.grad_sample import GradSampleModuleFastGradientClippingTP
from opacus.optimizers import FSDPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from torch.distributed._tensor import Replicate
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import LlamaConfig, LlamaForCausalLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_LENGTH = 32000
OUTPUT_FEATURE_LENGTH = 4096
SEQ_LENGTH = 1
BATCH_SIZE = 1


def get_input_and_output():
    input = torch.randint(EMBEDDING_LENGTH, size=(BATCH_SIZE, SEQ_LENGTH))
    # labels = torch.randint(100, size = (BATCH_SIZE, SEQ_LENGTH))
    labels = torch.randn(size=(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_LENGTH))
    return input, labels


def profile_mem(f, arg1=None, karg=None):
    torch.cuda.reset_peak_memory_stats()
    m1_max = torch.cuda.max_memory_allocated() / 2**20
    m1 = torch.cuda.memory_allocated() / 2**20
    assert m1 == m1_max

    # if arg1 is not None and karg is not None:
    #     ret = f(arg1, inputs = karg)
    if arg1 is not None:
        ret = f(arg1)
    else:
        ret = f()

    m2_max_mem = torch.cuda.max_memory_allocated() / 2**20
    m2 = torch.cuda.memory_allocated() / 2**20
    print(f"Mem history: {m1} -> {m2_max_mem} -> {m2} MB\n")
    print(f"Max Mem difference: {m2_max_mem-m1} MB\n")
    return ret, m2_max_mem - m1


# pyre-ignore
def model_parallel(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print("current_local_rank is", rank)

    torch.cuda.set_device(rank)

    input, labels = get_input_and_output()

    # Define Llama model
    model_config = LlamaConfig()

    torch.cuda.reset_peak_memory_stats()

    llama_model = LlamaForCausalLM(model_config).to(device="cuda")

    random.seed(42)

    llama_model.init_weights()
    llama_model.train()

    torch.cuda.reset_peak_memory_stats()
    m1 = torch.cuda.memory_allocated() / 2**20

    # Define the tensor parallelization plan
    tp_mesh = init_device_mesh("cuda", (world_size,))
    for layer_id, transformer_block in enumerate(llama_model.model.layers):
        layer_tp_plan = {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            # "input_layernorm": SequenceParallel(),
            # "post_attention_layernorm": SequenceParallel(),
        }
        # Adjust attention module to use the local number of heads, or else there will be a dimension mismatch error
        attn_layer = transformer_block.self_attn
        attn_layer.num_heads = attn_layer.num_heads // tp_mesh.size()
        attn_layer.num_key_value_heads = (
            attn_layer.num_key_value_heads // tp_mesh.size()
        )

        # Freeze layernorm layers
        for name, param in transformer_block.input_layernorm.named_parameters():
            param.requires_grad = False
        for (
            name,
            param,
        ) in transformer_block.post_attention_layernorm.named_parameters():
            param.requires_grad = False

        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    llama_model = parallelize_module(
        llama_model,
        tp_mesh,
        {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            # "model.norm": SequenceParallel(),
            "lm_head": RowwiseParallel(input_layouts=Replicate()),
        },
    )

    # Freeze the layernorm layers
    # for name, param in llama_model.model.embed_tokens.named_parameters():
    #     param.requires_grad = True
    for name, param in llama_model.model.norm.named_parameters():
        param.requires_grad = False

    print("model architecture", llama_model)

    m2_max_mem = torch.cuda.max_memory_allocated() / 2**20
    m2 = torch.cuda.memory_allocated() / 2**20
    print(f"Mem history: {m1} -> {m2_max_mem} -> {m2} MB\n")
    print(f"Max Mem difference: {m2_max_mem-m1} MB\n")

    DP_sharded_model = GradSampleModuleFastGradientClippingTP(
        llama_model, loss_reduction="mean", batch_first=True
    )

    optimizer_gc = torch.optim.SGD(DP_sharded_model.parameters(), lr=1)
    optimizer_gc = FSDPOptimizerFastGradientClipping(
        optimizer_gc,
        noise_multiplier=0.0,
        max_grad_norm=0.1,
        expected_batch_size=1,
        loss_reduction="mean",
    )

    criterion_gc = torch.nn.CrossEntropyLoss(reduction="mean")
    criterion_gc = DPLossFastGradientClipping(
        DP_sharded_model, optimizer_gc, copy.deepcopy(criterion_gc)
    )
    optimizer_gc.zero_grad()

    print("forward pass, memory usage")
    output, temp = profile_mem(DP_sharded_model, input.to(device="cuda"))

    loss = criterion_gc(output.logits.view(1, -1), labels.to(device="cuda").view(1, -1))

    print("Backward pass, memory usage")
    profile_mem(loss.backward)

    print("Optimizer step, memory usage")
    profile_mem(optimizer_gc.step)

    print("zero gradient, memory usage")
    profile_mem(optimizer_gc.zero_grad, True)


def main():

    n_gpus = torch.cuda.device_count()
    print(n_gpus)
    print(torch.version.cuda)
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(model_parallel, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
