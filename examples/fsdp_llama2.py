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

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.utils.fsdp_utils import FSDP2Wrapper
from peft import LoraConfig, TaskType, get_peft_model  # LORA
from torch.utils.data import DataLoader, DistributedSampler  # , RandomSampler
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM


#########################################################################


def is_file_readable(file_path):
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# Create a dummy dataset for LlamaForCausalLM
class DummyLlamaDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        label = torch.randn((self.seq_length, self.vocab_size))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }


def train_step(model, optimizer, criterion, batch, device):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.logits.view(1, -1), labels.view(1, -1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def prepare_model(model_config, is_lora, lora_rank):
    llama_model = LlamaForCausalLM(model_config)

    if is_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Causal language modeling for GPT-style models
            inference_mode=False,  # Enable training mode
            r=lora_rank,  # Low-rank dimension
            lora_alpha=32,  # Alpha scaling factor
            lora_dropout=0.05,  # Dropout for LoRA layers
        )
        model_with_lora = get_peft_model(llama_model, lora_config)

    trainable_parameters = 0
    if is_lora:
        for name, param in model_with_lora.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False  # opacus doesn't support positional embedding layer and GC doesn't support tyed parameters
            if param.requires_grad:
                trainable_parameters += param.numel()
    else:
        for name, param in llama_model.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False  # opacus doesn't support positional embedding layer and GC doesn't support tyed parameters
            if param.requires_grad:
                trainable_parameters += param.numel()

    print(f"Trainable parameters: {trainable_parameters}")
    if is_lora:
        return model_with_lora
    else:
        return llama_model


# training loop
def train(
    rank: int,
    world_size: int,
    device: torch.device,
    is_lora: bool = True,
    lora_rank: int = 16,
    seq_length: int = 128,
    batch_size: int = 32,
    max_physical_batch_size: int = 1,
    learning_rate: float = 1e-5,
    sigma: float = 1,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    mp_policy: dist.fsdp.MixedPrecisionPolicy = None,
):
    model_config = LlamaConfig()
    model_final = prepare_model(model_config, is_lora, lora_rank)

    # dataset
    vocab_size = (
        model_config.vocab_size if hasattr(model_config, "vocab_size") else 32000
    )
    train_dataset = DummyLlamaDataset(
        num_samples=512, seq_length=seq_length, vocab_size=vocab_size
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,
        sampler=DistributedSampler(train_dataset),
    )

    # wrap model into FSDP container
    model = FSDP2Wrapper(model_final, mp_policy=mp_policy)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, train_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="ghost_fsdp",
        criterion=torch.nn.CrossEntropyLoss(ignore_index=-1),
        poisson_sampling=False,
    )

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        with BatchMemoryManager(
            data_loader=train_dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for _, batch in tqdm(
                enumerate(memory_safe_data_loader), desc=f"Training epoch {epoch}: "
            ):
                train_step(model, optimizer, criterion, batch, device)
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage: {max_memory_allocated / 1024**3:.2f} GB on rank {rank}")


def launch(
    rank: int,
    world_size: int,
    batch_size: int = 32,
    max_physical_batch_size: int = 4,
    seq_length: int = 128,
    is_lora: bool = True,
    lora_rank: int = 8,
    learning_rate: float = 1e-5,
    sigma: float = 1.0,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    mp_policy: dist.fsdp.MixedPrecisionPolicy = None,
):
    # set the device for the current process
    torch.cuda.set_device(rank)
    # setup environment for distributed training
    setup(rank, world_size)
    # -----------------------------------------------------------------------------
    master_process = rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = rank  # each process gets a different seed
    # -----------------------------------------------------------------------------

    tokens_per_iter = batch_size * seq_length
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs("/tmp/out", exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    train(
        rank,
        world_size,
        device=torch.device(f"cuda:{rank}"),
        seq_length=seq_length,
        batch_size=batch_size,
        max_physical_batch_size=max_physical_batch_size,
        lora_rank=lora_rank,
        is_lora=is_lora,
        learning_rate=learning_rate,
        sigma=sigma,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        mp_policy=mp_policy,
    )
    # cleanup process group
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Distributed Llama Training Arguments")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per process"
    )
    parser.add_argument(
        "--max_physical_batch_size", type=int, default=4, help="Max physical batch size"
    )
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--is_lora", type=bool, default=False, help="Use LoRA fine-tuning"
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Noise multiplier for DP"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max grad norm for DP"
    )
    parser.add_argument(
        "--enable_mixed_precision",
        type=bool,
        default=True,
        help="enable mixed precision with bf16",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(
        f"Is cuda available: {torch.cuda.is_available()}, number of devices: {world_size}"
    )
    if torch.cuda.current_device() == 0:
        print(f"Args: {args}")
    if args.enable_mixed_precision:
        mp_policy = dist.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None

    args = (
        world_size,
        args.batch_size,
        args.max_physical_batch_size,
        args.seq_length,
        args.is_lora,
        args.lora_rank,
        args.learning_rate,
        args.sigma,
        args.max_grad_norm,
        args.epochs,
        mp_policy,
    )
    mp.spawn(
        launch,
        args=args,
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
