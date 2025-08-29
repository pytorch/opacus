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
from datasets import load_dataset
from huggingface_hub import login  # used to download from HuggingFace
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.utils.fsdp_utils import FSDP2Wrapper
from peft import LoraConfig, TaskType, get_peft_model  # LORA
from torch.utils.data import DataLoader, DistributedSampler  # , RandomSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def is_file_readable(file_path):
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_snli_dataset(tokenizer, split="train", max_len=128):
    dataset = load_dataset("snli", split=split)
    dataset = dataset.filter(lambda example: example["label"] != -1)

    def tokenize_function(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

    encoded_dataset = dataset.map(
        tokenize_function, batched=True
    )  # , keep_in_memory=True)
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return encoded_dataset


def prepare_model(
    token: str,
    is_lora: bool = True,
    lora_rank: int = 16,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
):
    login(token)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    if is_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Causal language modeling for GPT-style models
            inference_mode=False,  # Enable training mode
            r=lora_rank,  # Low-rank dimension
            lora_alpha=32,  # Alpha scaling factor
            lora_dropout=0.05,  # Dropout for LoRA layers
        )
        model_with_lora = get_peft_model(pretrained_model, lora_config)

    trainable_parameters = 0
    if is_lora:
        for name, param in model_with_lora.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False  # opacus doesn't support positional embedding layer and GC doesn't support tyed parameters
            if param.requires_grad:
                trainable_parameters += param.numel()

    else:
        for name, param in pretrained_model.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False  # opacus doesn't support positional embedding layer and GC doesn't support tyed parameters
            if param.requires_grad:
                trainable_parameters += param.numel()

    print(f"Trainable parameters: {trainable_parameters}")
    if is_lora:
        return model_with_lora, tokenizer
    else:
        return pretrained_model, tokenizer


def train_step(model, optimizer, criterion, batch, device):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


# training loop
def train(
    token: str,
    master_process: bool,
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
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    mp_policy: dist.fsdp.MixedPrecisionPolicy = None,
):
    assert (
        token is not None
    ), "Please provide a valid huggingface token to access gated models"

    model_final, tokenizer = prepare_model(token, is_lora, lora_rank, model_name)
    # dataset
    train_dataset = prepare_snli_dataset(tokenizer, split="train", max_len=seq_length)

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
        criterion=torch.nn.CrossEntropyLoss(),
        poisson_sampling=False,
    )

    for epoch in range(1, epochs + 1):
        with BatchMemoryManager(
            data_loader=train_dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for step, batch in tqdm(
                enumerate(memory_safe_data_loader), desc=f"Training epoch {epoch}: "
            ):
                loss = train_step(model, optimizer, criterion, batch, device)
                if master_process:
                    print(f"Step: {step}, Loss: {loss.item()}")
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage: {max_memory_allocated / 1024**3:.2f} GB on rank {rank}")


def launch(
    rank: int,
    world_size: int,
    token: str,
    batch_size: int = 32,
    max_physical_batch_size: int = 4,
    seq_length: int = 128,
    is_lora: bool = True,
    lora_rank: int = 8,
    learning_rate: float = 1e-5,
    sigma: float = 1.0,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    mp_policy: dist.fsdp.MixedPrecisionPolicy = None,
):
    # set the device for the current process
    torch.cuda.set_device(rank)
    # setup environment for distributed training
    setup(rank, world_size)
    master_process = rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = rank  # each process gets a different seed

    tokens_per_iter = batch_size * seq_length
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs("/tmp/out", exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    train(
        token,
        master_process,
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
        model_name=model_name,
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
    parser.add_argument("--is_lora", type=bool, default=False, help="Use LoRA")
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
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",  # "meta-llama/Llama-3.2-1B"
        help="Model name",
    )
    parser.add_argument(
        "--enable_mixed_precision",
        type=bool,
        default=True,
        help="enable mixed precision with bf16",
    )
    parser.add_argument(
        "--token", type=str, default=None, help="Huggingface token", required=True
    )
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
        )
    else:
        mp_policy = None

    args = (
        world_size,
        args.token,
        args.batch_size,
        args.max_physical_batch_size,
        args.seq_length,
        args.is_lora,
        args.lora_rank,
        args.learning_rate,
        args.sigma,
        args.max_grad_norm,
        args.epochs,
        args.model_name,
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
