# Common imports
import os
import random
import sys
from pathlib import Path
from typing import *

import numpy as np
import opacus
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm


BATCH_SZ = 2
FEATURE_DIM = 16
N_CLASSES = 3
SEED = 30
DATASET_LEN = 10000

DEVICE = torch.device("cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

rng = torch.Generator(DEVICE)
rng.manual_seed(SEED)


class DummyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, i):
        x = torch.randn(
            [FEATURE_DIM]
        )  # for categoricals like nn.Embedding you can do something like `torch.randint(0, VOC_SZ-1, (SEQ_LEN,), dtype=torch.long).to(device)`
        y = 0
        return x, y

    def __len__(self):
        return DATASET_LEN


train_ds = DummyDataset()
train_loader = DataLoader(train_ds, BATCH_SZ, shuffle=True, generator=rng)

model = nn.Linear(FEATURE_DIM, N_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

privacy_engine = opacus.PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=0.9,
    max_grad_norm=1.0,
)

from opacus.utils.batch_memory_manager import wrap_data_loader


train_loader = wrap_data_loader(
    data_loader=train_loader, max_batch_size=BATCH_SZ, optimizer=optimizer
)

train_iter = iter(train_loader)
batch_idx = 0
for batch in train_iter:
    print(f"batch {batch_idx}")
    print(batch)
    batch_idx += 1
