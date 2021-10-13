import math
from contextlib import contextmanager
from typing import List

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.optimizer import DPOptimizer
from torch.utils.data import BatchSampler, DataLoader, Sampler, TensorDataset


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(x)


class BatchSplittingSampler(Sampler[List[int]]):
    def __init__(
        self, sampler: Sampler[List[int]], max_batch_size: int, optimizer: DPOptimizer
    ):
        self.sampler = sampler
        self.max_batch_size = max_batch_size
        self.optimizer = optimizer

    def __iter__(self):
        for batch_idxs in self.sampler:
            split_idxs = np.array_split(
                batch_idxs, math.ceil(len(batch_idxs) / self.max_batch_size)
            )
            for x in split_idxs[:-1]:
                self.optimizer.skip_next_step()
                yield x
            yield split_idxs[-1]

    def __len__(self):
        return len(self.sampler)


def wrap_data_loader(data_loader, max_batch_size: int, optimizer: DPOptimizer):
    return DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=BatchSplittingSampler(
            data_loader.batch_sampler, max_batch_size, optimizer
        ),
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        drop_last=data_loader.drop_last,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=data_loader.generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )


class BatchMemoryManager(object):
    def __init__(
        self,
        data_loader: DataLoader,
        max_physical_batch_size: int,
        optimizer: DPOptimizer,
    ):
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.max_physical_batch_size = max_physical_batch_size

    def __enter__(self):
        return wrap_data_loader(
            self.data_loader, self.max_physical_batch_size, self.optimizer
        )

    def __exit__(self, type, value, traceback):
        pass


inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10, dtype=torch.float32).view(
    10,
)

if __name__ == "__main__":
    dataset = TensorDataset(inps, tgts)
    data_loader = DataLoader(dataset, batch_size=4, pin_memory=True)
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    privacy_engine = PrivacyEngine()
    model, optimizer, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    with BatchMemoryManager(
        data_loader=data_loader, max_physical_batch_size=2, optimizer=optimizer
    ) as data_loader2:
        for x, y in data_loader2:
            # print(x.shape)
            out = model(x)
            loss = (y - out).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # data_loader.batch_sampler
