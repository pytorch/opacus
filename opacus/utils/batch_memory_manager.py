import math
from typing import List

import numpy as np
from opacus.optimizers import DPOptimizer
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import BatchSampler, DataLoader, Sampler


class BatchSplittingSampler(Sampler[List[int]]):
    def __init__(
        self,
        *,
        sampler: Sampler[List[int]],
        max_batch_size: int,
        optimizer: DPOptimizer,
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
                self.optimizer.signal_skip_step(do_skip=True)
                yield x
            self.optimizer.signal_skip_step(do_skip=False)
            yield split_idxs[-1]

    def __len__(self):
        if isinstance(self.sampler, BatchSampler):
            return int(
                len(self.sampler) * (self.sampler.batch_size / self.max_batch_size)
            )
        elif isinstance(self.sampler, UniformWithReplacementSampler) or isinstance(
            self.sampler, DistributedUniformWithReplacementSampler
        ):
            expected_batch_size = self.sampler.sample_rate * self.sampler.num_samples
            return int(len(self.sampler) * (expected_batch_size / self.max_batch_size))

        return len(self.sampler)


def wrap_data_loader(*, data_loader, max_batch_size: int, optimizer: DPOptimizer):
    return DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=BatchSplittingSampler(
            sampler=data_loader.batch_sampler,
            max_batch_size=max_batch_size,
            optimizer=optimizer,
        ),
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
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
        *,
        data_loader: DataLoader,
        max_physical_batch_size: int,
        optimizer: DPOptimizer,
    ):
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.max_physical_batch_size = max_physical_batch_size

    def __enter__(self):
        return wrap_data_loader(
            data_loader=self.data_loader,
            max_batch_size=self.max_physical_batch_size,
            optimizer=self.optimizer,
        )

    def __exit__(self, type, value, traceback):
        pass
