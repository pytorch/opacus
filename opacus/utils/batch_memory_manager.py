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
    """
    Samples according to the underlying instance of ``Sampler``, but splits
    the index sequences into smaller chunks.

    Used to split large logical batches into physocal batches of a smaller size,
    while coordinating with DPOptimizer when the logical batch has ended.
    """

    def __init__(
        self,
        *,
        sampler: Sampler[List[int]],
        max_batch_size: int,
        optimizer: DPOptimizer,
    ):
        """

        Args:
            sampler: Wrapped Sampler instance
            max_batch_size: Max size of emitted chunk of indices
            optimizer: optimizer instance to notify when the logical batch is over
        """
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


def wrap_data_loader(
    *, data_loader: DataLoader, max_batch_size: int, optimizer: DPOptimizer
):
    """
    Replaces batch_sampler in the input data loader with ``BatchSplittingSampler``

    Args:
        data_loader: Wrapper DataLoader
        max_batch_size: max physical batch size we want to emit
        optimizer: DPOptimizer instance used for training

    Returns:
        New DataLoader instance with batch_sampler wrapped in ``BatchSplittingSampler``
    """

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
    """
    Context manager to manage memory consumption during training.

    Allows setting hard limit on the physical batch size as a just one line code change.
    Can be used both for simulating large logical batches with limited memory and for
    safeguarding against occasinal large batches produced by
    :class:`~opacus.utils.uniform_sampler.UniformWithReplacementSampler`.

    Note that it doesn't modify the input DataLoader, you'd need to use new DataLoader
    returned by the context manager.

    BatchSplittingSampler will split large logical batches into smaller sub-batches with
    certain maximum size.
    On every step optimzer will check if the batch was the last physical batch comprising
    a logical one, and will change behaviour accordignly.

    If it was not the last, ``optimizer.step()`` will only clip per sample gradients and
    sum them into ``p.summed_grad`.` ``optimizeer.zero_grad()`` will clear ``p.grad_sample``,
    but will leave ``p.grad`` and ``p.summed_grad``

    If the batch was the last one of the current logical batch, then
    ``optimizer.step()`` and ``optimizer.zero_grad()`` will behave normally.

    Example:
        >>> # Assuming you've initialized you objects and passed them to PrivacyEngine.
        >>> # For this example we assume data_loader is initalized with batch_size=4
        >>> model, optimizer, data_loader = _init_private_training()
        >>> criterion = nn.CrossEntropyLoss()
        >>> with BatchMemoryManager(
        ...     data_loader=data_loader, max_physical_batch_size=2, optimizer=optimizer
        ... ) as new_data_loader:
        ...     for data, label in new_data_loader:
        ...         assert len(data) <= 2 # physical batch is no more than 2
        ...         output = model(data)
        ...         loss = criterion(output, label)
        ...         loss.backward()
        ...         # optimizer won't actually make a step unless logical batch is over
        ...         optimizer.step()
        ...         # optimizer won't actually clear gradients unless logical batch is over
        ...         optimizer.zero_grad()
    """

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
