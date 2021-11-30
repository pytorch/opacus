from typing import Any, Optional, Sequence

import torch
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t


def wrap_collate_with_empty(
    collate_fn: Optional[_collate_fn_t], sample_empty_shapes: Sequence
):
    """
    Wraps given collate function to handle empty batches.

    Args:
        collate_fn: collate function to wrap
        sample_empty_shapes: expected shape for a batch of size 0. Input is a sequence -
            one for each tensor in the dataset

    Returns:
        New collate function, which is equivalent to input ``collate_fn`` for non-empty
        batches and outputs empty tensors with shapes from ``sample_empty_shapes`` if
        the input batch is of size 0
    """

    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return [torch.zeros(x) for x in sample_empty_shapes]

    return collate


def shape_safe(x: Any):
    """
    Exception-safe getter for ``shape`` attribute

    Args:
        x: any object

    Returns:
        ``x.shape`` if attribute exists, empty tuple otherwise
    """
    return x.shape if hasattr(x, "shape") else ()


class DPDataLoader(DataLoader):
    """
    DataLoader subclass that always does Poisson sampling and supports empty batches
    by default.

    Typically instantiated via ``DPDataLoader.from_data_loader()`` method based
    on another DataLoader. DPDataLoader would preserve the behaviour of the original
    data loader, except for the two aspects.

    First, it switches ``batch_sampler`` to ``UniformWithReplacementSampler``, thus enabling
    Poisson sampling (i.e. each element in the dataset is selected to be in the
    next batch with a certain probability defined by ``sample_rate`` parameter).
    NB: this typically leads to a batches of variable size.
    NB2: By default, ``sample_rate`` is calculated based on the ``batch_size`` of the
    original data loader, so that the average batch size stays the same

    Second, it wraps collate function with support for empty batches.
    Most PyTorch modules will happily process tensors of shape ``(0, N, ...)``,
    but many collate functions will fail to produce such a batch. As with the
    Poisson sampling empty batches become a possibility, we need a DataLoader that
    can handle them.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: float,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        distributed: bool = False,
    ):
        """

        Args:
            dataset: See :class:`torch.utils.data.DataLoader`
            sample_rate: probability with which each element of the dataset is included
                in the next batch.
            num_workers: See :class:`torch.utils.data.DataLoader`
            collate_fn: See :class:`torch.utils.data.DataLoader`
            pin_memory: See :class:`torch.utils.data.DataLoader`
            drop_last: See :class:`torch.utils.data.DataLoader`
            timeout: See :class:`torch.utils.data.DataLoader`
            worker_init_fn: See :class:`torch.utils.data.DataLoader`
            multiprocessing_context: See :class:`torch.utils.data.DataLoader`
            generator: Random number generator used to sample elements
            prefetch_factor: See :class:`torch.utils.data.DataLoader`
            persistent_workers: See :class:`torch.utils.data.DataLoader`
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
                Selects between ``DistributedUniformWithReplacementSampler`` and
                ``UniformWithReplacementSampler`` sampler implementations
        """

        self.sample_rate = sample_rate
        self.distributed = distributed

        if distributed:
            batch_sampler = DistributedUniformWithReplacementSampler(
                total_size=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        else:
            batch_sampler = UniformWithReplacementSampler(
                num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        sample_empty_shapes = [[0, *shape_safe(x)] for x in dataset[0]]
        if collate_fn is None:
            collate_fn = default_collate

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=wrap_collate_with_empty(collate_fn, sample_empty_shapes),
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    @classmethod
    def from_data_loader(
        cls, data_loader: DataLoader, *, distributed: bool = False, generator=None
    ):
        """
        Creates new ``DPDataLoader`` based on passed ``data_loader`` argument.

        Args:
            data_loader: Any DataLoader instance. Must not be over an ``IterableDataset``
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
            generator: Random number generator used to sample elements. Defaults to
                generator from the original data loader.

        Returns:
            New DPDataLoader instance, with all attributes and parameters inherited
            from the original data loader, except for sampling mechanism.

        Examples:
            >>> x, y = torch.randn(64, 5), torch.randint(0, 2, (64,))
            >>> dataset = TensorDataset(x,y)
            >>> data_loader = DataLoader(dataset, batch_size=4)
            >>> dp_data_loader = DPDataLoader.from_data_loader(data_loader)
        """

        if isinstance(data_loader.dataset, IterableDataset):
            raise ValueError("Uniform sampling is not supported for IterableDataset")

        return cls(
            dataset=data_loader.dataset,
            sample_rate=1 / len(data_loader),
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=generator if generator else data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
            distributed=distributed,
        )


def _is_supported_batch_sampler(sampler: Sampler):
    return (
        isinstance(sampler, BatchSampler)
        or isinstance(sampler, UniformWithReplacementSampler)
        or isinstance(sampler, DistributedUniformWithReplacementSampler)
    )


def switch_generator(*, data_loader: DataLoader, generator):
    """
    Creates new instance of a ``DataLoader``, with the exact same behaviour of the
    provided data loader, except for the source of randomness.

    Typically used to enhance a user-provided data loader object with cryptographically
    secure random number generator

    Args:
        data_loader: Any ``DataLoader`` object
        generator:  Random number generator object

    Returns:
        New ``DataLoader`` object with the exact same behaviour as the input data loader,
        except for the source of randomness.
    """
    batch_sampler = data_loader.batch_sampler

    if batch_sampler is None or not _is_supported_batch_sampler(batch_sampler):
        raise ValueError(
            "Non-batch processing is not supported: Opacus always assumes one of the input dimensions to be batch dimension."
        )

    if isinstance(batch_sampler, BatchSampler):
        if not hasattr(batch_sampler.sampler, "generator"):
            raise ValueError(
                "Target sampler doesn't have generator attribute: nothing to switch"
            )

        batch_sampler.sampler.generator = generator
    else:
        batch_sampler.generator = generator

    return DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=batch_sampler,
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        drop_last=data_loader.drop_last,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )
