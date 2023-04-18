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

import logging
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import torch
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _collate_fn_t


logger = logging.getLogger(__name__)


def collate(
    batch: List[torch.Tensor],
    *,
    collate_fn: Optional[_collate_fn_t],
    sample_empty_shapes: Sequence[Tuple],
    dtypes: Sequence[Union[torch.dtype, Type]],
):
    """
    Wraps `collate_fn` to handle empty batches.

    Default `collate_fn` implementations typically can't handle batches of length zero.
    Since this is a possible case for poisson sampling, we need to wrap the collate
    method, producing tensors with the correct shape and size (albeit the batch
    dimension being zero-size)

    Args:
        batch: List of tensort to be passed to collate_fn implementation
        collate_fn: Collame method to be wrapped
        sample_empty_shapes: Sample tensors with the expected shape
        dtypes: Expected dtypes

    Returns:
        Batch tensor(s)
    """

    if len(batch) > 0:
        return collate_fn(batch)
    else:
        return [
            torch.zeros(shape, dtype=dtype)
            for shape, dtype in zip(sample_empty_shapes, dtypes)
        ]


def wrap_collate_with_empty(
    *,
    collate_fn: Optional[_collate_fn_t],
    sample_empty_shapes: Sequence[Tuple],
    dtypes: Sequence[Union[torch.dtype, Type]],
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

    return partial(
        collate,
        collate_fn=collate_fn,
        sample_empty_shapes=sample_empty_shapes,
        dtypes=dtypes,
    )


def shape_safe(x: Any) -> Tuple:
    """
    Exception-safe getter for ``shape`` attribute

    Args:
        x: any object

    Returns:
        ``x.shape`` if attribute exists, empty tuple otherwise
    """
    return getattr(x, "shape", ())


def dtype_safe(x: Any) -> Union[torch.dtype, Type]:
    """
    Exception-safe getter for ``dtype`` attribute

    Args:
        x: any object

    Returns:
        ``x.dtype`` if attribute exists, type of x otherwise
    """
    return getattr(x, "dtype", type(x))


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
        collate_fn: Optional[_collate_fn_t] = None,
        drop_last: bool = False,
        generator=None,
        distributed: bool = False,
        **kwargs,
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
        sample_empty_shapes = [(0, *shape_safe(x)) for x in dataset[0]]
        dtypes = [dtype_safe(x) for x in dataset[0]]
        if collate_fn is None:
            collate_fn = default_collate

        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
                sample_empty_shapes=sample_empty_shapes,
                dtypes=dtypes,
            ),
            generator=generator,
            **kwargs,
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
