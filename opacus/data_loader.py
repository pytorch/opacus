from typing import Optional, Sequence

import torch
from opacus.utils.uniform_sampler import UniformWithReplacementSampler, DistributedUniformWithReplacementSampler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t


def wrap_collate_with_empty(
    collate_fn: Optional[_collate_fn_t], sample_empty_shapes: Sequence
):
    # TODO: does it work with packed sequences?
    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return [torch.zeros(x) for x in sample_empty_shapes]

    return collate


def shape_safe(x):
    if hasattr(x, "shape"):
        return x.shape
    else:
        return ()


class DPDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        sample_rate: float,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        distributed: bool = False,
    ):

        self.sample_rate = sample_rate
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
    def from_data_loader(cls, data_loader: DataLoader, distributed: bool):
        if isinstance(data_loader, cls):
            assert data_loader.distributed == distributed
            return data_loader

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
            generator=data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
            distributed=distributed
        )
