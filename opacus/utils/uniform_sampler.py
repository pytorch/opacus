#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.utils.data import Sampler


class UniformWithReplacementSampler(Sampler):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    """

    def __init__(self, num_samples: int, sample_rate: float, generator=None):
        r"""
        Args:
            num_samples (int): number of samples to draw.
            sample_rate (float): probability used in sampling.
            generator (Generator): Generator used in sampling.
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.generator = generator
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __len__(self):
        return int(1 / self.sample_rate)

    def __iter__(self):
        num_batches = int(1 / self.sample_rate)
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices
            num_batches -= 1
