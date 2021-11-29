#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import numpy as np
import torch
from opacus.utils.uniform_sampler import DistributedUniformWithReplacementSampler


class PoissonSamplingTest(unittest.TestCase):
    def _init_data(self, seed=0):
        generator = torch.Generator()
        generator.manual_seed(seed)

        samplers = []
        dataloaders = []
        torch.distributed.get_world_size = lambda: self.world_size
        for rank in range(self.world_size):
            torch.distributed.get_rank = lambda: rank
            sampler = DistributedUniformWithReplacementSampler(
                total_size=len(self.dataset),
                sample_rate=self.batch_size / len(self.dataset),
                generator=generator,
            )
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_sampler=sampler
            )

            samplers.append(sampler)
            dataloaders.append(dataloader)

        return samplers, dataloaders

    def setUp(self) -> None:
        self.world_size = 2
        self.data_size = 100
        self.batch_size = 10
        self.dataset = [
            (torch.randn(10), torch.randn(10)) for _ in range(self.data_size)
        ]

        self.samplers, self.dataloaders = self._init_data(seed=7)

    def test_length(self):
        for sampler in self.samplers:
            self.assertEqual(len(sampler), 10)
        for dataloader in self.dataloaders:
            self.assertEqual(len(dataloader), 10)

    def test_batch_sizes(self):
        for dataloader in self.dataloaders:
            batch_sizes = []
            for x, _y in dataloader:
                batch_sizes.append(x.shape[0])

            self.assertGreater(len(set(batch_sizes)), 1)
            self.assertAlmostEqual(
                np.mean(batch_sizes), self.batch_size // self.world_size, delta=2
            )

    def test_separate_batches(self):
        indices = {
            rank: [i.item() for batch in self.samplers[rank] for i in batch]
            for rank in range(self.world_size)
        }
        for rank1 in range(self.world_size):
            for rank2 in range(rank1 + 1, self.world_size):
                # Separate workers output separate indices
                self.assertEqual(len(set(indices[rank1]) & set(indices[rank2])), 0)

        all_indices = set()
        for rank in range(self.world_size):
            all_indices |= set(indices[rank])

        # Note that with poisson sampling, the proportion of distinct samples seen during one epoch (1 - 1/e)
        # So the number of distinct samples is expected to be strictly less than the number of samples
        self.assertLessEqual(len(all_indices), self.data_size)
