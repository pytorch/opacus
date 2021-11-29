#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import numpy as np
import torch
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


class PoissonSamplingTest(unittest.TestCase):
    def _init_data(self, seed=0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = UniformWithReplacementSampler(
            num_samples=len(self.dataset),
            sample_rate=self.batch_size / len(self.dataset),
            generator=generator,
        )
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_sampler=sampler)

        return sampler, dataloader

    def setUp(self) -> None:
        self.data_size = 100
        self.batch_size = 10
        self.dataset = [
            (torch.randn(10), torch.randn(10)) for _ in range(self.data_size)
        ]

        self.sampler, self.dataloader = self._init_data(seed=7)

    def test_length(self):
        self.assertEqual(len(self.sampler), 10)
        self.assertEqual(len(self.dataloader), 10)

    def test_batch_sizes(self):
        batch_sizes = []
        for x, _y in self.dataloader:
            batch_sizes.append(x.shape[0])

        self.assertGreater(len(set(batch_sizes)), 1)
        self.assertAlmostEqual(np.mean(batch_sizes), self.batch_size, delta=2)

    def test_same_seed(self):
        batch_sizes1 = []
        for x, _y in self.dataloader:
            batch_sizes1.append(x.shape[0])

        _, dataloader = self._init_data(seed=7)
        batch_sizes2 = []
        for x, _y in dataloader:
            batch_sizes2.append(x.shape[0])

        self.assertEqual(batch_sizes1, batch_sizes2)

    def test_different_seed(self):
        batch_sizes1 = []
        for x, _y in self.dataloader:
            batch_sizes1.append(x.shape[0])

        _, dataloader = self._init_data(seed=8)
        batch_sizes2 = []
        for x, _y in dataloader:
            batch_sizes2.append(x.shape[0])

        self.assertNotEqual(batch_sizes1, batch_sizes2)
