#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch import nn, optim


class PoissonSamplingTest(unittest.TestCase):
    def test_poisson_sampling(self):
        B = 1
        N = 10
        d = 10
        dataset = [(i, torch.randn(d), torch.randn(d)) for i in range(N)]

        model = nn.Linear(d, d)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        engine = PrivacyEngine(
            model,
            sample_rate=B / N,
            target_epsilon=1.0,
            epochs=10,
            poisson=True,
            max_grad_norm=1,
            sample_size=N,
        )
        engine.attach(optimizer)

        generator = torch.Generator()
        generator.manual_seed(7)
        sampler = UniformWithReplacementSampler(
            num_samples=N, sample_rate=B / N, generator=generator
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

        # Sampler with seed=7 should generate [], [7], [], [], [9], [0], [], [], [1], [4]
        for (_, x, y) in dataloader:
            prediction = model(x)
            loss = torch.mean((prediction - y) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
