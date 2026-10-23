#!/usr/bin/env python3
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

import unittest

import torch
from opacus import PrivacyEngine
from opacus.schedulers import ExponentialGradClip, LambdaGradClip, StepGradClip
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class GradClipSchedulerTest(unittest.TestCase):
    def setUp(self):
        n_data, dim = 100, 10
        data = torch.randn(n_data, dim)
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        data_loader = DataLoader(TensorDataset(data), batch_size=10)
        self.engine = PrivacyEngine()

        self.module, self.optimizer, self.data_loader = self.engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

    def test_exponential_scheduler(self):
        gamma = 0.99
        scheduler = ExponentialGradClip(self.optimizer, gamma=gamma)

        self.assertEqual(self.optimizer.max_grad_norm, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.max_grad_norm, gamma)

    def test_step_scheduler(self):
        gamma = 0.1
        step_size = 2
        scheduler = StepGradClip(self.optimizer, step_size=step_size, gamma=gamma)

        self.assertEqual(self.optimizer.max_grad_norm, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.max_grad_norm, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.max_grad_norm, gamma)
        scheduler.step()
        self.assertEqual(self.optimizer.max_grad_norm, gamma)
        scheduler.step()
        self.assertEqual(self.optimizer.max_grad_norm, gamma**2)

    def test_lambda_scheduler(self):
        def scheduler_function(epoch):
            return 1 - epoch / 10

        scheduler = LambdaGradClip(
            self.optimizer, scheduler_function=scheduler_function
        )

        self.assertEqual(self.optimizer.max_grad_norm, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.max_grad_norm, scheduler_function(1))
