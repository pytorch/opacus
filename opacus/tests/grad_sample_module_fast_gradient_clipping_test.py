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

import logging

import hypothesis.strategies as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import given, settings
from opacus.grad_sample import GradSampleModule, GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizer, DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.utils.per_sample_gradients_utils import clone_module
from torch.utils.data import DataLoader, Dataset

from .grad_sample_module_test import GradSampleModuleTest, SampleConvNet


class SyntheticDataset(Dataset):
    def __init__(self, size, length, dim):
        self.size = size
        self.length = length
        self.dim = dim
        self.images = torch.randn(self.size, self.length, self.dim, dtype=torch.float32)
        self.labels = torch.randint(
            0, 2, size=(self.size, self.length), dtype=torch.float32
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


class SampleModule(nn.Module):
    def __init__(self):
        super(SampleModule, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1)
        self.layer_norm = nn.LayerNorm(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x).flatten(start_dim=1)
        x = F.softmax(x)
        return x


class GradSampleModuleFastGradientClippingTest(GradSampleModuleTest):
    CLS = GradSampleModuleFastGradientClipping

    def setUp(self):
        self.dim = 2
        self.size = 10
        self.length = 5
        # self.original_model = SampleModule()
        # copy_of_original_model = SampleModule()

        self.original_model = SampleConvNet()
        copy_of_original_model = SampleConvNet()

        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        self.grad_sample_module = self.CLS(
            copy_of_original_model,
            batch_first=True,
            max_grad_norm=1,
            use_ghost_clipping=True,
        )
        self.DATA_SIZE = self.size
        self.setUp_data()
        self.criterion = nn.L1Loss()

    def setUp_data_sequantial(self, size, length, dim):
        self.size = size
        self.length = length
        self.dim = dim
        dataset = SyntheticDataset(size=size, length=length, dim=dim)
        self.dl = DataLoader(dataset, batch_size=size, shuffle=True)

    @given(
        size=st.sampled_from([10]),
        length=st.sampled_from([1]),
        dim=st.sampled_from([2]),
    )
    @settings(deadline=1000000)
    def test_norm_calculation_fast_gradient_clipping(self, size, length, dim):
        """
        Tests if norm calculation is the same between standard (opacus) and fast gradient clipping"
        """
        self.length = length
        self.size = size
        self.dim = dim

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.setUp_data_sequantial(self.size, self.length, self.dim)
        noise_multiplier = 0.0
        batch_size = self.size
        max_grad_norm = 1.0
        sample_module = SampleModule()
        self.model_normal = GradSampleModule(clone_module(sample_module))
        optimizer_normal = torch.optim.SGD(self.model_normal.parameters(), lr=1)
        optimizer_normal = DPOptimizer(
            optimizer_normal,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        self.grad_sample_module = GradSampleModuleFastGradientClipping(
            clone_module(sample_module),
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )
        optimizer_gc = torch.optim.SGD(self.grad_sample_module.parameters(), lr=1)
        optimizer_gc = DPOptimizerFastGradientClipping(
            optimizer_gc,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        (input_data, target_data) = list(self.dl)[0]
        optimizer_normal.zero_grad()
        output_normal = self.model_normal(input_data)
        loss_normal = torch.mean(self.criterion(output_normal, target_data), dim=0)
        loss_normal.backward()
        all_norms_normal = torch.stack(
            [
                torch.stack([g.norm() for g in param.grad_sample], dim=0)
                for param in self.model_normal.parameters()
            ],
            dim=0,
        )
        flat_norms_normal = torch.cat([p.flatten() for p in all_norms_normal])

        self.grad_sample_module.enable_hooks()
        output_gc = self.grad_sample_module(input_data)

        first_loss_per_sample = self.criterion(output_gc, target_data)
        first_loss = torch.mean(first_loss_per_sample)
        first_loss.backward(retain_graph=True)

        optimizer_gc.zero_grad()
        coeff = self.grad_sample_module.get_clipping_coef()
        second_loss_per_sample = coeff * first_loss_per_sample
        second_loss = torch.sum(second_loss_per_sample)
        self.grad_sample_module.disable_hooks()
        second_loss.backward()

        all_norms_gc = [
            param._norm_sample for param in self.grad_sample_module.parameters()
        ]
        flat_norms_gc = torch.cat([p.flatten() for p in all_norms_gc])

        diff = flat_norms_normal - flat_norms_gc

        logging.info(f"Diff = {diff}"),
        msg = "Fail: Gradients from vanilla DP-SGD and from fast gradient clipping are different"
        assert torch.allclose(flat_norms_normal, flat_norms_gc, atol=1e-3), msg

    @given(
        size=st.sampled_from([10]),
        length=st.sampled_from([1, 5]),
        dim=st.sampled_from([2]),
    )
    @settings(deadline=1000000)
    def test_gradient_calculation_fast_gradient_clipping(self, size, length, dim):
        """
        Tests if gradients are the same between standard (opacus) and fast gradient clipping"
        """

        noise_multiplier = 0.0
        batch_size = size
        self.length = length
        self.size = size
        self.dim = dim
        self.setUp_data_sequantial(self.size, self.length, self.dim)
        max_grad_norm = 1.0
        self.criterion = torch.nn.CrossEntropyLoss()

        sample_module = SampleModule()
        self.model_normal = GradSampleModule(clone_module(sample_module))
        self.grad_sample_module = GradSampleModuleFastGradientClipping(
            clone_module(sample_module),
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )

        optimizer_normal = torch.optim.SGD(self.model_normal.parameters(), lr=1)
        optimizer_normal = DPOptimizer(
            optimizer_normal,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        optimizer_gc = torch.optim.SGD(self.grad_sample_module.parameters(), lr=1)
        optimizer_gc = DPOptimizerFastGradientClipping(
            optimizer_gc,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        criterion_gc = DPLossFastGradientClipping(
            self.grad_sample_module, optimizer_gc, self.criterion
        )

        (input_data, target_data) = list(self.dl)[0]
        optimizer_normal.zero_grad()
        output_normal = self.model_normal(input_data)
        loss_normal = torch.mean(self.criterion(output_normal, target_data), dim=0)
        loss_normal.backward()
        optimizer_normal.step()

        all_grads_normal = [
            param.summed_grad for param in self.model_normal.parameters()
        ]
        flat_grads_normal = torch.cat([p.flatten() for p in all_grads_normal])

        output_gc = self.grad_sample_module(input_data)

        loss_gc = criterion_gc(output_gc, target_data)
        loss_gc.backward()
        # double_backward(self.grad_sample_module, optimizer_gc, first_loss_per_sample)

        all_grads_gc = [param.grad for param in self.grad_sample_module.parameters()]
        flat_grads_gc = torch.cat([p.flatten() for p in all_grads_gc])

        diff = torch.tensor(
            [
                (g_gc - g_normal).norm()
                for (g_gc, g_normal) in zip(flat_grads_gc, flat_grads_normal)
            ]
        )
        logging.info(f"Diff = {diff}")
        msg = "Fail: Gradients from vanilla DP-SGD and from fast gradient clipping are different"
        assert torch.allclose(flat_grads_normal, flat_grads_gc, atol=1e-3), msg
