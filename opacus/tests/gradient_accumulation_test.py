#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData


# TODO: add recurrent model here too
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.convf = nn.Conv1d(32, 32, 1, 1)
        for p in self.convf.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(23, 17)
        self.fc2 = nn.Linear(32 * 17, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 10, 10]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 16, 5, 5]
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # -> [B, 16, 25]
        x = F.relu(self.conv2(x))  # -> [B, 32, 23]
        x = self.convf(x)  # -> [B, 32, 23]
        x = self.fc1(x)  # -> [B, 32, 17]
        x = x.view(-1, x.shape[-2] * x.shape[-1])  # -> [B, 32 * 17]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class GradientAccumulationTest(unittest.TestCase):
    def setUp(self):
        self.DATA_SIZE = 128
        self.BATCH_SIZE = 16
        self.SAMPLE_RATE = self.BATCH_SIZE / self.DATA_SIZE
        self.LR = 0  # we want to call optimizer.step() without modifying the model
        self.ALPHAS = [1 + x / 10.0 for x in range(1, 100, 10)]
        self.criterion = nn.CrossEntropyLoss()

        self.setUp_data()
        self.setUp_model_and_optimizer()

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(1, 35, 35),
            num_classes=10,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.BATCH_SIZE)

    def setUp_model_and_optimizer(self):
        self.model = SampleConvNet()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.LR, momentum=0
        )

        self.optimizer.zero_grad()

        # accumulate .grad over the entire dataset
        for x, y in self.dl:
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

        self.effective_batch_grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        ) * (self.BATCH_SIZE / self.DATA_SIZE)

        self.optimizer.zero_grad()

    def model_forward_backward(
        self,
        model: nn.Module,
        data_iter: Iterable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_steps=1,
        do_zero_grad: bool = True,
    ):
        for x, y in data_iter:
            if optimizer and do_zero_grad:
                optimizer.zero_grad()

            num_steps -= 1
            logits = model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if optimizer:
                optimizer.step()

            if num_steps == 0:
                break

    def test_grad_sample_accumulation(self):
        """
        Calling loss.backward() multiple times should sum up the gradients in .grad
        and accumulate all the individual gradients in .grad-sample
        """
        grad_sample_module = GradSampleModule(self.model)
        data_iter = iter(self.dl)  # 4 batches of size 4 each
        self.model_forward_backward(grad_sample_module, data_iter, num_steps=8)
        # should accumulate grads in .grad and .grad_sample

        for p in self.model.parameters():
            if not p.requires_grad:
                continue

            self.assertTrue(isinstance(p.grad_sample, list))
            self.assertEqual(len(p.grad_sample), 8)

            for gs in p.grad_sample:
                self.assertEqual(gs.shape[0], self.BATCH_SIZE)

        # the accumulated per-sample gradients
        per_sample_grads = torch.cat(
            [
                torch.cat(p.grad_sample).reshape(self.DATA_SIZE, -1)
                for p in self.model.parameters()
                if p.requires_grad
            ],
            dim=-1,
        )
        # average up all the per-sample gradients
        accumulated_grad = torch.mean(per_sample_grads, dim=0)

        # the full data gradient accumulated in .grad
        grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        ) * (self.BATCH_SIZE / self.DATA_SIZE)

        self.optimizer.step()

        # the accumulated gradients in .grad without any hooks
        orig_grad = self.effective_batch_grad

        self.assertTrue(
            torch.allclose(accumulated_grad, orig_grad, atol=10e-5, rtol=10e-3)
        )
        self.assertTrue(torch.allclose(grad, orig_grad, atol=10e-5, rtol=10e-3))

    def test_privacy_engine_poisson_accumulation(self):
        privacy_engine = PrivacyEngine()
        model, optimizer, dl = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dl,
            noise_multiplier=0.0,
            max_grad_norm=999,
        )

        self.model_forward_backward(model, dl, num_steps=1)

        with self.assertRaises(ValueError):
            self.model_forward_backward(model, dl, num_steps=1)

    def test_privacy_engine_no_poisson_accumulation(self):
        privacy_engine = PrivacyEngine()
        model, optimizer, dl = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dl,
            noise_multiplier=0.0,
            max_grad_norm=999,
            poisson_sampling=False,
        )

        self.model_forward_backward(model, dl, num_steps=8)
        self.assertEqual(optimizer.accumulated_iterations, 8)

        for grad_sample in optimizer.grad_samples:
            self.assertEqual(grad_sample.shape[0], 8 * self.BATCH_SIZE)

        optimizer.step()

        # .grad should contain the average gradient over the entire dataset
        accumulated_grad = torch.cat(
            [p.grad.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        )

        # the accumulated gradients in .grad without any hooks
        orig_grad = self.effective_batch_grad

        self.assertTrue(
            torch.allclose(accumulated_grad, orig_grad, atol=10e-5, rtol=10e-3),
            f"Values are {accumulated_grad} vs {orig_grad}."
            f"MAD is {(orig_grad - accumulated_grad).abs().mean()}",
        )

    def test_privacy_engine_zero_grad(self):
        privacy_engine = PrivacyEngine()
        model, optimizer, dl = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

        # should work fine with zero_grad
        self.model_forward_backward(
            model, dl, optimizer, num_steps=2, do_zero_grad=True
        )

        # should fail if not calling zero_grad
        with self.assertRaises(ValueError):
            self.model_forward_backward(
                model, dl, optimizer, num_steps=2, do_zero_grad=False
            )

    def test_batch_splitter_zero_grad(self):
        privacy_engine = PrivacyEngine()
        model, optimizer, dl = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

        with BatchMemoryManager(
            data_loader=dl, max_physical_batch_size=2, optimizer=optimizer
        ) as new_data_loader:
            self.model_forward_backward(
                model, new_data_loader, optimizer, num_steps=3, do_zero_grad=True
            )

            with self.assertRaises(ValueError):
                self.model_forward_backward(
                    model, new_data_loader, optimizer, num_steps=3, do_zero_grad=False
                )
