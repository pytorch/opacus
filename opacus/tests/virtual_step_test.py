#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
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


class GradientAccumulation_test(unittest.TestCase):
    def setUp(self):
        self.DATA_SIZE = 64
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

    def model_forward_backward(self, model, data_iter, num_steps=1):
        for x, y in data_iter:
            num_steps -= 1
            logits = model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if num_steps == 0:
                break

    def test_grad_sample_accumulation(self):
        """
        Calling loss.backward() multiple times should sum up the gradients in .grad
        and accumulate all the individual gradients in .grad-sample
        """
        grad_sample_module = GradSampleModule(self.model)
        data_iter = iter(self.dl)  # 4 batches of size 4 each
        self.model_forward_backward(grad_sample_module, data_iter, num_steps=4)
        # should accumulate grads in .grad and .grad_sample

        # the accumulated per-sample gradients
        per_sample_grads = torch.cat(
            [
                p.grad_sample.reshape(self.DATA_SIZE, -1)
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

    def test_optimizer_accumulation(self):
        """
        Calling `privacy_engine.make_private` should call optimizer.virtual_step()
        under the hood.
        """
        privacy_engine = PrivacyEngine()
        model, optimizer, _ = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dl,
            noise_multiplier=0.0,
            max_grad_norm=999,
        )
        data = iter(self.dl)  # 4 batches of size 4 each

        for _ in range(3):  # take 3 virtual steps
            self.model_forward_backward(model, data, num_steps=1)

        # accumulate on the last step
        self.model_forward_backward(model, data, num_steps=1)
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
