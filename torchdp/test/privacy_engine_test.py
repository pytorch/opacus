#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdp import PrivacyEngine
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 1 * 1, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 10, 10]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 16, 5, 5]
        x = F.relu(self.conv2(x))  # -> [B, 32, 3, 3]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 32, 1, 1]
        x = x.view(-1, 32 * 1 * 1)  # -> [B, 32 * 1 * 1]
        x = self.fc1(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class PrivacyEngine_test(unittest.TestCase):
    def setUp(self):
        self.DATA_SIZE = 64
        self.LR = 0.5
        self.ALPHAS = [1 + x / 10.0 for x in range(1, 100, 10)]
        self.criterion = nn.CrossEntropyLoss()

        self.setUp_data()
        self.setUp_original_model()
        self.setUp_private_model(noise_multiplier=1.3, max_grad_norm=1.0)

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(1, 35, 35),
            num_classes=10,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.DATA_SIZE)

    def setUp_original_model(self):
        self.original_model = SampleConvNet()
        self.original_optimizer = torch.optim.SGD(
            self.original_model.parameters(), lr=self.LR, momentum=0
        )
        for x, y in self.dl:
            logits = self.original_model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.original_optimizer.step()
        self.original_grads_norms = torch.stack(
            [p.grad.norm() for p in self.original_model.parameters()], dim=-1
        )

    def setUp_private_model(
        self,
        noise_multiplier=1.3,
        max_grad_norm=1.0,
    ):
        # Deep copy
        self.private_model = SampleConvNet()  # create the structure
        self.private_model.load_state_dict(self.original_model.state_dict())  # fill it
        self.private_optimizer = torch.optim.SGD(
            self.private_model.parameters(), lr=self.LR, momentum=0
        )

        privacy_engine = PrivacyEngine(
            self.private_model,
            self.dl,
            alphas=self.ALPHAS,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        privacy_engine.attach(self.private_optimizer)

        for x, y in self.dl:
            logits = self.private_model(x)
            loss = self.criterion(logits, y)
            loss.backward()  # puts grad in self.private_model.parameters()
            self.private_optimizer.step()
        self.private_grad_norms = torch.stack(
            [p.grad.norm() for p in self.private_model.parameters()], dim=-1
        )

    def test_privacy_analysis_alpha_in_alphas(self):
        target_delta = 1e-5
        eps, alpha = self.private_optimizer.privacy_engine.get_privacy_spent(
            target_delta
        )
        self.assertTrue(alpha in self.ALPHAS)

    def test_privacy_analysis_epsilon(self):
        target_delta = 1e-5
        eps, alpha = self.private_optimizer.privacy_engine.get_privacy_spent(
            target_delta
        )
        self.assertTrue(eps > 0)

    def test_gradients_change(self):
        """
        Test that gradients are different after one step of SGD
        """
        for layer_grad, private_layer_grad in zip(
            self.original_model.parameters(), self.private_model.parameters()
        ):
            self.assertFalse(torch.allclose(layer_grad, private_layer_grad))

    def test_model_weights_change(self):
        """
        Test that the updated models are different after one step of SGD
        """
        for layer, private_layer in zip(
            self.original_model.parameters(), self.private_model.parameters()
        ):
            self.assertFalse(torch.allclose(layer, private_layer))

    def test_noise_changes_every_time(self):
        """
        Test that adding noise results in ever different model params.
        We disable clipping in this test by setting it to a very high threshold.
        """
        self.setUp_private_model(noise_multiplier=1.3, max_grad_norm=999)
        first_run_params = [p for p in self.private_model.parameters()]

        self.setUp_private_model(noise_multiplier=1.3, max_grad_norm=999)
        second_run_params = [p for p in self.private_model.parameters()]
        for p0, p1 in zip(first_run_params, second_run_params):
            self.assertFalse(torch.allclose(p0, p1))
