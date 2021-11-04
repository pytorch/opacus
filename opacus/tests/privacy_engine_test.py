#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import abc
import unittest
from typing import Optional, OrderedDict
from abc import ABC

import hypothesis.strategies as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import given, settings
from opacus import PrivacyEngine
from opacus.validators.errors import UnsupportedModuleError
from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import FakeData

def get_grad_sample_aggregated(tensor: torch.Tensor, loss_type: str = "mean"):
    if tensor.grad_sample is None:
        raise ValueError(
            f"The input tensor {tensor} has grad computed, but missing grad_sample."
            f"Please attach PrivacyEngine"
        )

    if loss_type not in ("sum", "mean"):
        raise ValueError(f"loss_type = {loss_type}. Only 'sum' and 'mean' supported")

    grad_sample_aggregated = torch.einsum("i...->...", tensor.grad_sample)
    if loss_type == "mean":
        b_sz = tensor.grad_sample.shape[0]
        grad_sample_aggregated /= b_sz

    return grad_sample_aggregated


class BasePrivacyEngineTest(ABC):

    @classmethod
    def setUpClass(cls):
        cls.DATA_SIZE = 512
        cls.BATCH_SIZE = 64
        cls.SAMPLE_RATE = cls.BATCH_SIZE / cls.DATA_SIZE
        cls.LR = 0.5
        cls.ALPHAS = [1 + x / 10.0 for x in range(1, 100, 10)]
        cls.criterion = nn.CrossEntropyLoss()

    def setUp(self):
        torch.manual_seed(42)

    @abc.abstractmethod
    def _init_data(self):
        pass

    @abc.abstractmethod
    def _init_model(self):
        pass

    def _init_vanilla_training(
        self,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
    ):
        model = self._init_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR, momentum=0)
        if state_dict:
            model.load_state_dict(state_dict)
        dl, _ = self._init_data()
        return model, optimizer, dl

    def _init_private_training(
        self,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
        secure_mode: bool = False,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        poisson_sampling: bool = True,
        try_fix_incompatible_modules: bool = False,
    ):
        model = self._init_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR, momentum=0)

        if state_dict:
            model.load_state_dict(state_dict)

        dl, _ = self._init_data()

        privacy_engine = PrivacyEngine(secure_mode=secure_mode)
        model, optimizer, poisson_dl = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=poisson_sampling,
            try_fix_incompatible_modules=try_fix_incompatible_modules,
        )

        return model, optimizer, dl, privacy_engine

    def _train_steps(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        dl: DataLoader,
        max_steps: Optional[int] = None,
    ):

        steps = 0
        for x, y in dl:
            if optimizer:
                optimizer.zero_grad()
            logits = model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if optimizer:
                optimizer.step()

            steps += 1
            if max_steps and steps >= max_steps:
                break

    def test_basic(self):
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=True,
        )
        self._train_steps(model, optimizer, dl)

    def _compare_to_vanilla(self, do_noise, do_clip, expected_match):
        torch.manual_seed(0)
        v_model, v_optimizer, v_dl = self._init_vanilla_training()
        self._train_steps(v_model, v_optimizer, v_dl, max_steps=1)
        vanilla_params = [
            (name, p) for name, p in v_model.named_parameters() if p.requires_grad
        ]

        torch.manual_seed(0)
        p_model, p_optimizer, p_dl, _ = self._init_private_training(
            poisson_sampling=False,
            noise_multiplier=1.0 if do_noise else 0.0,
            max_grad_norm=1.0 if do_clip else 9999.0,
        )
        self._train_steps(p_model, p_optimizer, p_dl, max_steps=1)
        private_params = [p for p in p_model.parameters() if p.requires_grad]

        for (name, vp), pp in zip(vanilla_params, private_params):
            self.assertEqual(
                torch.allclose(vp, pp, atol=1e-8, rtol=1e-3),
                expected_match,
                f"Unexpected private/vanilla weight match ({name})"
                f"Should be: {expected_match}",
            )
            self.assertEqual(
                torch.allclose(vp.grad, pp.grad, atol=1e-8, rtol=1e-3),
                expected_match,
                f"Unexpected private/vanilla gradient match ({name})"
                f"Should be: {expected_match}",
            )

    def _compare_to_vanilla_accumulated(self, do_noise, do_clip, expected_match):
        torch.manual_seed(0)
        v_model, v_optimizer, v_dl = self._init_vanilla_training()
        self._train_steps(v_model, v_optimizer, v_dl, max_steps=4)
        v_optimizer.step()
        vanilla_params = [
            (name, p) for name, p in v_model.named_parameters() if p.requires_grad
        ]

        torch.manual_seed(0)
        p_model, p_optimizer, p_dl, _ = self._init_private_training(
            poisson_sampling=False,
            noise_multiplier=1.0 if do_noise else 0.0,
            max_grad_norm=1.0 if do_clip else 9999.0,
        )
        self._train_steps(p_model, p_optimizer, p_dl, max_steps=4)
        p_optimizer.step()
        private_params = [p for p in p_model.parameters() if p.requires_grad]

        for (name, vp), pp in zip(vanilla_params, private_params):
            self.assertEqual(
                torch.allclose(vp, pp, atol=1e-8, rtol=1e-3),
                expected_match,
                f"Unexpected private/vanilla weight match ({name})"
                f"Should be: {expected_match}",
            )
            self.assertEqual(
                torch.allclose(vp.grad, pp.grad, atol=1e-8, rtol=1e-3),
                expected_match,
                f"Unexpected private/vanilla gradient match ({name})"
                f"Should be: {expected_match}",
            )

    def test_compare_to_vanilla(self):
        """
        Compare gradients and updated weights with vanilla model initialized
        with the same seed
        """
        for do_noise in (False, True):
            for do_clip in (False, True):
                with self.subTest(do_noise=do_noise, do_clip=do_clip):
                    self._compare_to_vanilla(
                        do_noise=do_noise,
                        do_clip=do_clip,
                        expected_match=not (do_noise or do_clip),
                    )

    def test_compare_to_vanilla_accumulated(self):
        """
        Compare gradients and updated weights with vanilla model initialized
        with the same seed
        """
        for do_noise in (False, True):
            for do_clip in (False, True):
                with self.subTest(do_noise=do_noise, do_clip=do_clip):
                    self._compare_to_vanilla(
                        do_noise=do_noise,
                        do_clip=do_clip,
                        expected_match=not (do_noise or do_clip),
                    )

    def test_sample_grad_aggregation(self):
        """
        Check if final gradient is indeed an aggregation over per-sample gradients
        """
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=0.0,
            max_grad_norm=99999.0,
        )
        self._train_steps(model, optimizer, dl, max_steps=1)

        for p_name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            summed_grad = p.grad_sample.sum(dim=0) / self.BATCH_SIZE
            self.assertTrue(
                torch.allclose(p.grad, summed_grad, atol=1e-8, rtol=1e-4),
                f"Per sample gradients don't sum up to the final grad value."
                f"Param: {p_name}",
            )

    def test_noise_changes_every_time(self):
        """
        Test that adding noise results in ever different model params.
        We disable clipping in this test by setting it to a very high threshold.
        """
        model, optimizer, dl, _ = self._init_private_training(poisson_sampling=False)
        self._train_steps(model, optimizer, dl, max_steps=1)
        first_run_params = (p for p in model.parameters() if p.requires_grad)

        model, optimizer, dl, _ = self._init_private_training(poisson_sampling=False)
        self._train_steps(model, optimizer, dl, max_steps=1)
        second_run_params = (p for p in model.parameters() if p.requires_grad)

        for p0, p1 in zip(first_run_params, second_run_params):
            self.assertFalse(torch.allclose(p0, p1))

    def test_model_validator(self):
        """
        Test that the privacy engine raises errors
        if there are unsupported modules
        """
        resnet = models.resnet18()
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1.0)
        privacy_engine = PrivacyEngine()
        dl, _ = self._init_data()

        with self.assertRaises(UnsupportedModuleError):
            _, _, _ = privacy_engine.make_private(
                module=resnet,
                optimizer=optimizer,
                data_loader=dl,
                noise_multiplier=1.3,
                max_grad_norm=1,
            )

    def test_model_validator_after_fix(self):
        """
        Test that the privacy engine fixes unsupported modules
        and succeeds.
        """
        resnet = models.resnet18()
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1.0)
        privacy_engine = PrivacyEngine()
        dl, _ = self._init_data()

        _, _, _ = privacy_engine.make_private(
            module=resnet,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=1.3,
            max_grad_norm=1,
            try_fix_incompatible_modules=True,
        )
        self.assertTrue(1, 1)

    def test_deterministic_run(self):
        """
        Tests that for 2 different models, secure seed can be fixed
        to produce same (deterministic) runs.
        """
        torch.manual_seed(0)
        m1, opt1, dl1, _ = self._init_private_training()
        self._train_steps(m1, opt1, dl1)
        params1 = [p for p in m1.parameters() if p.requires_grad]

        torch.manual_seed(0)
        m2, opt2, dl2, _ = self._init_private_training()
        self._train_steps(m2, opt2, dl2)
        params2 = [p for p in m2.parameters() if p.requires_grad]

        for p1, p2 in zip(params1, params2):
            self.assertTrue(
                torch.allclose(p1, p2),
                "Model parameters after deterministic run must match",
            )

    @given(
        noise_multiplier=st.floats(0.5, 5.0),
        max_steps=st.integers(8, 10),
    )
    @settings(max_examples=20, deadline=1000)
    def test_noise_level(self, noise_multiplier: float, max_steps: int):
        """
        Tests that the noise level is correctly set
        """
        # Initialize models with parameters to zero
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=noise_multiplier
        )
        for p in model.parameters():
            p.data.zero_()

        # Do max_steps steps of DP-SGD
        n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        steps = 0
        for x, y in dl:
            optimizer.zero_grad()
            logits = model(x)
            loss = logits.view(logits.size(0), -1).sum(dim=1)
            # Gradient should be 0
            loss.backward(torch.zeros(logits.size(0)))

            optimizer.step()
            steps += 1

            if max_steps and steps >= max_steps:
                break

        # Noise should be equal to lr*sigma*sqrt(n_params * steps) / batch_size
        expected_norm = (
            steps
            * n_params
            * optimizer.noise_multiplier ** 2
            * self.LR ** 2
            / (optimizer.expected_batch_size ** 2)
        )
        real_norm = sum(
            [torch.sum(torch.pow(p.data, 2)) for p in model.parameters()]
        ).item()

        self.assertAlmostEqual(real_norm, expected_norm, delta=0.05 * expected_norm)

    @unittest.skip("Not yet implemented")
    def test_raises_seed_set_on_secure_rng(self):
        """
        Tests that when a seed is set on a secure PrivacyEngine, we raise a ValueError
        """
        model, optimizer, dl = self.setUp_init_model(
            private=True, secure_mode=True, noise_multiplier=1.3, max_grad_norm=1.0
        )
        with self.assertRaises(ValueError):
            optimizer.privacy_engine._set_seed(20)

    @unittest.skip("Not yet implemented")
    def test_noise_changes_every_time_secure_rng(self):
        """
        Test that adding noise results in ever different model params.
        We disable clipping in this test by setting it to a very high threshold.
        """
        model, optimizer, dl = self.setUp_init_model(
            private=True,
            state_dict=self.original_model.state_dict(),
            secure_mode=True,
            noise_multiplier=1.3,
            max_grad_norm=999,
        )
        self.setUp_model_step(model, optimizer, dl)
        first_run_params = (p for p in model.parameters() if p.requires_grad)

        model, optimizer, dl = self.setUp_init_model(
            private=True,
            state_dict=self.original_model.state_dict(),
            secure_mode=True,
            noise_multiplier=1.3,
            max_grad_norm=999,
        )
        self.setUp_model_step(model, optimizer, dl)
        second_run_params = (p for p in model.parameters() if p.requires_grad)
        for p0, p1 in zip(first_run_params, second_run_params):
            self.assertFalse(torch.allclose(p0, p1))

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.gnorm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.lnorm1 = nn.LayerNorm((32, 23))
        self.conv3 = nn.Conv1d(32, 32, 3, 1)
        self.instnorm1 = nn.InstanceNorm1d(32, affine=True)
        self.convf = nn.Conv1d(32, 32, 1, 1)
        for p in self.convf.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(21, 17)
        self.lnorm2 = nn.LayerNorm(17)
        self.fc2 = nn.Linear(32 * 17, 10)

        for layer in (self.gnorm1, self.lnorm1, self.lnorm2, self.instnorm1):
            nn.init.uniform_(layer.weight)
            nn.init.uniform_(layer.bias)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = self.conv1(x)  # -> [B, 16, 10, 10]
        x = self.gnorm1(x)  # -> [B, 16, 10, 10]
        x = F.relu(x)  # -> [B, 16, 10, 10]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 16, 5, 5]
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # -> [B, 16, 25]
        x = self.conv2(x)  # -> [B, 32, 23]
        x = self.lnorm1(x)  # -> [B, 32, 23]
        x = F.relu(x)  # -> [B, 32, 23]
        x = self.conv3(x)  # -> [B, 32, 21]
        x = self.instnorm1(x)  # -> [B, 32, 21]
        x = self.convf(x)  # -> [B, 32, 21]
        x = self.fc1(x)  # -> [B, 32, 17]
        x = self.lnorm2(x)  # -> [B, 32, 17]
        x = x.view(-1, x.shape[-2] * x.shape[-1])  # -> [B, 32 * 17]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class PrivacyEngineConvNetTest(BasePrivacyEngineTest, unittest.TestCase):

    def _init_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(1, 35, 35),
            num_classes=10,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.BATCH_SIZE)

    def _init_model(
        self, private=False, state_dict=None, model=None, **privacy_engine_kwargs
    ):
        return SampleConvNet()


class SampleAttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 8)
        self.attn = DPMultiheadAttention(8, 1)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.attn(x, x, x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)
        return x


class PrivacyEngineTextTest(BasePrivacyEngineTest, unittest.TestCase):
    def _init_data(self):
        x = torch.randint(0, 100, (12, self.BATCH_SIZE))
        y = torch.randint(0, 12, (self.BATCH_SIZE,))
        self.dl = [(x, y)]

    def _init_model(
        self, private=False, state_dict=None, model=None, **privacy_engine_kwargs
    ):
        return SampleAttnNet()