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
import warnings

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers import AdaClipDPOptimizer, DPOptimizer, DPPerLayerOptimizer
from opacus.validators.errors import (
    IllegalModuleConfigurationError,
    UnsupportedModuleError,
)
from opacus.validators.module_validator import ModuleValidator

try:
    from peft import LoraConfig, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from transformers.pytorch_utils import Conv1D

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LinearModel(nn.Module):
    def __init__(self, in_features=16, out_features=8):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.fc(x)


class LoRAPEFTTest(unittest.TestCase):
    def test_dp_optimizer_zero_grad_warning(self):
        """DPOptimizer should emit a warning when all trainable parameters have zero gradients."""
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_opt = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=4,
        )

        for p in model.parameters():
            p.grad_sample = torch.zeros(4, *p.shape)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            dp_opt.clip_and_accumulate()

            self.assertTrue(
                any(
                    "zero per-sample gradients" in str(w.message)
                    for w in recorded_warnings
                ),
                f"Expected zero-gradient warning, got: {[str(w.message) for w in recorded_warnings]}",
            )

    def test_adaclip_dp_optimizer_zero_grad_warning(self):
        """AdaClipDPOptimizer should emit a warning when all trainable parameters have zero gradients."""
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_opt = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            expected_batch_size=4,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.1,
            max_clipbound=10.0,
            min_clipbound=0.1,
            unclipped_num_std=0.5,
        )

        for p in model.parameters():
            p.grad_sample = torch.zeros(4, *p.shape)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            dp_opt.clip_and_accumulate()

            self.assertTrue(
                any(
                    "zero per-sample gradients" in str(w.message)
                    for w in recorded_warnings
                ),
                f"Expected zero-gradient warning, got: {[str(w.message) for w in recorded_warnings]}",
            )

    def test_per_layer_optimizer_zero_grad_warning(self):
        """DPPerLayerOptimizer should emit a warning when parameter has zero gradients."""
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_opt = DPPerLayerOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=[1.0],
            expected_batch_size=4,
        )

        for p in model.parameters():
            p.grad_sample = torch.zeros(4, *p.shape)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            dp_opt.clip_and_accumulate()

            self.assertTrue(
                any(
                    "zero per-sample gradients" in str(w.message)
                    for w in recorded_warnings
                ),
                f"Expected zero-gradient warning, got: {[str(w.message) for w in recorded_warnings]}",
            )

    def test_module_validator_no_trainable_parameters(self):
        """ModuleValidator should fail if module has no trainable parameters."""
        model = LinearModel()
        for p in model.parameters():
            p.requires_grad = False

        errors = ModuleValidator.validate(model, strict=False)
        self.assertTrue(
            any(isinstance(e, IllegalModuleConfigurationError) for e in errors),
            f"Expected IllegalModuleConfigurationError, got: {errors}",
        )

        with self.assertRaises(UnsupportedModuleError):
            ModuleValidator.validate(model, strict=True)

    def test_privacy_engine_validate_optimizer_no_trainable_params(self):
        """PrivacyEngine.validate should raise ValueError if optimizer has no trainable parameters."""
        model = LinearModel()
        frozen_param = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        optimizer = torch.optim.SGD([frozen_param], lr=0.1)
        engine = PrivacyEngine()

        with self.assertRaises(ValueError) as ctx:
            engine.validate(module=model, optimizer=optimizer, data_loader=None)
        self.assertIn(
            "does not manage any parameters that require gradients", str(ctx.exception)
        )

    def test_privacy_engine_validate_optimizer_mismatched_params(self):
        """PrivacyEngine.validate should raise ValueError if optimizer has parameters from a different model."""
        model_a = LinearModel()
        model_b = LinearModel()
        optimizer = torch.optim.SGD(model_b.parameters(), lr=0.1)
        engine = PrivacyEngine()

        with self.assertRaises(ValueError) as ctx:
            engine.validate(module=model_a, optimizer=optimizer, data_loader=None)
        self.assertIn(
            "do not belong to the module passed to PrivacyEngine", str(ctx.exception)
        )

    def test_abstract_grad_sample_module_apply(self):
        """_apply should move grad_sample and summed_grad attributes."""
        model = LinearModel()
        gsm = GradSampleModule(model)
        x = torch.randn(4, 16)
        out = gsm(x)
        out.sum().backward()

        self.assertIsNotNone(model.fc.weight.grad_sample)
        model.fc.weight.summed_grad = torch.randn_like(model.fc.weight)

        # Apply cast to double
        gsm.to(dtype=torch.float64)
        self.assertEqual(model.fc.weight.grad_sample.dtype, torch.float64)
        self.assertEqual(model.fc.weight.summed_grad.dtype, torch.float64)

    @unittest.skipUnless(
        HAS_PEFT and HAS_TRANSFORMERS, "PEFT and transformers are required"
    )
    def test_peft_conv1d_warning(self):
        """ModuleValidator should emit a warning when PEFT wraps a Conv1D layer."""

        class Conv1DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = Conv1D(nf=8, nx=16)

            def forward(self, x):
                return self.conv(x)

        m = Conv1DModel()
        lora_cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["conv"], bias="none")
        peft_m = get_peft_model(m, lora_cfg)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            ModuleValidator.validate(peft_m, strict=False)

            self.assertTrue(
                any(
                    "wraps a Conv1D layer" in str(w.message) for w in recorded_warnings
                ),
                f"Expected Conv1D PEFT warning, got: {[str(w.message) for w in recorded_warnings]}",
            )

    @unittest.skipUnless(HAS_PEFT, "PEFT is not installed")
    def test_peft_lora_dp_training_end_to_end(self):
        """PEFT LoRA training on nn.Linear layers works with DPOptimizer and updates adapter weights."""

        class MultiLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 4)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = MultiLinearModel()
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["fc1", "fc2"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-2)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 16), torch.randn(32, 4)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
            module=peft_model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=0.1,
            max_grad_norm=1.0,
        )

        initial_weights = {
            name: p.clone()
            for name, p in dp_model._module.named_parameters()
            if p.requires_grad
        }

        criterion = nn.MSELoss()
        for x, y in dp_dataloader:
            dp_optimizer.zero_grad()
            out = dp_model(x)
            loss = criterion(out, y)
            loss.backward()
            dp_optimizer.step()
            break

        # Check adapter weights received non-zero updates
        has_update = False
        for name, p in dp_model._module.named_parameters():
            if p.requires_grad:
                diff = (p - initial_weights[name]).abs().sum().item()
                if diff > 1e-6:
                    has_update = True
        self.assertTrue(has_update, "Expected LoRA adapter weights to be updated.")


if __name__ == "__main__":
    unittest.main()
