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

"""
Tests for AdaClipDPOptimizer to ensure correct privacy accounting.

The AdaClip optimizer uses an adjusted noise multiplier for gradient noise
(Theorem 1 from https://arxiv.org/pdf/1905.03871.pdf), but the original
noise_multiplier should be preserved for privacy accounting.
"""

import math
import unittest
from unittest import mock

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.optimizer import DPOptimizer
from torch.utils.data import DataLoader, TensorDataset


CLIPPED = 10.0
UNCLIPPED = 0.5


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class AdaClipNoiseMultiplierTest(unittest.TestCase):
    """Test that AdaClip preserves original noise_multiplier for privacy accounting."""

    def setUp(self):
        # For AdaClip: noise_multiplier must be < 2 * unclipped_num_std
        self.noise_multiplier = 1.5
        self.unclipped_num_std = 1.0  # 2 * 1.0 = 2.0 > 1.5, so valid
        self.max_grad_norm = 1.0
        torch.manual_seed(42)

    def test_adaclip_preserves_noise_multiplier(self):
        """Test that AdaClipDPOptimizer preserves original noise_multiplier."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # noise_multiplier should remain unchanged (original value)
        self.assertEqual(adaclip_optimizer.noise_multiplier, self.noise_multiplier)

    def test_adaclip_with_zero_noise(self):
        """Test that AdaClipDPOptimizer handles zero noise_multiplier correctly."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.0,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # noise_multiplier should be zero
        self.assertEqual(adaclip_optimizer.noise_multiplier, 0.0)

    def test_accountant_uses_original_noise_multiplier(self):
        """Test that accountant hook uses original noise_multiplier from AdaClip optimizer."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        accountant = RDPAccountant()

        # Create AdaClip optimizer
        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        sample_rate = 0.01

        # Manually call accountant.step with noise_multiplier
        # (mimicking what the hook would do)
        initial_len = len(accountant)
        accountant.step(
            noise_multiplier=adaclip_optimizer.noise_multiplier,
            sample_rate=sample_rate,
        )

        # Accountant should have recorded one step
        self.assertEqual(len(accountant), initial_len + 1)

        # Verify that the accountant used the original noise_multiplier
        # accountant.history stores tuples of (noise_multiplier, sample_rate, num_steps)
        last_entry = accountant.history[-1]
        recorded_noise_multiplier = last_entry[0]

        # Should use original noise_multiplier
        self.assertEqual(recorded_noise_multiplier, self.noise_multiplier)

    def test_privacy_accounting_with_adaclip_e2e(self):
        """End-to-end test: verify privacy accounting is correct with AdaClip via PrivacyEngine."""
        # Create synthetic data
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10)

        # Setup model and use PrivacyEngine with adaptive clipping
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Use PrivacyEngine to set up AdaClip with all required kwargs
        privacy_engine = PrivacyEngine()
        model, dp_optimizer, private_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            clipping="adaptive",
            grad_sample_mode="hooks",
            # AdaClip specific parameters
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
        )

        # Verify optimizer is AdaClip
        self.assertIsInstance(dp_optimizer, AdaClipDPOptimizer)

        # Verify noise_multiplier is preserved
        self.assertEqual(dp_optimizer.noise_multiplier, self.noise_multiplier)

        # Get the accountant
        accountant = privacy_engine.accountant

        # Train for a few steps
        criterion = nn.CrossEntropyLoss()
        for i, (data, target) in enumerate(private_dataloader):
            if i >= 5:  # Just a few steps for testing
                break
            dp_optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            dp_optimizer.step()

        # Verify accountant recorded steps with original noise_multiplier
        self.assertGreater(len(accountant), 0)

        # All recorded noise multipliers should be the original value
        for entry in accountant.history:
            recorded_noise = entry[0]
            # Should match original noise_multiplier
            self.assertEqual(recorded_noise, self.noise_multiplier)

    def test_comparison_dpoptimizer_vs_adaclip_accounting(self):
        """Compare accounting between standard DPOptimizer and AdaClip with same initial noise."""
        model1 = SimpleModel()
        model2 = SimpleModel()
        model2.load_state_dict(model1.state_dict())  # Same initial weights

        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

        # Standard DPOptimizer
        dp_optimizer = DPOptimizer(
            optimizer=optimizer1,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # AdaClip optimizer
        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer2,
            noise_multiplier=self.noise_multiplier,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # Both should have the same noise_multiplier for accounting
        self.assertEqual(
            dp_optimizer.noise_multiplier,
            adaclip_optimizer.noise_multiplier,
        )

    def test_adaclip_noise_adjustment_calculation(self):
        """Test that the adjusted noise follows Theorem 1 formula when applied internally."""
        # According to Theorem 1: σ_eff = (σ^-2 - (2σ_u)^-2)^(-1/2)
        sigma = self.noise_multiplier
        sigma_u = self.unclipped_num_std

        expected_adjusted = (sigma ** (-2) - (2 * sigma_u) ** (-2)) ** (-1 / 2)

        # Verify the formula produces valid results
        self.assertGreater(expected_adjusted, 0)
        # The adjusted noise is larger than the original
        # (σ^-2 - positive_term)^(-1/2) > σ when σ < 2*σ_u
        self.assertGreater(expected_adjusted, sigma)

    def test_adaclip_constraint_validation(self):
        """Test that AdaClip raises error when noise_multiplier >= 2 * unclipped_num_std."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # This should raise ValueError: 2.0 >= 2 * 1.0 = 2.0
        with self.assertRaises(ValueError) as context:
            AdaClipDPOptimizer(
                optimizer=optimizer,
                noise_multiplier=2.0,
                target_unclipped_quantile=0.5,
                clipbound_learning_rate=0.01,
                max_clipbound=2.0,
                min_clipbound=0.5,
                unclipped_num_std=1.0,
                max_grad_norm=self.max_grad_norm,
                expected_batch_size=32,
            )

        self.assertIn(
            "noise_multiplier must be smaller than 2 * unclipped_num_std",
            str(context.exception),
        )


class AdaClipUnclippedFractionTest(unittest.TestCase):
    def setUp(self):
        self.noise_multiplier = 1.0
        self.unclipped_num_std = 1.0
        self.max_grad_norm = 1.0
        self.clipbound_learning_rate = 0.2
        self.target_unclipped_quantile = 0.5
        self.noise_stds = []
        self.noise_secure_modes = []

    def _zero_noise(self, std, reference, generator=None, secure_mode=False):
        self.noise_stds.append(std)
        self.noise_secure_modes.append(secure_mode)
        return torch.zeros_like(reference)

    def _make_optimizer(self, expected_batch_size, **kwargs):
        self.param = torch.nn.Parameter(torch.zeros(1))
        return AdaClipDPOptimizer(
            torch.optim.SGD([self.param], lr=0.0),
            noise_multiplier=self.noise_multiplier,
            target_unclipped_quantile=self.target_unclipped_quantile,
            clipbound_learning_rate=self.clipbound_learning_rate,
            max_clipbound=1e9,
            min_clipbound=1e-9,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction="sum",
            **kwargs,
        )

    def _step(self, optimizer, per_sample_grads):
        optimizer.max_grad_norm = self.max_grad_norm
        if isinstance(per_sample_grads[0], list):
            self.param.grad_sample = [
                torch.tensor(g).reshape(-1, 1) for g in per_sample_grads
            ]
        else:
            self.param.grad_sample = torch.tensor(per_sample_grads).reshape(-1, 1)
        self.param.grad = torch.zeros(1)
        with mock.patch(
            "opacus.optimizers.adaclipoptimizer._generate_noise", self._zero_noise
        ):
            optimizer.step()
        grad_sum = self.param.grad.item()
        new_max_grad_norm = float(optimizer.max_grad_norm)
        optimizer.zero_grad()
        return grad_sum, new_max_grad_norm

    def _released_fraction(self, new_max_grad_norm):
        return (
            self.target_unclipped_quantile
            - math.log(new_max_grad_norm / self.max_grad_norm)
            / self.clipbound_learning_rate
        )

    def test_requires_expected_batch_size(self):
        with self.assertRaises(ValueError):
            self._make_optimizer(expected_batch_size=None)

    def test_joint_sensitivity_matches_noise_multiplier(self):
        n = 8
        for base, extra in [([CLIPPED] * n, UNCLIPPED), ([UNCLIPPED] * n, CLIPPED)]:
            optimizer = self._make_optimizer(expected_batch_size=n)
            self.noise_stds = []
            grad_b, clip_b = self._step(optimizer, base)
            grad_bp, clip_bp = self._step(optimizer, base + [extra])
            grad_std, count_std = self.noise_stds[:2]

            centred_b = (self._released_fraction(clip_b) - 0.5) * n
            centred_bp = (self._released_fraction(clip_bp) - 0.5) * n
            delta_count = centred_bp - centred_b
            delta_grad = grad_bp - grad_b

            expected_delta = 0.5 if extra == UNCLIPPED else -0.5
            self.assertAlmostEqual(delta_count, expected_delta, places=5)
            self.assertLessEqual(abs(delta_grad), self.max_grad_norm + 1e-5)
            self.assertAlmostEqual(count_std, self.unclipped_num_std)

            worst_case = math.hypot(self.max_grad_norm / grad_std, 0.5 / count_std)
            self.assertAlmostEqual(worst_case, 1 / self.noise_multiplier, places=5)
            self.assertLessEqual(
                math.hypot(delta_grad / grad_std, delta_count / count_std),
                1 / self.noise_multiplier + 1e-5,
            )

    def test_fraction_uses_expected_batch_size(self):
        m = 10
        optimizer = self._make_optimizer(expected_batch_size=m)
        grads = [UNCLIPPED] * 4 + [CLIPPED] * 2
        _, new_max_grad_norm = self._step(optimizer, grads)
        expected_frac = (4 - 6 / 2) / m + 0.5
        self.assertAlmostEqual(
            self._released_fraction(new_max_grad_norm), expected_frac, places=5
        )
        self.assertAlmostEqual(
            new_max_grad_norm,
            self.max_grad_norm
            * math.exp(
                -self.clipbound_learning_rate
                * (expected_frac - self.target_unclipped_quantile)
            ),
            places=5,
        )

    def test_fraction_equals_unclipped_share_at_expected_batch_size(self):
        m = 10
        optimizer = self._make_optimizer(expected_batch_size=m)
        grads = [UNCLIPPED] * 3 + [CLIPPED] * 7
        _, new_max_grad_norm = self._step(optimizer, grads)
        self.assertAlmostEqual(
            self._released_fraction(new_max_grad_norm), 3 / m, places=5
        )

    def test_fraction_with_accumulated_iterations(self):
        m = 4
        optimizer = self._make_optimizer(expected_batch_size=m)
        grads = [[UNCLIPPED] * 3 + [CLIPPED], [CLIPPED]]
        _, new_max_grad_norm = self._step(optimizer, grads)
        expected_frac = (3 - 5 / 2) / (2 * m) + 0.5
        self.assertAlmostEqual(
            self._released_fraction(new_max_grad_norm), expected_frac, places=5
        )

    def test_fraction_with_skipped_virtual_steps(self):
        m = 6
        optimizer = self._make_optimizer(expected_batch_size=m)
        optimizer.signal_skip_step(do_skip=True)
        _, clip_after_skip = self._step(optimizer, [UNCLIPPED] * 3)
        self.assertEqual(clip_after_skip, self.max_grad_norm)
        optimizer.signal_skip_step(do_skip=False)
        _, new_max_grad_norm = self._step(optimizer, [CLIPPED] * 3)
        self.assertAlmostEqual(
            self._released_fraction(new_max_grad_norm), 3 / m, places=5
        )

    def test_unclipped_count_noise_uses_secure_mode(self):
        optimizer = self._make_optimizer(expected_batch_size=4, secure_mode=True)
        self._step(optimizer, [UNCLIPPED] * 4)
        self.assertEqual(self.noise_secure_modes, [True, True])


if __name__ == "__main__":
    unittest.main()
