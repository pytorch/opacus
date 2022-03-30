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

import hypothesis.strategies as st
from hypothesis import given, settings
from opacus.accountants import GaussianAccountant, RDPAccountant, create_accountant
from opacus.accountants.utils import get_noise_multiplier


class AccountingTest(unittest.TestCase):
    def test_rdp_accountant(self):
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(epsilon, 7.32911117143)

    def test_gdp_accountant(self):
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 // 0.04)

        accountant = GaussianAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertLess(6.59, epsilon)
        self.assertLess(epsilon, 6.6)

    def test_get_noise_multiplier_rdp_epochs(self):
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="rdp",
        )

        self.assertAlmostEqual(noise_multiplier, 1.416, places=4)

    def test_get_noise_multiplier_rdp_steps(self):
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        steps = 2000

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
        )

        self.assertAlmostEqual(noise_multiplier, 1.3562, places=4)

    @given(
        epsilon=st.floats(1.0, 10.0),
        epochs=st.integers(10, 100),
        sample_rate=st.sampled_from(
            [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
        ),
        delta=st.sampled_from([1e-4, 1e-5, 1e-6]),
    )
    @settings(deadline=10000)
    def test_get_noise_multiplier_overshoot(self, epsilon, epochs, sample_rate, delta):

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
        )

        accountant = create_accountant(mechanism="rdp")
        accountant.history = [
            (noise_multiplier, sample_rate, int(epochs / sample_rate))
        ]

        actual_epsilon = accountant.get_epsilon(delta=delta)
        self.assertLess(actual_epsilon, epsilon)

    def test_get_noise_multiplier_gdp(self):
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="gdp",
        )

        self.assertAlmostEqual(noise_multiplier, 1.3232421875)
