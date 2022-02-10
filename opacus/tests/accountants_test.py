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

from opacus.accountants import GaussianAccountant, RDPAccountant
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

    def test_get_noise_multiplier(self):
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
        )

        self.assertAlmostEqual(noise_multiplier, 1.425307617)
