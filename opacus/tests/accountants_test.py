#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from opacus.accountants import GaussianAccountant, RDPAccountant
from opacus.accountants.rdp import get_noise_multiplier


class AccountingTest(unittest.TestCase):
    def test_rdp_accountant(self):
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier, sample_rate)

        epsilon = accountant.get_privacy_spent(delta=1e-5)[0]
        self.assertAlmostEqual(epsilon, 7.32911117143)

    def test_gdp_accountant(self):
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 // 0.04)

        accountant = GaussianAccountant(noise_multiplier, sample_rate, poisson=True)
        for _ in range(steps):
            accountant.step(noise_multiplier, sample_rate)

        epsilon = accountant.get_privacy_spent(delta=1e-5)
        self.assertLess(6.59, epsilon)
        self.assertLess(epsilon, 6.6)

    def test_get_noise_multiplier(self):
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(epsilon, delta, sample_rate, epochs)

        self.assertAlmostEqual(noise_multiplier, 1.425307617)
