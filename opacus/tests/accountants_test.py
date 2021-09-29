#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from opacus.accountants import RDPAccountant, GaussianAccountant


class AccountingTest(unittest.TestCase):
    def test_rdp_accountant(self):
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier, sample_rate)

        epsilon = accountant.get_privacy_spent(delta=1e-5)[0]
        self.assertLess(7.3291, epsilon)
        self.assertLess(epsilon, 7.3292)

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
