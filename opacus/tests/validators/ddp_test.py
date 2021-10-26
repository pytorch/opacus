#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import unittest

import torch.distributed as dist
import torch.nn as nn
from opacus.validators.errors import ShouldReplaceModuleError
from opacus.validators.errors import ShouldReplaceModuleError
from opacus.validators.module_validator import ModuleValidator
from torch.nn.parallel import DistributedDataParallel as DDP


class DDPValidator_test(unittest.TestCase):
    def setUp(self):
        self.module = nn.Linear(8, 4)
        self.mv = ModuleValidator.VALIDATORS
        self.mf = ModuleValidator.FIXERS
        self._setup_dist()

    def tearDown(self):
        self._cleanup_dist()

    def _cleanup_dist(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_dist(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        # initialize the process group
        dist.init_process_group("gloo", rank=0, world_size=1)

    def test_validate(self):
        ddp = DDP(self.module)
        val_ddp = self.mv[type(ddp)](ddp)
        self.assertTrue(len(val_ddp), 1)
        self.assertTrue(isinstance(val_ddp[0], ShouldReplaceModuleError))

    def test_fix(self):
        ddp = DDP(self.module)
        with self.assertRaises(ShouldReplaceModuleError):
            fix_ddp = self.mf[type(ddp)](ddp)
