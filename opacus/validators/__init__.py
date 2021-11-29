#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# we import fix and validate methods from all submodules here
# to ensure that when `opacus.validators` is imported,
# we call register_module_validator and register_module_fixer
# on respective methods

from .batch_norm import fix, validate  # noqa
from .instance_norm import fix, validate  # noqa
from .lstm import fix, validate  # noqa
from .module_validator import ModuleValidator
from .multihead_attention import fix, validate  # noqa
from .utils import register_module_fixer, register_module_validator


__all__ = [
    "ModuleValidator",
    "register_module_validator",
    "register_module_fixer",
]
