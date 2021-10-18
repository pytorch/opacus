#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

import torch.nn as nn
from opacus.utils.module_utils import clone_module

from .errors import IllegalConfigurationError
from .utils import register_module_validator, register_module_fixer


CONV = Union[nn.Conv2d, nn.Conv3d]


@register_module_validator([nn.Conv2d, nn.Conv3d])
def validate(module: CONV) -> None:
    return (
        [
            IllegalConfigurationError(
                "TODO: add a good reason."
            )
        ]
        if not (module.groups == 1 or module.groups == module.in_channels)
        else []
    )


@register_module_fixer([nn.Conv2d, nn.Conv3d])
def fix(module: CONV) -> CONV:
    if len(self.validate(module)) == 0:
        return module
    # else
    new_module = clone_module(module)
    new_module.groups = 1
    return new_module
