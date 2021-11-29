#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Union

import torch.nn as nn
from opacus.utils.module_utils import clone_module

from .errors import IllegalModuleConfigurationError, UnsupportedModuleError
from .utils import register_module_fixer, register_module_validator


INSTANCENORM = Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]


@register_module_validator([nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d])
def validate(module: INSTANCENORM) -> List[UnsupportedModuleError]:
    return (
        [
            IllegalModuleConfigurationError(
                "We do not support tracking running stats with differential privacy. "
                "To support it, we would have to add a DP mechanism for these statistics too, "
                "which would incur a privacy cost for little value in model accuracy. "
                "Just say no to running stats :)"
            )
        ]
        if module.track_running_stats
        else []
    )


@register_module_fixer([nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d])
def fix(module: INSTANCENORM) -> INSTANCENORM:
    if len(validate(module)) == 0:
        return module
    # else
    new_module = clone_module(module)
    new_module.track_running_stats = False
    return new_module
