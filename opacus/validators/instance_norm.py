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
