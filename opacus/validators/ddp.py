#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import List

from torch.nn.parallel import DistributedDataParallel as DDP

from .errors import ShouldReplaceModuleError, UnsupportedModuleError
from .utils import register_module_fixer, register_module_validator


@register_module_validator(DDP)
def validate(module: DDP) -> List[UnsupportedModuleError]:
    return [
        ShouldReplaceModuleError(
            "We do not support DistributedDataParallel as we need to perform"
            " additonal synchronization across workers. We provide implementation"
            " DifferentiallyPrivateDistributedDataParallel, which is compatible"
            " with our DP. Please use the recommended replacement!"
        )
    ]


@register_module_fixer(DDP)
def fix(module: DDP) -> None:
    raise ShouldReplaceModuleError(
        "We do not support DistributedDataParallel as we need to perform"
        " additonal synchronization across workers. We provide implementation"
        " DifferentiallyPrivateDistributedDataParallel, which is compatible"
        " with our DP. Unfortunately, automatic fix for DDP module is not possible"
        " since torch distributed process group initialization infomation is "
        " unavailable to PrivacyEngine. Please fix this manually."
    )
