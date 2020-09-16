#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Union

import torch.nn as nn

from .errors import UnsupportableModuleError
from .module_checker import ModuleChecker


class InstanceNormChecker(ModuleChecker):
    def __init__(self):
        super().__init__([nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d])

    def is_valid(
        self, module: Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    ) -> bool:
        return not module.track_running_stats

    def validate(
        self, module: Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    ) -> None:
        if self.is_watching(module) and not self.is_valid(module):
            raise UnsupportableModuleError(
                "We do not support tracking running stats with differential privacy. "
                "To support it, we would have to add a DP mechanism for these statistics too "
                "which would incur a privacy cost for little value in model accuracy. "
                "Just say no to running stats :)"
            )

    def recommended_replacement(
        self, module: Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    ) -> Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]:
        module.track_running_stats = False
        return module
