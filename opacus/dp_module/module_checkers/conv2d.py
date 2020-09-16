#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch.nn as nn

from .errors import NotYetSupportedModuleError
from .module_checker import ModuleChecker


class Conv2dChecker(ModuleChecker):
    def __init__(self):
        super().__init__([nn.Conv2d])

    def is_valid(self, module: nn.Conv2d) -> bool:
        return module.groups == 1 or module.groups == module.in_channels

    def validate(self, module: nn.Conv2d) -> None:
        if self.is_watching(module) and not self.is_valid(module):
            raise NotYetSupportedModuleError(
                "nn.Conv2d supported only if its groups are of the same dim as its channels, or 1"
            )

    def recommended_replacement(self, module: nn.Conv2d) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.in_channels,  # Only thing we change
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
