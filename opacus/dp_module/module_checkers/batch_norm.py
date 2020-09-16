#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Union

import torch.nn as nn

from .errors import UnsupportableModuleError
from .module_checker import ModuleChecker


BATCHNORM = Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]


class BatchNormChecker(ModuleChecker):
    def __init__(self):
        super().__init__(
            [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]
        )

    def is_valid(self, module: BATCHNORM) -> bool:
        return False

    def validate(self, module: BATCHNORM) -> None:
        if self.is_watching(module) and not self.is_valid(module):
            raise UnsupportableModuleError(
                "BatchNorm cannot support training with differential privacy. "
                "The reason for it is that BatchNorm makes each sample's normalized value "
                "depend on its peers in a batch, ie the same sample x will get normalized to "
                "a different value depending on who else is on its batch. "
                "Privacy-wise, this means that we would have to put a privacy mechanism there too. "
                "While it can in principle be done, there are now multiple normalization layers that "
                "do not have this issue: LayerNorm, InstanceNorm and their generalization GroupNorm "
                "are all privacy-safe since they don't have this property."
                "We offer utilities to automatically replace BatchNorms to GroupNorms and we will "
                "release pretrained models to help transition, such as GN-ResNet ie a ResNet using "
                "GroupNorm, pretrained on ImageNet"
            )

    def recommended_replacement(self, module: BATCHNORM) -> nn.GroupNorm:
        return nn.GroupNorm(
            min(32, module.num_features), module.num_features, affine=True
        )
