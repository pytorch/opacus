#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import List, Union

import torch.nn as nn

from .errors import (
    ShouldReplaceModuleError,
    UnsupportableModuleError,
    UnsupportedModuleError,
)
from .utils import register_module_fixer, register_module_validator


logger = logging.getLogger(__name__)

BATCHNORM = Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]
INSTANCENORM = Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]


@register_module_validator(
    [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]
)
def validate(module: BATCHNORM) -> List[UnsupportedModuleError]:
    return [
        ShouldReplaceModuleError(
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
    ]


@register_module_fixer(
    [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]
)
def fix(module: BATCHNORM) -> nn.GroupNorm:
    logger.info(
        "The default batch_norm fixer replaces BatchNorm with GroupNorm."
        " The batch_norm validator module also offers implementations to replace"
        " it with InstanceNorm or Identity. Please check them out and override the"
        " fixer if those are more suitable for your needs."
    )
    return _batchnorm_to_groupnorm(module)


def _batchnorm_to_groupnorm(module: BATCHNORM) -> nn.GroupNorm:
    """
    Converts a BatchNorm ``module`` to GroupNorm module.
    This is a helper function.

    Args:
        module: BatchNorm module to be replaced

    Returns:
        GroupNorm module that can replace the BatchNorm module provided

    Notes:
        A default value of 32 is chosen for the number of groups based on the
        paper *Group Normalization* https://arxiv.org/abs/1803.08494
    """
    return nn.GroupNorm(
        min(32, module.num_features), module.num_features, affine=module.affine
    )


def _batchnorm_to_instancenorm(module: BATCHNORM) -> INSTANCENORM:
    """
    Converts a BatchNorm module to the corresponding InstanceNorm module

    Args:
        module: BatchNorm module to be replaced

    Returns:
        InstanceNorm module that can replace the BatchNorm module provided
    """

    def match_dim():
        if isinstance(module, nn.BatchNorm1d):
            return nn.InstanceNorm1d
        elif isinstance(module, nn.BatchNorm2d):
            return nn.InstanceNorm2d
        elif isinstance(module, nn.BatchNorm3d):
            return nn.InstanceNorm3d
        elif isinstance(module, nn.SyncBatchNorm):
            raise UnsupportableModuleError(
                "There is no equivalent InstanceNorm module to replace"
                " SyncBatchNorm with. Consider replacing it with GroupNorm instead."
            )

    return match_dim()(module.num_features)


def _nullify_batch_norm():
    """
    Replaces all the BatchNorm with :class:`torch.nn.Identity`.
    Args:
        module: BatchNorm module to be replaced

    Returns:
        InstanceNorm module that can replace the BatchNorm module provided

    Notes:
        Most of the times replacing a BatchNorm module with Identity
        will heavily affect convergence of the model.
    """
    return nn.Identity()
