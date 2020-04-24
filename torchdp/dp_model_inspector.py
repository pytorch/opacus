#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch import nn

from .autograd_grad_sample import is_supported
from .utils import ModelInspector, get_layer_type, requires_grad


class IncompatibleModuleException(Exception):
    """
    Exception class to be thrown from Privacy Engine in case
    the given model contains incompatible modules.
    """

    pass


class DPModelInspector:
    """
    Class to wrap `ModelInspector`s that are relevant for the
    `PrivacyEngine`. This class provides an easy interface for the
    privacy engine to validate a model.

    Attributes:
        inspectors: List of ModuleInspectors that are used for model
        validation.
        should_throw: A flag (`True` by default) that makes the inspector throw
        if any of the ModuleInspectors return `False`. To continue (without
        privacy and/or run-time error guarantee) you can set this flag to `False`
    """

    def __init__(self):
        self.should_throw = True

        def is_valid(module: nn.Module):
            valid = (not requires_grad(module)) or is_supported(module)
            if valid and isinstance(module, nn.Conv2d):
                valid = module.groups == 1 or module.groups == module.in_channels
            return valid

        def no_batchnorm(module: nn.Module):
            return not (
                requires_grad(module)
                and isinstance(module, nn.modules.batchnorm._BatchNorm)
            )

        def no_running_stats_instancenorm(module: nn.Module):
            is_instancenorm = get_layer_type(module) in (
                "InstanceNorm1d",
                "InstanceNorm2d",
                "InstanceNorm3d",
            )

            return (
                not is_instancenorm
                or not requires_grad(module)
                or not module.track_running_stats
            )

        self.inspectors = [
            # Inspector to check model only consists of sub-modules we support
            ModelInspector(
                name="validity",
                predicate=is_valid,
                message="Some modules are not valid.",
            ),
            # Inspector to check for BatchNorms as they could be replaced with groupnorm
            ModelInspector(
                name="batchnorm",
                predicate=no_batchnorm,
                message="Model contains BatchNorm layers. It is recommended"
                "That they are replaced with GroupNorm.",
            ),
            # Inspector to check that instance norms doesn't track running stats
            ModelInspector(
                name="running_stats",
                predicate=no_running_stats_instancenorm,
                message="InstanceNorm layer initialised with track_running_stats=True."
                "This is currently not supported",
            ),
        ]

    def validate(self, model: nn.Module) -> True:
        """
        Runs the existing `inspectors` on all the sub-modules of the model. Returns
        `True` if all the predicates pass on all the sub-modules, throws
        `IncompatibleModuleException` if not. The list of modules/sub-modules that
        violated each of the `predicates` are returned as part of the exception message.


        Args:
            model: The model to validate.

        Returns:
            A boolean if all the inspectors pass on all modules.

        Examples:

            insp = DPModelInspector()
            model = nn.BatchNorm1d(2)
            valid = inspector.validate(model)
            # returns False, look at insp.inspectors[i].violators.
        """
        valid = all(inspector.validate(model) for inspector in self.inspectors)
        if self.should_throw and (not valid):
            message = "Model contains incompatible modules."
            for inspector in self.inspectors:
                if inspector.violators:
                    message += f"\n{inspector.message}: {inspector.violators}"
            raise IncompatibleModuleException(message)
        return valid
