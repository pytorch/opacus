#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch import nn

from .autograd_grad_sample import is_supported
from .utils.module_inspection import ModelInspector, get_layer_type, requires_grad


class IncompatibleModuleException(Exception):
    r"""
    Exception class to be thrown in case
    the given model contains incompatible modules.
    """

    pass


class DPModelInspector:
    r""" Class to validate if a given module meets the requirements for attaching :class:`~torchdp.privacy_engine.PrivacyEngine`.

    Active checks are listed in :attr:`~torchdp.dp_model_inspector.DPModelInspector.inspectors` attribute.
    """

    def __init__(self):
        self.should_throw = True

        def is_valid(module: nn.Module):
            valid = (not requires_grad(module)) or is_supported(module)
            if valid and isinstance(module, nn.Conv2d):
                # pyre-fixme[16]: `Conv2d` has no attribute `in_channels`.
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
                # pyre-fixme[16]: `Module` has no attribute `track_running_stats`.
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

    def validate(self, model: nn.Module) -> bool:
        r"""Runs the validation on the model and all its submodules.


        Validation comprises a series of individual :class:`ModelInspectors <torchdp.utils.module_inspection.ModelInspector>`,
        each checking one predicate.
        Depending on ``should_throw`` flag in the constructor, will either return
        False or throw :class:`~torchdp.dp_model_inspector.IncompatibleModuleException` in case of validation failure.

        Note, that this method is called within :meth:`torchdp.privacy_engine.PrivacyEngine.attach`.

        Parameters
        ----------
            model: torch.nn.Module
                The model to validate.

        Returns
        ----------
        bool
            True if successful. False if validation fails and ``should_throw == False``

        Raises
        ------
        IncompatibleModuleException
            If the validation fails and ``should_throw == True``. Exception message will
            contain the details of validation failure reason.

        Example
        -------
            >>> insp = DPModelInspector()
            >>> valid_model = nn.Linear(16, 32)
            >>> is_valid = inspector.validate(model)
            >>> is_valid
            True
            >>> invalid_model = nn.BatchNorm1d(2)
            >>> is_valid = inspector.validate(model)
            # IncompatibleModuleException is thrown.
        """
        valid = all(inspector.validate(model) for inspector in self.inspectors)
        if self.should_throw and not valid:
            message = "Model contains incompatible modules."
            for inspector in self.inspectors:
                if inspector.violators:
                    message += f"\n{inspector.message}: {inspector.violators}"
            raise IncompatibleModuleException(message)
        return valid
