#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch import nn

from .autograd_grad_sample import is_supported
from .utils.module_inspection import ModelInspector, get_layer_type


class IncompatibleModuleException(Exception):
    r"""
    Exception class to be thrown in case
    the given model contains incompatible modules.
    """

    pass


class DPModelInspector:
    r"""
    Class to validate if a given module meets the requirements for attaching
     ``PrivacyEngine``.

    Active checks are listed in the ``DPModelInspector.inspectors`` attribute.
    """

    def __init__(self, should_throw: bool = True):
        r"""
        Args:
            should_throw: Whether the inspector should throw an exception or
                return False in case of validation error
        """
        self.should_throw = should_throw

        self.inspectors = [
            # Inspector to check model only consists of sub-modules we support
            ModelInspector(
                name="validity",
                predicate=_is_valid_check,
                message="Some modules are not valid.",
            ),
            # Inspector to check for BatchNorms as they could be replaced with groupnorm
            ModelInspector(
                name="batchnorm",
                predicate=_no_batchnorm_check,
                message="Model contains BatchNorm layers. It is recommended"
                "That they are replaced with GroupNorm.",
            ),
            # Inspector to check that instance norms doesn't track running stats
            ModelInspector(
                name="running_stats",
                predicate=_no_running_stats_instancenorm_check,
                message="InstanceNorm layer initialised with track_running_stats=True."
                "This is currently not supported",
            ),
            # Inspector to check the number of groups in Conv2d layers
            ModelInspector(
                name="conv_group_number",
                predicate=_conv_group_number_check,
                message="Number of groups in Conv2d layer must be either 1 or equal to number of channels",
            ),
            # Inspector to check for LSTM as it can be replaced with DPLSTM
            ModelInspector(
                name="lstm",
                predicate=_no_lstm,
                message="Model contains LSTM layers. It is recommended that they are"
                "replaced with DPLSTM",
            ),
        ]

    def validate(self, model: nn.Module) -> bool:
        r"""
        Runs the validation on the model and all its submodules.


        Validation comprises a series of individual
        :class:`ModelInspectors <opacus.utils.module_inspection.ModelInspector>`,
        each checking one predicate. Depending on ``should_throw`` flag in
        the constructor, will either return False or throw
        :class:`~opacus.dp_model_inspector.IncompatibleModuleException` in case of
        validation failure.

        Notes:
            This method is called in :meth:`opacus.privacy_engine.PrivacyEngine.attach`.

        Args:
            model: The model to validate.

        Returns:
            True if successful. False if validation fails and ``should_throw == False``

        Raises:
            IncompatibleModuleException
                If the validation fails and ``should_throw == True``. Exception message will
                contain the details of validation failure reason.

        Example:
            >>> inspector = DPModelInspector()
            >>> valid_model = nn.Linear(16, 32)
            >>> is_valid = inspector.validate(valid_model)
            >>> is_valid
            True
            >>> invalid_model = nn.BatchNorm1d(2)
            >>> is_valid = inspector.validate(invalid_model)
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


def _is_valid_check(module: nn.Module) -> bool:
    r"""
    Checks if the ``module``  is supported by ``autograd_grad_sample``
    """
    return is_supported(module)


def _conv_group_number_check(module: nn.Module) -> bool:
    r"""
    Checks if number of groups in `nn.Conv2d` layer is valid
    """
    if isinstance(module, nn.Conv2d):
        # pyre-fixme[16]: `Conv2d` has no attribute `in_channels`.
        return module.groups == 1 or module.groups == module.in_channels

    return True


def _no_batchnorm_check(module: nn.Module) -> bool:
    r"""
    Checks if the module is not BatchNorm.

    This check overlaps with `_is_valid_check`, but provides more targeted remedy.
    """
    return not isinstance(module, nn.modules.batchnorm._BatchNorm)


def _no_running_stats_instancenorm_check(module: nn.Module) -> bool:
    r"""
    Checks that InstanceNorm layer has `track_running_stats` set to False
    """
    is_instancenorm = get_layer_type(module) in (
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
    )

    if is_instancenorm:
        # pyre-fixme[16]: `Module` has no attribute `track_running_stats`.
        return not module.track_running_stats
    return True


def _no_lstm(module: nn.Module):
    is_lstm = True if get_layer_type(module) == "LSTM" else False

    return not is_lstm
