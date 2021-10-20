#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from functools import partial
from typing import Iterable, List, Tuple

import torch.nn as nn
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.utils.module_utils import trainable_modules
from opacus.validators.errors import (
    IllegalConfigurationError,
    NotYetSupported,
    UnsupportedError,
)

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ModuleValidator:
    """
    Encapsulates all the validation logic required by Opacus.
    Also works as a namespace to hold registered validators and fixers.
    """

    VALIDATORS = {}
    FIXERS = {}

    @classmethod
    def _check_for_validation_errors(cls, module: nn.Module) -> List[UnsupportedError]:
        """
        Validate module and sub_modules by running registered custom validators.
        Also perform additional validation to ensure DP compatible training.

        Args:
            module: The root module to validate.

        Returns:
            List of Errors encountered during validation.
            Empty list in case of successful of validation.
        """
        errors = []
        # 1. validate that module is in trainig mode
        if not module.training:
            errors.append(
                IllegalConfigurationError("Model needs to be in training mode")
            )
        # 2. validate that all trainable modules are supported by GradSampleModule.
        errors.extend(
            [
                NotYetSupportedError(f"grad sampler is not yet implemented for {m}")
                for m in trainable_modules(module)
                if not GradSampleModule.is_supported(m)
            ]
        )
        # 3. perform module specific validations.
        for _, sub_module in module.named_children():
            if type(sub_module) in ModuleValidator.VALIDATORS:
                sub_module_validator = ModuleValidator.VALIDATORS[type(sub_module)]
                errors.extend(sub_module_validator(sub_module))
        return errors


    @classmethod
    def is_valid(cls, module: nn.Module) -> bool:
        """
        Check if module and sub_modules are valid by running registered custom validators.

        Args:
            module: The root module to validate.

        Returns:
            bool
        """
        return len(cls._check_for_validation_errors(module) == 0)


    @classmethod
    def validate(cls, module: nn.Module) -> None:
        """
        Validate module and sub_modules by running registered custom validators.
        This is same as ``is_valid()``, but throws the excpetions rather than returning
        a boolean.

        Args:
            module: The root module to validate.

        Raises:
            List of UnsupportedError in case of validation failures.
        """
        errors = cls._check_for_validation_errors(module)
        # raise Error if applicable
        if len(errors) > 0:
            raise UnsupportedError(errors)


    @classmethod
    def fix_(cls, module: nn.Module) -> None:
        """
        Make the module and sub_modules DP compatible by running registered custom fixers.

        Args:
            module: The root module to be made compatible.

        Returns:
            None. Fix happens in place.
        """
        for sub_module_name, sub_module in module.named_children():
            if type(sub_module) in ModuleValidator.FIXERS:
                sub_module_fixer = ModuleValidator.FIXERS[type(sub_module)]
                new_sub_module = sub_module_fixer(sub_module)
                setattr(module, sub_module_name, new_sub_module)
                logger.info(
                    f"Replaced `{sub_module_name}` from {sub_module} to {new_sub_module}"
                )

    @classmethod
    def fix_and_validate_(cls, module: nn.Module) -> None:
        """
        Fix the module and sub_modules first, and then run validation.

        Args:
            module: The root module to be fixed and validted

        Returns:
            None. Fix happens in place.

        Raises:
            List of UnsupportedError in case of validation failures.
        """

        errors = []
        # 1. replace any fixable modules
        cls.fix_(module)
        # 2. perform module specific validations.
        cls.validate(module)
