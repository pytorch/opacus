#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from abc import ABC, abstractmethod
from typing import Sequence

import torch.nn as nn


"""
It is not enough to be a GradSampleModule: some modules support or can support per-sample
gradients but do not support differential privacy. For example, BatchNorm's coefficients can be
made to support per-sample gradients, but the very nature of BatchNorm makes it incompatible with
differential privacy.

For this reason, we separate GradSampleModules from DP Modules and we put utilities to do
further conversion to DP Modules here.

"""


class ModuleChecker(ABC):
    def __init__(self, watched_modules: Sequence[type]):
        self.watched_modules = set(watched_modules)

    @abstractmethod
    def is_valid(self, module: nn.Module) -> bool:
        """
        Validates a module, without raising an exception.
        Run this function on a module


        Parameters
        ----------
        module : nn.Module
            The module to validate

        Returns
        -------
        bool
            True if it's a valid instance of the watched_modules, or False otherwise
        """
        pass

    @abstractmethod
    def recommended_replacement(self, module: nn.Module) -> nn.Module:
        """
        Returns a recommended replacement for this module, for a module that is valid instead.

        Parameters
        ----------
        module : nn.Module
            The module to replace

        Returns
        -------
        nn.Module
            The new module
        """
        pass

    def is_watching(self, module: nn.Module) -> bool:
        return type(module) in self.watched_modules

    def validate(self, module: nn.Module) -> None:
        """
        Similar to `is_valid`, but will raise if a module is not invalid.
        Override this to have properly made exceptions.

        Parameters
        ----------
        module : nn.Module
            The module to validate

        Raises
        ------
        ValueError
            If the module is not valid.
        """
        if self.is_watching(module) and not self.is_valid(module):
            raise ValueError(f"The module {module} is not valid!")
