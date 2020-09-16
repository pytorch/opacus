#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Optional

import torch.nn as nn

from .grad_sample_module import GradSampleModule
from .module_checkers import DP_CHECKERS, ModuleChecker


class DPModule(GradSampleModule):
    r"""
    A DPModule is a nn.Module amenable to DP-SGD.
    This means that the following two conditions have to be satisfied:
    1. It is a valid GradSampleModule, ie you can compute per-sample gradients from it
    2. There is nothing prohibiting differential privacy in the module, or in any of its submodules,
     either in the design of the module or in its implementation.

    To explain 2, the simplest example is nn.BatchNorm. You *can* compute per-sample gradients for
    the affine transform inside nn.BatchNorm, so you can make it a GradSampleModule. However,
    the operations in BatchNorm introduce dependency between peers in a batch which is fundamentally
    incompatible with privacy. For this reason, BN is rejected.

    Another example is nn.MultiheadAttention. While there is nothing wrong with its design, the
    `torch.nn` implementation uses special submodules for performance's reason which don't work
    with our hooks for grad samples. For this reason, we reimplemented it (in DPMultiheadAttention)
    and this module will perform the necessary surgery to find the instances of nn.MultiheadAttention
    and replace them.
    """

    def __init__(self, module: nn.Module, strict: bool = False, verbose: bool = True):
        r"""
        Instantiates a DPModule from a nn.Module. This is done in two parts:
        1. Runs DPModuleAssistant with the indicated ``strict`` and ``verbose``
        2. Wraps the result in a GradSampleModule

        Parameters
        ----------
        module : torch.nn.Module
            [description]
        strict : bool, optional
            If True, will raise if something is wrong without trying to fix anything, by default False
        verbose : bool, optional
            Will print out what it did, by default True
        """
        self.checkers = DP_CHECKERS
        self.strict = strict
        self.verbose = verbose
        self.replace_modules(module)  # in-place

        super().__init__(module)  # Instantiates GradSampleModule

    def applicable_checker(self, module: nn.Module) -> Optional[ModuleChecker]:
        r"""
        Given a module, returns the relevant checker if there is one.

        Parameters
        ----------
        module : nn.Module
            The module to check

        Returns
        -------
        Optional[ModuleChecker]
            The relevant checker if there is one

        Raises
        ------
        ValueError
            At most a single checker can apply. This module will raise if multiple are returned.
        """
        checkers = [checker for checker in self.checkers if checker.is_watching(module)]
        if len(checkers) > 1:
            raise ValueError(
                f"Multiple checker returned for module {module}: \n\t{checkers}"
            )
        return checkers[0] if checkers else None

    def check_replacements(self, module: nn.Module) -> Optional[nn.Module]:
        """
        Checks `module` against the list of module to replace.

        Parameters
        ----------
        module : nn.Module
            The module to check against the list of module to replace.
        Returns
        -------
        Optional[Callable]
            If `module` is found, it returns that module's replacement function.
            Otherwise, it returns None.
        """
        checker = self.applicable_checker(module)
        if not checker:
            return None
        if self.strict:
            checker.validate(module)
            return None
        else:
            return (
                None
                if checker.is_valid(module)
                else checker.recommended_replacement(module)
            )

    def replace_modules(self, module: nn.Module) -> None:
        """
        Runs the checkers on `module`, and recurses on all its submodules.
        This will run IN-PLACE.

        Parameters
        ----------
        module : nn.Module
            The root module this checkers will operate on.
        """
        for attr, child in module.named_children():
            new_module = self.check_replacements(child)
            if new_module is not None:
                if self.verbose:
                    print(f"Replaced {child} with {new_module}")
                setattr(module, attr, new_module)
            else:
                self.replace_modules(child)
