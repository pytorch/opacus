#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import sys
from collections import OrderedDict
from typing import Iterable, List

import torch
import torch.nn as nn


logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def parametrized_modules(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters (as opposed to "wrapper modules" that just organize modules).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in module.named_modules()
        if any(p is not None for p in m.parameters(recurse=False))
    )


def trainable_modules(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters and are trainable (ie they want a grad).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in parametrized_modules(module)
        if any(p.requires_grad for p in m.parameters(recurse=False))
    )


def requires_grad(module: nn.Module, *, recurse: bool = False) -> bool:
    """
    Checks if any parameters in a specified module require gradients.

    Args:
        module: PyTorch module whose parameters are to be examined.
        recurse: Flag specifying if the gradient requirement check should
            be applied recursively to sub-modules of the specified module

    Returns:
        Flag indicate if any parameters require gradients
    """
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def clone_module(module: nn.Module) -> nn.Module:
    """
    Handy utility to clone an nn.Module. PyTorch doesn't always support copy.deepcopy(), so it is
    just easier to serialize the model to a BytesIO and read it from there.

    Args:
        module: The module to clone

    Returns:
        The clone of ``module``
    """
    with io.BytesIO() as bytesio:
        torch.save(module, bytesio)
        bytesio.seek(0)
        module_copy = torch.load(bytesio)
    return module_copy


def get_submodule(module: nn.Module, target: str) -> nn.Module:
    """
    Returns the submodule given by target if it exists, otherwise throws an error.

    This is copy-pasta of Pytorch 1.9's ``get_submodule()`` implementation; and is
    included here to also support Pytorch 1.8. This function can be removed in favour
    of ``module.get_submodule()`` once Opacus abandons support for torch 1.8.

    See more details at https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=get_submodule#torch.nn.Module.get_submodule

    Args:
        module: module
        target: submodule string

    Returns:
        The submodule given by target if it exists

    Raises:
        AttributeError
            If submodule doesn't exist
    """

    if target == "":
        return module

    atoms: List[str] = target.split(".")
    mod: nn.Module = module

    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(
                mod._get_name() + " has no " "attribute `" + item + "`"
            )
        mod = getattr(mod, item)
        if not isinstance(mod, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")
    return mod


def are_state_dict_equal(sd1: OrderedDict, sd2: OrderedDict):
    """
    Compares two state dicts, while logging discrepancies
    """
    if len(sd1) != len(sd2):
        logger.error(f"Length mismatch: {len(sd1)} vs {len(sd2)}")
        return False

    for k1, v1 in sd1.items():
        # check that all keys are accounted for.
        if k1 not in sd2:
            logger.error(f"Key missing: {k1} not in {sd2}")
            return False
        # check that value tensors are equal.
        v2 = sd2[k1]
        if not torch.allclose(v1, v2):
            logger.error(f"Tensor mismatch: {v1} vs {v2}")
            return False
    return True
