#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import sys
from collections import OrderedDict
from typing import Iterable

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
        m
        for m in module.modules()
        if any(p is not None for p in m.parameters(recurse=False))
    )


def trainable_modules(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters and are trainable (ie they want a grad).
    """
    yield from (
        m
        for m in parametrized_modules(module)
        if any(p.requires_grad for p in m.parameters(recurse=False))
    )


def requires_grad(module: nn.Module, recurse: bool = False) -> bool:
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


def are_state_dict_equal(sd1: OrderedDict, sd2: OrderedDict):
    if len(sd1) != len(sd2):
        logger.error(f"Length mismatch: {len(sd1)} vs {len(sd2)}")
        return False

    for k_1, v_1 in sd1.items():
        # keys are accounted for
        if k_1 not in sd2:
            logger.error(f"Key missing: {k_1} not in {sd2}")
            return False
        # value tensors are equal
        v_2 = sd2[k_1]
        if not torch.allclose(v_1, v_2):
            logger.error(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True
