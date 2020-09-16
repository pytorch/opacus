#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from opacus.dp_module.modules import DPLSTM, DPMultiheadAttention, SequenceBias

from .conv1d import compute_grad_sample as conv1d_grad_sample
from .conv2d import compute_grad_sample as conv2d_grad_sample
from .dp_lstm import compute_grad_sample as dplstm_grad_sample
from .embedding import compute_grad_sample as embedding_grad_sample
from .group_norm import compute_grad_sample as group_norm_grad_sample
from .instance_norm import compute_grad_sample as instance_norm_grad_sample
from .layer_norm import compute_grad_sample as layer_norm_grad_sample
from .linear import compute_grad_sample as linear_grad_sample
from .sequence_bias import compute_grad_sample as sequence_bias_grad_sample


GRAD_SAMPLERS = {
    nn.Conv1d: conv1d_grad_sample,
    nn.Conv2d: conv2d_grad_sample,
    nn.Embedding: embedding_grad_sample,
    nn.GroupNorm: group_norm_grad_sample,
    nn.InstanceNorm1d: instance_norm_grad_sample,
    nn.InstanceNorm2d: instance_norm_grad_sample,
    nn.InstanceNorm3d: instance_norm_grad_sample,
    nn.LayerNorm: layer_norm_grad_sample,
    nn.Linear: linear_grad_sample,
    SequenceBias: sequence_bias_grad_sample,
    DPLSTM: dplstm_grad_sample,
}

SUPPORTED_MODULES = set(GRAD_SAMPLERS.keys())

__all__ = [
    "GRAD_SAMPLERS",
    "SUPPORTED_MODULES",
    "NotYetSupportedModuleError",
    "UnsupportableModuleError",
]
