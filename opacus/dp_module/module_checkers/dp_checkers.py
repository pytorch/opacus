#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .batch_norm import BatchNormChecker
from .conv2d import Conv2dChecker
from .instance_norm import InstanceNormChecker
from .multihead_attention import MultiheadAttentionChecker


DP_CHECKERS = [
    BatchNormChecker(),
    Conv2dChecker(),
    InstanceNormChecker(),
    MultiheadAttentionChecker(),
]
