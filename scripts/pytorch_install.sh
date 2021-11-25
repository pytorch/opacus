#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Borrowed from https://github.com/OpenMined/PySyft

set -e
TORCH_VERSION=$1

if [ "$TORCH_VERSION" = "1.8.0" ]
then
    TORCHVISION_VERSION="0.9.0"
    TORCHCSPRNG_VERSION="0.2.0"
elif [ "$TORCH_VERSION" = "1.8.1" ]
then
    TORCHVISION_VERSION="0.9.1"
elif [ "$TORCH_VERSION" = "1.9.0" ]
then
    TORCHVISION_VERSION="0.10.0"
elif [ "$TORCH_VERSION" = "1.9.1" ]
then
    TORCHVISION_VERSION="0.10.1"
fi

pip install torch=="${TORCH_VERSION}"
pip install torchvision==${TORCHVISION_VERSION}

# torchcsprng
if [ "$TORCH_VERSION" = "1.8.0" ]
then
    pip install torchcsprng==${TORCHCSPRNG_VERSION}
else
    echo "No torchcsprng"
fi
