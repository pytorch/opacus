#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Borrowed from https://github.com/OpenMined/PySyft

set -e
TORCH_VERSION=$1

if [ "$TORCH_VERSION" = "1.4.0" ]
then
    TORCHVISION_VERSION="0.5.0"
elif [ "$TORCH_VERSION" = "1.5.0" ]
then
    TORCHVISION_VERSION="0.6.0"
elif [ "$TORCH_VERSION" = "1.5.1" ]
then
    TORCHVISION_VERSION="0.6.1"
elif [ "$TORCH_VERSION" = "1.6.0" ]
then
    TORCHVISION_VERSION="0.7"
    TORCHCSPRNG_VERSION="0.1.2"
elif [ "$TORCH_VERSION" = "1.7.0" ]
then
    TORCHVISION_VERSION="0.8.1"
    TORCHCSPRNG_VERSION="0.1.3"
elif [ "$TORCH_VERSION" = "1.7.1" ]
then
    TORCHVISION_VERSION="0.8.2"
    TORCHCSPRNG_VERSION="0.1.4"
fi
pip install torch=="${TORCH_VERSION}"
pip install torchvision==${TORCHVISION_VERSION}

# torchcsprng
if [ "$TORCH_VERSION" = "1.4.0" ]
then
    echo "No torchcsprng"
elif [ "$TORCH_VERSION" = "1.5.0" ]
then
    echo "No torchcsprng"
elif [ "$TORCH_VERSION" = "1.5.1" ]
then
    echo "No torchcsprng"
else
    pip install torchcsprng==${TORCHCSPRNG_VERSION}
fi
