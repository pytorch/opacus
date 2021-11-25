#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Borrowed from https://github.com/OpenMined/PySyft

[string]$TORCH_VERSION=$args[0]
If ($TORCH_VERSION -eq "1.8.0") {
  $TORCHVISION_VERSION="0.9.0"
  $TORCHCSPRNG_VERSION="0.2.0"
} Elseif ( $TORCH_VERSION -eq "1.8.1" ) {
  $TORCHVISION_VERSION="0.9.1"
} Elseif ( $TORCH_VERSION -eq "1.9.0" ) {
  $TORCHVISION_VERSION="0.10.0"
} Elseif ($TORCH_VERSION -eq "1.9.1") {
  $TORCHVISION_VERSION="0.10.1"
}
pip install torch==$TORCH_VERSION+cpu torchvision==$TORCHVISION_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html

If ($TORCH_VERSION -eq "1.8.0") {
  pip install torchcsprng==$TORCHCSPRNG_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html
} Else {
  echo "No torchcsprng"
}
