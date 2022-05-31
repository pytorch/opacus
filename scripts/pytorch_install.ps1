#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
