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
If ($TORCH_VERSION -eq "2.6.0") {
  $TORCHVISION_VERSION="0.21.0"
}
pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION -f https://download.pytorch.org/whl/torch_stable.html
