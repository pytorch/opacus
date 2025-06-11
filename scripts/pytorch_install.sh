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

set -e
TORCH_VERSION=$1

if [ "$TORCH_VERSION" = "2.6.0" ]
then
    TORCHVISION_VERSION="0.21.0"
fi

pip install torch=="${TORCH_VERSION}" --extra-index-url https://download.pytorch.org/whl/cpu
pip install torchvision==${TORCHVISION_VERSION} --extra-index-url https://download.pytorch.org/whl/cpu
