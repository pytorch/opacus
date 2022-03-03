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
