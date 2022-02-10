#!/usr/bin/env python3
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


class UnsupportedError(ValueError):
    """
    Raised for unsupported components in Opacus.
    """

    pass


class UnsupportedModuleError(UnsupportedError):
    """
    Raised for unsupported modules in Opacus.
    """


class UnsupportableModuleError(UnsupportedModuleError):
    """
    Raised whenever there is a module we can't support ever.
    BatchNorm is the largest offender.
    """

    pass


class NotYetSupportedModuleError(UnsupportedModuleError):
    """
    Raised whenever there is a module that we don't yet support.
    This is the "catch-all": the number of modules we won't ever support
    is very short, so a priori if we don't support it now it doesn't mean
    we can't extend support later (and PRs are welcome!!).
    """

    pass


class ShouldReplaceModuleError(UnsupportedModuleError):
    """
    Raised whenever there is a module that we don't support as-is but we do support via
    replacement (and we have made a replacement ourselves).
    """

    pass


class IllegalModuleConfigurationError(UnsupportedModuleError):
    """
    Raised whenever there is an illegal configuration of the training setup.
    Examples include, not setting the module to training mode, enabling running stats
    in InstanceNorm, etc.
    """

    pass
