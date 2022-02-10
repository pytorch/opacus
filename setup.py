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

import os
import sys

from setuptools import find_packages, setup


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 7
REQUIRED_MICRO = 5

version = {}
with open("opacus/version.py") as fp:
    exec(fp.read(), version)

__version__ = version["__version__"]

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR, REQUIRED_MICRO):
    error = (
        "Your version of python ({major}.{minor}.{micro}) is too old. You need "
        "python >= {required_major}.{required_minor}.{required_micro}"
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        micro=sys.version_info.micro,
        required_major=REQUIRED_MAJOR,
        required_minor=REQUIRED_MINOR,
        required_micro=REQUIRED_MICRO,
    )
    sys.exit(error)


src_dir = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

requirements_txt = os.path.join(src_dir, "requirements.txt")
with open("requirements.txt", encoding="utf8") as f:
    required = f.read().splitlines()

with open("dev_requirements.txt", encoding="utf8") as f:
    dev_required = f.read().splitlines()

setup(
    name="opacus",
    version=__version__,
    author="The Opacus Team",
    description="Train PyTorch models with Differential Privacy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://opacus.ai",
    project_urls={
        "Documentation": "https://opacus.ai/api",
        "Source": "https://github.com/pytorch/opacus",
    },
    license="Apache-2.0",
    install_requires=required,
    extras_require={"dev": dev_required},
    packages=find_packages(),
    keywords=[
        "PyTorch",
        "Differential Privacy",
        "DP-SGD",
        "DP SGD",
        "Privacy Preserving Machine Learning",
        "PPML",
        "PPAI",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=f">={REQUIRED_MAJOR}.{REQUIRED_MINOR}.{REQUIRED_MICRO}",
)
