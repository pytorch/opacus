#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import setuptools


src_dir = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements_txt = os.path.join(src_dir, "requirements.txt")
with open("requirements.txt") as f:
    required = f.read().splitlines()


setuptools.setup(
    name="pytorch-dp",
    version="0.1-beta.1",
    author="PyTorch Team",
    description="Train PyTorch models with Differential Privacy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/pytorch-dp",
    license="Apache-2.0",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
