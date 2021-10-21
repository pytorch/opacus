#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

set -e

PYTORCH_NIGHTLY=false
DEPLOY=false
CHOSEN_TORCH_VERSION=-1

while getopts 'ncdv:' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHTLY=true ;;
    c) CUDA=true ;;
    d) DEPLOY=true ;;
    v) CHOSEN_TORCH_VERSION=${OPTARG} ;;
    *) echo "usage: $0 [-n] [-d] [-v version]" >&2
       exit 1 ;;
    esac
  done

# NOTE: Only Debian variants are supported, since this script is only
# used by our tests on CircleCI.


sudo apt-get update
sudo apt-get upgrade ca-certificates -y
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs
curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt-get update && sudo apt-get install yarn

# yarn needs terminal info
export TERM=xterm

# NOTE: We don't use sudo for the below installs, because we expect these to come from pyenv which
# installs Python in a folder for which we have user access.

# upgrade pip
pip install --upgrade pip

# install with dev dependencies
pip install -e .[dev] --user

# install pytorch nightly if asked for
if [[ $PYTORCH_NIGHTLY == true ]]; then
  if [[ $CUDA == true ]]; then
    pip install --upgrade --pre torch torchvision torchcsprng -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
  else
    pip install --upgrade --pre torch torchvision torchcsprng -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
  fi
else
  # If no version specified, upgrade to latest release.
  if [[ $CHOSEN_TORCH_VERSION == -1 ]]; then
    pip install --upgrade torch
  else
    pip install torch=="$CHOSEN_TORCH_VERSION"
  fi
fi

# install deployment bits if asked for
if [[ $DEPLOY == true ]]; then
  pip install beautifulsoup4 ipython nbconvert
fi
