#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

set -e

PYTORCH_NIGHTLY=false
DEPLOY=false
CHOSEN_TORCH_VERSION=-1

while getopts 'ncdv:' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHTLY=true ;;
    c) CUDA=true;;
    d) DEPLOY=true ;;
    v) CHOSEN_TORCH_VERSION=${OPTARG};;
    *) echo "usage: $0 [-n] [-d] [-v version]" >&2
       exit 1 ;;
    esac
  done

# NOTE: Only Debian variants are supported, since this script is only
# used by our tests on CircleCI.

# install nodejs and yarn for website build
sudo apt install apt-transport-https ca-certificates
curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update
sudo apt install nodejs
sudo apt install yarn

# yarn needs terminal info
export TERM=xterm

# NOTE: All of the below installs use sudo, b/c otherwise pip will get
# permission errors installing in the docker container. An alternative would be
# to use a virtualenv, but that would lead to bifurcation of the CircleCI config
# since we'd need to source the environemnt in each step.

# upgrade pip
sudo pip install --upgrade pip

# install with dev dependencies
sudo pip install -e .[dev]

# install pytorch nightly if asked for
if [[ $PYTORCH_NIGHTLY == true ]]; then
  if [[ $CUDA == true ]]; then
    sudo pip install --upgrade --pre torch torchvision torchcsprng -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
  else
    sudo pip install --upgrade --pre torch torchvision torchcsprng -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
  fi
else
  # If no version specified, upgrade to latest release.
  if [[ $CHOSEN_TORCH_VERSION == -1 ]]; then
    sudo pip install --upgrade torch
  else
    sudo pip install torch==$CHOSEN_TORCH_VERSION
  fi
fi

# install deployment bits if asked for
if [[ $DEPLOY == true ]]; then
  sudo pip install beautifulsoup4 ipython nbconvert
fi
