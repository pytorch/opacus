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


usage()
{
    echo "usage Example: bash make_small_imagenet_N_classes.sh -o Documents/imagenet_full_size -d Documents/imagenet_small_50_classes -n 50"
    echo "usage: bash make_small_imagenet_N_classes.sh -o <path to the orignal ImageNet> -d <path of a new folder (will be created) where a sampled version of imagenet will be copied to> -n <Number of desired classes (in each train and val) that will be copied out of 1000 classes (maximum value is 1000)>"
}

if [ "$1" == "" ]; then # If arguments are not specified
    usage
    exit 1
fi
while [ "$1" != "" ]; do # Read the input arguments
    case $1 in
        -d | --destination )    shift
                                destination=$1
                                ;;
        -o | --origin )         shift 
                                origin=$1
                                ;;
        -n | --N )              shift
                                N=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     echo "ERROR: unknown parameter \"$1\""
                                usage
                                exit 1
    esac
    shift
done

# If all necessary arguments are not supplied
if [[ -z $destination || -z $origin || -z $N ]] 
then
    echo "You must specify all necessary parameters."
    usage
    exit 1
fi

# Get absolute path
destination="$(readlink -f $destination)"
origin="$(readlink -f $origin)"

mkdir "$destination"
mkdir "$destination/train"
mkdir "$destination/val"

echo 'Copying'
for val_train_folder in val train; do # Do copying for both 'train' and 'val' folders
    cd "$origin/$val_train_folder" || { echo "Failure"; exit 1; } # change directory to origin's train or val folders
    find . -maxdepth 1 -mindepth 1 | head -n "$N" | xargs cp -ir -t "$destination/$val_train_folder" # select and copy N classes
    echo "Copying folder $val_train_folder is done."
done
