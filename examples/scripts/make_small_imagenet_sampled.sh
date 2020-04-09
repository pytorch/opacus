#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# ===================
# This script makes a sampled version of ImageNet, with all 1000 classes.
# ===================

usage()
{
    echo "usage Example: bash make_small_imagenet_N_classes.sh -o Documents/imagenet_full_size -d Documents/imagenet_small_100 -nt 100 -nv 10"
    echo "usage: bash make_small_imagenet_N_classes.sh -o <path to the orignal ImageNet> -d <path of a new folder (will be created) where a sampled version of imagenet will be copied to> -nt <Number of desired samples (images) in train folder> -nv <Number of desired samples (images) in val folder>"
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
        -nt )                   shift
                                N_train=$1
                                ;;
        -nv )                   shift
                                N_val=$1
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
if [[ -z $destination || -z $origin || -z $N_train || -z $N_val ]] 
then
    echo "You must specify all necessary parameters."
    usage
    exit 1
fi

# Get absolute paths
destination="$(readlink -f "$destination")"
origin="$(readlink -f "$origin")"

mkdir "$destination"
mkdir "$destination/train"
mkdir "$destination/val"

echo 'Copying'
for val_train_folder in val train; do # Do copying for both 'train' and 'val' folders
    if [[ $val_train_folder = 'train' ]]; then
        N=$N_train   
    else
        N=$N_val   
    fi
    cd "$origin/$val_train_folder" || { echo "Failure"; exit 1; } # change directory to origin's train or val folders
    for d in */ ; do # loop through the 1000 folders 
        mkdir "$destination/$val_train_folder/$d"
        cd "$origin/$val_train_folder/$d" || { echo "Failure"; exit 1; } 
        find . -maxdepth 1 -mindepth 1 -type f |sort -R |tail -"$N" |while read -r file; do # select N files from each 1000 folders
            cp "$file" "$destination/$val_train_folder/$d/$file"
            printf "."
        done
    done
    echo "Copying folder $val_train_folder is done."
done
