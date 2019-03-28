#!/usr/bin/env bash

prefix="configs/"
file_name=$1
ext=".yaml"

config_file=$prefix$file_name$ext
if [ -z "$1" ]
  then
    echo "No config_file (.yaml) argument."
    exit
else
    echo $config_file
fi

DEVICE_ID=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --config_name=None \
    --debug=0 \
    --config_file=$config_file


