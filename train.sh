#!/bin/sh

base_dir="/directory/contains/model_and_dataset"
model_name="Llama-2-7b-chat-hf"
dataset_name="guanaco-llama2-1k"
new_model_name="llama-2-7b-chat-guanaco"

python train.py -m $base_dir/$model_name -d $base_dir/$dataset_name -s $new_model_name
