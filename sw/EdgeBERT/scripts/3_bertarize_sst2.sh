#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ../examples/bertarize.py \
    --pruning_method magnitude \
    --threshold 0.5 \
    --model_name_or_path ./saved_models/masked_albert/SST-2/two_stage_pruned_0.5 \
