#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ../examples/run_highway_glue.py --model_type albert \
  --model_name_or_path albert-base-v2 \
  --task_name SST-2 \
  --do_train \
  --fine-tune \
  --do_eval \
  --do_lower_case \
  --data_dir ./glue_data/SST-2 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=1 \
  --per_gpu_train_batch_size=64 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --seed 42 \
  --output_dir ./saved_models/highway_albert/SST-2 \
  --plot_data_dir ./plotting/ \
  --save_steps 0 \
  --overwrite_cache \
  --evaluate_during_training \
  --eval_after_first_stage \
  --warmup_steps 1000 \
