#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ../examples/masked_run_highway_glue.py --model_type masked_albert \
  --model_name_or_path albert-base-v2 \
  --task_name SST-2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./glue_data/SST-2 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=1 \
  --per_gpu_train_batch_size=64 \
  --learning_rate 3e-5 \
  --num_train_epochs 30 \
  --overwrite_output_dir \
  --seed 42 \
  --output_dir ./saved_models/masked_albert/SST-2/two_stage_pruned_0.5 \
  --plot_data_dir ./plotting/ \
  --save_steps 0 \
  --overwrite_cache \
  --eval_after_first_stage \
  --adaptive \
  --adaptive_span_ramp 256 \
  --max_span 512 \
  --warmup_steps 1000 \
  --mask_scores_learning_rate 1e-2 \
  --initial_threshold 1 --final_threshold 0.5 \
  --initial_warmup 2 --final_warmup 3 \
  --pruning_method magnitude --mask_init constant --mask_scale 0. \
  --fxp_and_prune \
  --prune_percentile 60 \
  --teacher_type albert_teacher --teacher_name_or_path ./saved_models/albert-base/SST-2/teacher \
  --alpha_ce 0.1 --alpha_distil 0.9
