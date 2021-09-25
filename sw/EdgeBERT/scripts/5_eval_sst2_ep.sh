#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ENTROPIES="0.09 0.16 0.28"

for ENTROPY in $ENTROPIES; do
    echo $ENTROPY
    python ../examples/masked_run_highway_glue.py --model_type albert \
      --model_name_or_path ./saved_models/masked_albert/SST-2/bertarized_two_stage_pruned_0.6 \
      --task_name SST-2 \
      --do_eval \
      --do_lower_case \
      --data_dir ./glue_data/SST-2 \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size=1 \
      --overwrite_output_dir \
      --output_dir ./saved_models/masked_albert/SST-2/bertarized_two_stage_pruned_0.6  \
      --plot_data_dir ./plotting/ \
      --early_exit_entropy $ENTROPY \
      --eval_highway \
      --entropy_predictor \
      --predict_layer 1 \
      --lookup_table_file ./sst2_lookup_table_opt.csv \
      --overwrite_cache
done
