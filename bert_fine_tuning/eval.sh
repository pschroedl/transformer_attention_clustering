#!/bin/bash
python run_squad.py  \
    --model_name_or_path models/bert/  \
    --data_dir data/squad   \
    --output_dir data/eval \
    --overwrite_output_dir \
    --version_2_with_negative \
    --do_lower_case  \
    --do_eval   \
    --predict_file dev-v2.0.json   \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 3e-5   \
    --max_seq_length 384   \
    --doc_stride 128   \
    --threads 10   \