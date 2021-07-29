#!/bin/bash
python run_squad.py  \
    --model_type bert   \
    --model_name_or_path bert-base-uncased  \
    --output_dir models/bert/ \
    --data_dir data/squad   \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train  \
    --train_file train-v2.0.json   \
    --version_2_with_negative \
    --do_lower_case  \
    --do_eval   \
    --predict_file dev-v2.0.json   \
    --per_gpu_train_batch_size 4   \
    --learning_rate 3e-5   \
    --num_train_epochs 2.0   \
    --max_seq_length 384   \
    --doc_stride 128   \
    --threads 10   \
    --save_steps 5000 