#!/bin/bash

python prepare_pretrain_data.py --dataset_path "JeanKaddour/minipile" \
  --tokenizer_name "roberta-base" --model_max_length 512 --final_output_dir "./data/minipile_roberta"
