#!/bin/bash

python prepare_pretrain_data.py --dataset_path "JeanKaddour/minipile" \
  --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2" --model_max_length 512 \
  --final_output_dir "./data/minipile_mistral" --pack False
