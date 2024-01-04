#!/bin/bash

python prepare_pretrain_data.py --dataset_path "wikimedia/wikisource" \
  --dataset_subset "20231201.en" --dataset_split "train" \
  --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2" --model_max_length 512 \
  --final_output_dir "./data/wikisource_mistral"
