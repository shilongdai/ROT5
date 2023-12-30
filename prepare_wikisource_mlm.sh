#!/bin/bash

python prepare_pretrain_data.py --dataset_path "wikimedia/wikisource" \
  --dataset_subset "20231201.en" --dataset_split "train" \
  --model_name "roberta-base" --model_max_length 512 --final_output_dir "./data/en_wikisource"
