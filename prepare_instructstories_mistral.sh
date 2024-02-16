#!/bin/bash

python prepare_pretrain_data.py --dataset_path "skeskinen/TinyStories-Instruct-hf" \
  --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2" --model_max_length 10000000000 \
  --final_output_dir "./data/tinystories_instruct_mistral" --pack False
