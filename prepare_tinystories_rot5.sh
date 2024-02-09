#!/bin/bash

python prepare_instruct_data.py --tokenizer_name "./t5-rgqa" --model_max_length 512 \
  --final_output_dir "./data/tinystories_instruct_mistral" --sample 1000000
