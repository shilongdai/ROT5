#!/bin/bash

python prepare_instruct_data.py --tokenizer_name "./t5-rgqa" \
  --final_output_dir "./data/tinystories_instruct_mistral" --sample 1000000
