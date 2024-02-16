#!/bin/bash

python prepare_sft_data.py --tokenizer_name "./t5-sum-rgqa" \
  --final_output_dir "./data/tinystories_instruct_mistral" --sample 1000000
