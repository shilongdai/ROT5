#!/bin/bash

python prepare_sft_data.py --tokenizer_name "./t5-baseline" \
  --final_output_dir "./data/tinystories_instruct_base" --sample 1000000
