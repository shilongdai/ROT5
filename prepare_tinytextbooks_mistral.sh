#!/bin/bash

python prepare_pretrain_data.py --dataset_path "nampdn-ai/tiny-textbooks" \
  --tokenizer_name "./rebert_tinystories_best" --text_field "textbook" --model_max_length 512 \
  --final_output_dir "./data/tinytextbooks_mistral" --dataset_split "train" --pack False
