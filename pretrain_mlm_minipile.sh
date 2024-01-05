#!/bin/bash

deepspeed train_mlm.py --output_dir="./rebert_minipile" --final_output_dir "./rebert_minipile_best" \
  --evaluation_strategy "steps" --per_device_train_batch_size 232 --per_device_eval_batch_size 232 \
  --gradient_accumulation_steps 1 \
  --learning_rate 0.0007 --max_steps 130000 --weight_decay 0.01 --warmup_ratio 0.01 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_minipile" --logging_steps 500 \
  --save_steps 500 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile" --eval_name "validation" --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2"
