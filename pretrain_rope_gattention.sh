#!/bin/bash

deepspeed train_mlm.py --output_dir="./rebert_rope" --final_output_dir "./rebert_rope_best" \
  --evaluation_strategy "steps" --per_device_train_batch_size 200 --per_device_eval_batch_size 200 \
  --gradient_accumulation_steps 3 --max_grad_norm 0.5 \
  --learning_rate 0.001 --num_train_epochs 6 --weight_decay 0.01 --warmup_ratio 0.02 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_rope" --logging_steps 50 \
  --save_steps 50 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile" --eval_name "validation" --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2"
