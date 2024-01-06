#!/bin/bash

deepspeed migrate_mlm.py --output_dir="./rebert_g4" --final_output_dir "./rebert_g4_best" \
  --evaluation_strategy "steps" --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 2 \
  --learning_rate 0.0007 --max_steps 8000 --weight_decay 0.01 --warmup_ratio 0.01 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_g4" --logging_steps 50 \
  --save_steps 50 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile_roberta" --eval_name "validation" \
  --num_kv_heads 4 --model_name "./rebert-roberta-2"
