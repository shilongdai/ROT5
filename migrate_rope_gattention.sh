#!/bin/bash

deepspeed migrate_mlm.py --output_dir="./rebert_roberta" --final_output_dir "./rebert_roberta_best" \
  --evaluation_strategy "steps" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 64 --max_grad_norm 0.5 \
  --learning_rate 0.001 --num_train_epochs 2 --weight_decay 0.01 --warmup_ratio 0.06 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_roberta" --logging_steps 50 \
  --save_steps 50 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile_roberta" --eval_name "validation"
