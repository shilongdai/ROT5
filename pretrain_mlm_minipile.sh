#!/bin/bash

 deepspeed train_mlm.py --output_dir="./rebert_minipile" --final_output_dir "./rebert_minipile_best" \
   --evaluation_strategy "steps" --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0002 --max_steps 6000 --weight_decay 0.01 --warmup_ratio 0.06 --lr_scheduler_type "linear" \
   --logging_dir "tb_rebert_minipile" --logging_steps 50 \
   --save_steps 50 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
   --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
   --dataset_path "data/minipile_roberta" --eval_name "validation" --model_path "./rebert_roberta_best"

deepspeed train_mlm.py --output_dir="./rebert_minipile_2" --final_output_dir "./rebert_minipile_2best" \
  --evaluation_strategy "steps" --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 2 \
  --learning_rate 0.00002 --max_steps 2600 --weight_decay 0.01 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_minipile2" --logging_steps 50 \
  --save_steps 50 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile_roberta" --eval_name "validation" --model_path "./rebert_minipile_best"
