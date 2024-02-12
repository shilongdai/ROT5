#!/bin/bash

deepspeed train_sft.py --output_dir="./t5-sum-real" --final_output_dir "./t5-sum-real-best" \
   --evaluation_strategy "steps" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 --predict_with_generate True \
   --learning_rate 0.0001 --num_train_epochs 1 --weight_decay 0.01 --warmup_ratio 0.01 --lr_scheduler_type "linear" \
   --logging_dir "tb-t5-sum-real" --logging_steps 300 --generation_max_length 128 \
   --save_steps 300 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
   --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
   --dataset_path "data/tinystories_instruct_base" --eval_name "validation" \
   --model_path "google/flan-t5-base"
