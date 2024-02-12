#!/bin/bash

 deepspeed train_ul2.py --output_dir="./grot5_tinystories_mini" --final_output_dir "./grot5_tinystories_bestmini" \
   --evaluation_strategy "steps" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 20 \
   --learning_rate 0.0005 --num_train_epochs 2 --weight_decay 0.01 --warmup_ratio 0.01 --lr_scheduler_type "linear" \
   --logging_dir "tb_grot5_tinystories_mini" --logging_steps 50 --generation_max_length 512 \
   --save_steps 50 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
   --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
   --dataset_path "data/tinystories_mistral" --eval_name "validation" \
   --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2" --kv_heads 4 --d_model 512 --d_ff 512 --num_heads 16 \
   --d_kv 32 --num_layers 8
