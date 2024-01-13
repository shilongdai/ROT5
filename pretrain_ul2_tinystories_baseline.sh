#!/bin/bash

 deepspeed train_ul2.py --output_dir="./t5_tinystories" --final_output_dir "./t5_tinystories_best" \
   --evaluation_strategy "steps" --per_device_train_batch_size 150 --per_device_eval_batch_size 150 \
   --gradient_accumulation_steps 1 \
   --learning_rate 0.0005 --num_train_epochs 1 --weight_decay 0.01 --warmup_ratio 0.01 --lr_scheduler_type "linear" \
   --logging_dir "tb_t5_tinystories" --logging_steps 100 --generation_max_length 512 \
   --save_steps 100 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
   --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
   --dataset_path "data/tinystories_mistral" --eval_name "validation" \
   --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2"
