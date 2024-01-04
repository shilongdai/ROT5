#!/bin/bash

deepspeed train_mlm.py --output_dir="./rebert_minipile_128" --final_output_dir "./rebert_minipile_best_128" \
  --evaluation_strategy "steps" --per_device_train_batch_size 32 --per_device_eval_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --learning_rate 0.001 --num_train_epochs 1 --weight_decay 0.01 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_minipile" --logging_steps 500 \
  --save_steps 500 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end False \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile" --eval_name "validation" --tokenizer_name "mistralai/Mistral-7B-Instruct-v0.2"

deepspeed train_mlm.py --output_dir="./rebert_minipile_256" --final_output_dir "./rebert_minipile_best_256" \
  --evaluation_strategy "steps" --per_device_train_batch_size 64 --per_device_eval_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --learning_rate 0.0005 --num_train_epochs 1 --weight_decay 0.01 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_minipile" --logging_steps 250 \
  --save_steps 250 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end False \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile" --eval_name "validation" --model_path "./rebert_minipile_best_128"

deepspeed train_mlm.py --output_dir="./rebert_minipile_512" --final_output_dir "./rebert_minipile_best_512" \
  --evaluation_strategy "steps" --per_device_train_batch_size 128 --per_device_eval_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --learning_rate 0.0001 --num_train_epochs 1 --weight_decay 0.01 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_minipile" --logging_steps 125 \
  --save_steps 125 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end False \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile" --eval_name "validation" --model_path "./rebert_minipile_best_256"

deepspeed train_mlm.py --output_dir="./rebert_minipile_1600" --final_output_dir "./rebert_minipile_best_1600" \
  --evaluation_strategy "steps" --per_device_train_batch_size 200 --per_device_eval_batch_size 200 \
  --gradient_accumulation_steps 2 \
  --learning_rate 0.0001 --num_train_epochs 3 --weight_decay 0.01 --lr_scheduler_type "linear" \
  --logging_dir "tb_rebert_minipile" --logging_steps 100 \
  --save_steps 100 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end False \
  --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
  --dataset_path "data/minipile" --eval_name "validation" --model_path "./rebert_minipile_best_512"

