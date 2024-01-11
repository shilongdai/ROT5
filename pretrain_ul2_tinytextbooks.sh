#!/bin/bash

 deepspeed train_ul2.py --output_dir="./rebert_tinytextbooks" --final_output_dir "./rebert_tinytextbooks_best" \
   --evaluation_strategy "steps" --per_device_train_batch_size 150 --per_device_eval_batch_size 150 \
   --gradient_accumulation_steps 1 \
   --learning_rate 0.0001 --num_train_epochs 2 --weight_decay 0.01 --warmup_ratio 0.01 --lr_scheduler_type "linear" \
   --logging_dir "tb_rebert_tinytextbooks" --logging_steps 100 --generation_max_length 512 \
   --save_steps 100 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
   --bf16 True --gradient_checkpointing True --deepspeed "./deepspeed/deepspeed_2.json" \
   --dataset_path "data/tinytextbooks_mistral" --eval_name "validation" \
   --model_path "./rebert_tinystories_best"
