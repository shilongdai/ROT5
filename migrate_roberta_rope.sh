#!/bin/bash

python migrate_mlm.py --output_dir="./rebert_rope" --final_output_dir "./rebert_rope_best" \
  --evaluation_strategy "steps" --per_device_train_batch_size 20 --per_device_eval_batch_size 20 \
  --learning_rate=1e-4 --num_train_epochs 4 --weight_decay 0.01 --warmup_steps 200 --lr_scheduler_type "cosine" \
  --logging_dir "tb_rebert_rope" --logging_steps 500 \
  --save_steps 2500 --save_strategy "steps" --save_total_limit 10 --load_best_model_at_end True \
  --bf16 True --gradient_checkpointing True \
  --dataset_path "data/en_wikisource"
