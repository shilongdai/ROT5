import random
from dataclasses import field, dataclass
from typing import Optional, cast

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, \
    HfArgumentParser, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from rot5 import (ROT5ForConditionalGeneration)


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="./t5-rgqa")
    dataset_path: Optional[str] = field(default="./data/mlm")
    train_name: Optional[str] = field(default="train")
    eval_name: Optional[str] = field(default="eval")
    eval_sample: Optional[int] = field(default=500)
    cache_dir: Optional[str] = field(default="./transformers_cache")
    final_output_dir: Optional[str] = field(default="./best_instruct_model")
    aux_loss: Optional[bool] = field(default=False)


if __name__ == "__main__":
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    rot5 = ROT5ForConditionalGeneration.from_pretrained(script_args.model_path)
    sink_token = rot5.config.decoder_start_token_id

    if train_args.gradient_checkpointing:
        rot5.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=rot5)
    print(f"Loading data from: {script_args.dataset_path}")
    ds = load_from_disk(script_args.dataset_path)
    train_set = ds[script_args.train_name]
    eval_set = None
    if script_args.eval_name in ds:
        eval_set = ds[script_args.eval_name]
        if len(eval_set) > script_args.eval_sample:
            idx = np.random.choice(len(eval_set), script_args.eval_sample)
            eval_set = eval_set.select(idx).flatten_indices()
    trainer = Seq2SeqTrainer(
        model=rot5,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        trainer.save_model(script_args.final_output_dir)
        tokenizer.save_pretrained(script_args.final_output_dir)
    trainer.accelerator.wait_for_everyone()
