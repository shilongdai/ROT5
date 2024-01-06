import random
from dataclasses import field, dataclass
from typing import Optional, cast

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer, HfArgumentParser, AutoModelForMaskedLM

from rebert.model import (ReBertConfig, ReBertForMaskedLM)


@dataclass
class ScriptArguments:
    tokenizer_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    model_path: Optional[str] = field(default=None)
    dataset_path: Optional[str] = field(default="./data/mlm")
    train_name: Optional[str] = field(default="train")
    eval_name: Optional[str] = field(default="eval")
    mlm_prob: Optional[float] = field(default=0.15)
    model_max_length: Optional[int] = field(default=512)
    cache_dir: Optional[str] = field(default="./transformers_cache")
    final_output_dir: Optional[str] = field(default="./best_migrated_model")


if __name__ == "__main__":
    parser = HfArgumentParser([TrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    if not script_args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
        if not tokenizer.pad_token_id:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print(f"Added Pad Token: {tokenizer.pad_token_id}")
        if not tokenizer.mask_token_id:
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            print(f"Added Mask Token: {tokenizer.mask_token_id}")

        print(f"Final Vocab Size: {len(tokenizer)}")
        max_length = script_args.model_max_length
        config = ReBertConfig(
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            vocab_size=len(tokenizer),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            classifier_dropout=0.1,
            max_length=max_length
        )
        rope_bert = ReBertForMaskedLM(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
        rope_bert = AutoModelForMaskedLM.from_pretrained(script_args.model_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=script_args.mlm_prob, mlm=True)
    print(f"Loading data from: {script_args.dataset_path}")
    ds = load_from_disk(script_args.dataset_path)
    train_set = ds[script_args.train_name]
    eval_set = None
    if script_args.eval_name in ds:
        eval_set = ds[script_args.eval_name]
    trainer = Trainer(
        model=rope_bert,
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
