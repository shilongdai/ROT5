import random
from dataclasses import field, dataclass
from typing import Optional, cast

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, \
    HfArgumentParser, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizer, T5Config, \
    T5ForConditionalGeneration

from rebert.model import (ReBertConfig, ReBertForConditionalGeneration)
from text_denoising import DataCollatorForUL2


@dataclass
class ScriptArguments:
    tokenizer_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    dataset_path: Optional[str] = field(default="./data/mlm")
    train_name: Optional[str] = field(default="train")
    eval_name: Optional[str] = field(default="eval")
    eval_sample: Optional[int] = field(default=500)
    model_max_length: Optional[int] = field(default=512)
    sentinel_tokens: Optional[int] = field(default=500)
    cache_dir: Optional[str] = field(default="./transformers_cache")
    final_output_dir: Optional[str] = field(default="./best_migrated_model")


if __name__ == "__main__":
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, padding_side="right")
    for i in range(script_args.sentinel_tokens):
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"[MASK-{script_args.sentinel_tokens - i - 1}]"]})
    print(f"Added {script_args.sentinel_tokens} MASK tokens")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[REBERT]"]})
    sink_token = tokenizer.encode("[REBERT]", add_special_tokens=False)[0]
    print(f"Added {tokenizer.decode(sink_token)}: {sink_token} as decoder start sink token")
    if not tokenizer.pad_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"Added Pad Token: {tokenizer.pad_token_id}")
    print(f"Final Vocab Size: {len(tokenizer)}")
    max_length = script_args.model_max_length
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=768,
        d_ff=3072,
        d_kv=64,
        num_layers=12,
        num_heads=12,
        feed_forward_proj="gelu",
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    t5 = T5ForConditionalGeneration(config)

    if train_args.gradient_checkpointing:
        t5.config.use_cache = False

    def sentinel_from_end(ids: np.ndarray, max_bound: int):
        return max_bound - ids

    data_collator = DataCollatorForUL2(tokenizer=tokenizer,
                                       decoder_start_token_id=sink_token,
                                       sentinel_map=lambda x: sentinel_from_end(x, sink_token))
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
        model=t5,
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
