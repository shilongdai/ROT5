import random
from dataclasses import field, dataclass
from typing import Optional, cast

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, \
    HfArgumentParser, Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizer

from rot5 import (ROT5Config, ROT5ForConditionalGeneration)
from text_denoising import DataCollatorForUL2


@dataclass
class ScriptArguments:
    tokenizer_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    model_path: Optional[str] = field(default=None)
    dataset_path: Optional[str] = field(default="./data/mlm")
    train_name: Optional[str] = field(default="train")
    eval_name: Optional[str] = field(default="eval")
    eval_sample: Optional[int] = field(default=500)
    model_max_length: Optional[int] = field(default=512)
    sentinel_tokens: Optional[int] = field(default=500)
    kv_heads: Optional[int] = field(default=12)
    num_experts: Optional[int] = field(default=1)
    topk_experts: Optional[int] = field(default=1)
    cache_dir: Optional[str] = field(default="./transformers_cache")
    final_output_dir: Optional[str] = field(default="./best_migrated_model")
    aux_loss: Optional[bool] = field(default=False)

    d_model: Optional[int] = field(default=768)
    d_ff: Optional[int] = field(default=3072)
    d_kv: Optional[int] = field(default=64)
    num_layers: Optional[int] = field(default=12)
    num_heads: Optional[int] = field(default=12)



if __name__ == "__main__":
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    if not script_args.model_path:
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
        config = ROT5Config(
            vocab_size=len(tokenizer),
            d_model=script_args.d_model,
            d_ff=script_args.d_ff,
            d_kv=script_args.d_kv,
            num_layers=script_args.num_layers,
            num_heads=script_args.num_heads,
            num_key_value_heads=script_args.kv_heads,
            feed_forward_proj="gelu",
            num_local_experts=script_args.num_experts,
            num_experts_per_tok=script_args.topk_experts,
            output_router_logits=script_args.aux_loss,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=sink_token
        )
        rot5 = ROT5ForConditionalGeneration(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
        rot5 = ROT5ForConditionalGeneration.from_pretrained(script_args.model_path)
        sink_token = rot5.config.decoder_start_token_id

    if train_args.gradient_checkpointing:
        rot5.config.use_cache = False

    num_params = sum(p.numel() for p in rot5.parameters() if p.requires_grad)
    print(f"Num Params: {num_params}")


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
