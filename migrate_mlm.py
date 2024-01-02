import random
from dataclasses import field, dataclass
from typing import Optional, cast
import gc

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer, HfArgumentParser, AutoModelForMaskedLM

from rebert.init_via_bert import load_transformers_base_bert, load_transformers_base_mlm
from rebert.model import (ReBertConfig, ReBertForMaskedLM)


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="roberta-base")
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

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    src_model = AutoModelForMaskedLM.from_pretrained(script_args.model_name)
    max_length = script_args.model_max_length
    config = ReBertConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=len(tokenizer),
        hidden_size=src_model.config.hidden_size,
        num_hidden_layers=src_model.config.num_hidden_layers,
        num_attention_heads=src_model.config.num_attention_heads,
        intermediate_size=src_model.config.intermediate_size,
        hidden_act=src_model.config.hidden_act,
        hidden_dropout_prob=src_model.config.hidden_dropout_prob,
        attention_probs_dropout_prob=src_model.config.attention_probs_dropout_prob,
        layer_norm_eps=src_model.config.layer_norm_eps,
        classifier_dropout=src_model.config.classifier_dropout,
        max_length=max_length
    )
    rope_bert = ReBertForMaskedLM(config)
    load_transformers_base_bert(src_model.base_model, rope_bert.rebert, config)
    load_transformers_base_mlm(src_model.lm_head, rope_bert.mlm_head)
    del src_model
    gc.collect()

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
    rope_bert.rebert.save_pretrained(script_args.final_output_dir)
    tokenizer.save_pretrained(script_args.final_output_dir)
