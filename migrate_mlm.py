import gc
import random
from dataclasses import field, dataclass
from typing import Optional, cast

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer, HfArgumentParser

from rebert.initialize_via_roberta import load_transformers_base_bert, load_transformers_base_mlm
from rebert.model import (ReBertConfig, ReBertForMaskedLM)


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="roberta-base")
    dataset_path: Optional[str] = field(default="./data/mlm")
    train_name: Optional[str] = field(default="train")
    eval_name: Optional[str] = field(default="eval")
    mlm_prob: Optional[float] = field(default=0.15)
    model_max_length: Optional[int] = field(default=None)
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

    bert_model = AutoModelForMaskedLM.from_pretrained(script_args.model_path, cache_dir=script_args.cache_dir)
    max_length = script_args.model_max_length
    if max_length is None:
        max_length = bert_model.config.max_length
    config = ReBertConfig(
        pad_token_id=bert_model.config.pad_token_id,
        bos_token_id=bert_model.config.bos_token_id,
        eos_token_id=bert_model.config.eos_token_id,
        vocab_size=bert_model.config.vocab_size,
        hidden_size=bert_model.config.hidden_size,
        num_hidden_layers=bert_model.config.num_hidden_layers,
        num_attention_heads=bert_model.config.num_attention_heads,
        intermediate_size=bert_model.config.intermediate_size,
        hidden_act=bert_model.config.hidden_act,
        hidden_dropout_prob=bert_model.config.hidden_dropout_prob,
        attention_probs_dropout_prob=bert_model.config.attention_probs_dropout_prob,
        layer_norm_eps=1e-12,
        classifier_dropout=bert_model.config.classifier_dropout,
        max_length=max_length
    )
    rope_bert = ReBertForMaskedLM(config)
    if bert_model.config.model_type != "rebert":
        load_transformers_base_bert(bert_model.base_model, rope_bert.rebert)
        load_transformers_base_mlm(bert_model.lm_head, rope_bert.mlm_head)
    else:
        rope_bert.load_state_dict(bert_model.state_dict())

    del bert_model
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
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
    rope_bert.save_pretrained(script_args.final_output_dir)
