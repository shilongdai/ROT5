import random
from dataclasses import dataclass, field
from typing import Optional, cast

import numpy as np
import torch
from datasets import load_from_disk
from transformers import HfArgumentParser, PreTrainedTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

import rot5
from train_sft import create_summarization_metrics


@dataclass
class ScriptArguments:
    deepspeed: Optional[str] = field(default=None)
    model_path: Optional[str] = field(default=None)
    dataset_path: Optional[str] = field(default="./data/mlm")
    eval_name: Optional[str] = field(default="eval")
    cache_dir: Optional[str] = field(default="./transformers_cache")
    per_device_eval_batch_size: Optional[int] = field(default=32)
    local_rank: Optional[int] = field(default=0)
    seed: Optional[int] = field(default=42)


if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)

    train_args = Seq2SeqTrainingArguments(output_dir="output",
                                          deepspeed=script_args.deepspeed,
                                          predict_with_generate=True,
                                          per_device_eval_batch_size=script_args.per_device_eval_batch_size,
                                          local_rank=script_args.local_rank)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_path)
    model.config.output_router_logits = False
    model.generation_config.max_new_tokens = 128

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=rot5)
    print(f"Loading data from: {script_args.dataset_path}")
    ds = load_from_disk(script_args.dataset_path)
    eval_set = ds[script_args.eval_name]
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=create_summarization_metrics(tokenizer)
    )
    print(trainer.evaluate())
