import random
from dataclasses import dataclass, field
from typing import Optional, cast

from datasets import load_from_disk
from transformers import HfArgumentParser, PreTrainedTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import torch

from text_denoising import DataCollatorForUL2
import rot5


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
                                          per_device_eval_batch_size=script_args.per_device_eval_batch_size,
                                          local_rank=script_args.local_rank)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_path)
    model.config.output_router_logits = False


    def sentinel_from_end(ids: np.ndarray, max_bound: int):
        return max_bound - ids


    data_collator = DataCollatorForUL2(tokenizer=tokenizer,
                                       decoder_start_token_id=model.config.decoder_start_token_id,
                                       sentinel_map=lambda x: sentinel_from_end(x, model.config.decoder_start_token_id))
    print(f"Loading data from: {script_args.dataset_path}")
    ds = load_from_disk(script_args.dataset_path)
    eval_set = ds[script_args.eval_name]
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        eval_dataset=eval_set,
        data_collator=data_collator,
    )
    print(trainer.evaluate())
