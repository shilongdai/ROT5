from multiprocessing import cpu_count
from dataclasses import dataclass, field
from typing import Optional, cast, List, Any, Dict

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    dataset_path: Optional[str] = field(default="JeanKaddour/minipile")
    dataset_subset: Optional[str] = field(default=None)
    dataset_split: Optional[str] = field(default=None)
    eval_amt: Optional[float] = field(default=500)
    model_name: Optional[str] = field(default="roberta-base")
    text_field: Optional[str] = field(default="text")
    model_max_length: Optional[int] = field(default=512)
    cache_dir: Optional[str] = field(default="./transformers_cache")
    final_output_dir: Optional[str] = field(default="./data/pretrain")
    seed: Optional[str] = field(default=42)


def sliding_window(lst: List[Any], max_len: int, overlaps: int):
    return [lst[i:i + max_len] for i in range(0, max_len, max_len - overlaps)]


def chunk_data(batch: Dict[str, List[Any]], tokenizer, max_seq=512, overlaps=256, text_field="text"):
    tokenized_texts = tokenizer(batch[text_field], add_special_tokens=False)
    effective_len = max_seq
    if tokenizer.bos_token_id:
        effective_len -= 1
    if tokenizer.eos_token_id:
        effective_len -= 1
    result = {}
    for k in tokenized_texts:
        result[k] = []
        for items in tokenized_texts[k]:
            window = [*sliding_window(items, effective_len, overlaps)]
            for w in window:
                if k == "input_ids":
                    if tokenizer.bos_token_id:
                        w.insert(0, tokenizer.bos_token_id)
                    if tokenizer.eos_token_id:
                        w.append(tokenizer.eos_token_id)
                elif k == "attention_mask":
                    if tokenizer.bos_token_id:
                        w.insert(0, 1)
                    if tokenizer.eos_token_id:
                        w.append(1)
                elif k == "token_type_ids":
                    if tokenizer.bos_token_id:
                        w.insert(0, 0)
                    if tokenizer.eos_token_id:
                        w.append(0)
                else:
                    raise ValueError()
            result[k].extend(window)
    return result


if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, cache_dir=script_args.cache_dir)
    ds = load_dataset(script_args.dataset_path, script_args.dataset_subset, split=script_args.dataset_split,
                      cache_dir=script_args.cache_dir)
    if script_args.dataset_split:
        ds = ds.train_test_split(test_size=script_args.eval_amt)
        ds = DatasetDict({
            "train": ds["train"],
            "eval": ds["test"]
        })
    print(f"Dataset {script_args.dataset_path}:\n{ds}")

    columns = set()
    for col_list in ds.column_names.values():
        columns.update(col_list)
    ds = ds.shuffle(seed=script_args.seed).map(lambda batch: chunk_data(batch, tokenizer,
                                                                        max_seq=script_args.model_max_length,
                                                                        text_field=script_args.text_field),
                                               batched=True,
                                               num_proc=cpu_count() - 1,
                                               remove_columns=list(columns))
    print(f"Chunked Dataset {script_args.dataset_path}:\n{ds}")
    splits = [s for s in ds.num_rows if len(ds[s]) > 0]
    print(f"Sample Row:\n{tokenizer.decode(ds[splits[0]]['input_ids'][0])}")
    ds.save_to_disk(script_args.final_output_dir)
