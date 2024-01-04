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
    tokenizer_name: Optional[str] = field(default="roberta-base")
    text_field: Optional[str] = field(default="text")
    model_max_length: Optional[int] = field(default=512)
    chunk_overlaps: Optional[int] = field(default=0)
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
    result = {
        "input_ids": [],
        "attention_mask": []
    }
    working_ids = []
    working_mask = []

    def add_special():
        if tokenizer.bos_token_id:
            working_ids.insert(0, tokenizer.bos_token_id)
            working_mask.insert(0, 1)
        if tokenizer.eos_token_id:
            working_ids.append(tokenizer.eos_token_id)
            working_mask.append(1)

    for i, _ in enumerate(batch[text_field]):

        if len(working_ids) > 0:
            if tokenizer.eos_token_id:
                working_ids.append(tokenizer.eos_token_id)
                working_mask.append(1)

            # No need to add second EOS token if it's added above
            if len(working_ids) == effective_len:
                if tokenizer.bos_token_id:
                    working_ids.insert(0, tokenizer.bos_token_id)
                    working_mask.insert(0, 1)
                result["input_ids"].append(working_ids)
                result["attention_mask"].append(working_mask)
                working_ids = []
                working_mask = []

        start = 0
        remaining_seq = len(tokenized_texts["input_ids"][i]) - start
        increment = min(effective_len - overlaps, remaining_seq, effective_len - len(working_ids))
        end = start + increment
        while remaining_seq > 0:
            window_ids = tokenized_texts["input_ids"][i][start:end]
            window_mask = tokenized_texts["attention_mask"][i][start:end]
            working_ids.extend(window_ids)
            working_mask.extend(window_mask)

            if len(working_ids) == effective_len:
                add_special()
                result["input_ids"].append(working_ids)
                result["attention_mask"].append(working_mask)
                working_ids = []
                working_mask = []

            start = end
            remaining_seq = len(tokenized_texts["input_ids"][i]) - start
            increment = min(effective_len, remaining_seq, effective_len - len(working_ids))
            end = start + increment

    # Throw away if too short
    assert len(working_ids) <= effective_len
    if len(working_ids) >= 32:
        add_special()
        result["input_ids"].append(working_ids)
        result["attention_mask"].append(working_mask)

    return result


if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, cache_dir=script_args.cache_dir)
    ds = load_dataset(script_args.dataset_path, script_args.dataset_subset, split=script_args.dataset_split,
                      cache_dir=script_args.cache_dir)
    if script_args.dataset_split:
        ds = ds.train_test_split(test_size=script_args.eval_amt)
        ds = DatasetDict({
            "train": ds["train"],
            "validation": ds["test"]
        })
    print(f"Dataset {script_args.dataset_path}:\n{ds}")

    columns = set()
    for col_list in ds.column_names.values():
        columns.update(col_list)
    ds = ds.map(lambda batch: chunk_data(batch, tokenizer,
                                         max_seq=script_args.model_max_length,
                                         overlaps=script_args.chunk_overlaps,
                                         text_field=script_args.text_field),
                batched=True,
                num_proc=8,
                remove_columns=list(columns)).shuffle(seed=script_args.seed)
    print(f"Chunked Dataset {script_args.dataset_path}:\n{ds}")
    splits = [s for s in ds.num_rows if len(ds[s]) > 0]
    print(f"Sample Row:\n{tokenizer.decode(ds[splits[0]]['input_ids'][0])}")
    ds.save_to_disk(script_args.final_output_dir)
