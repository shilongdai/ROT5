import random
import re
from dataclasses import dataclass, field
from typing import Optional, cast, Dict, List, Any

import numpy as np
import torch
from datasets import load_dataset
from datasets import Dataset
from transformers import HfArgumentParser, AutoTokenizer, PreTrainedTokenizer


@dataclass
class ScriptArguments:
    tokenizer_name: Optional[str] = field(default="roberta-base")
    sample: Optional[int] = field(default=None)
    cache_dir: Optional[str] = field(default="./transformers_cache")
    final_output_dir: Optional[str] = field(default="./data/instruct")
    seed: Optional[int] = field(default=42)


DATA_PATH = "skeskinen/TinyStories-Instruct-hf"


def split_input_output(batch: Dict[str, List[Any]], tokenizer: PreTrainedTokenizer):
    inputs = []
    outputs = []
    preamble_match = re.compile(r"\s*\w+:")
    for entry in batch["text"]:
        if "Summary:" not in entry:
            continue

        current_story = []
        current_summary = []
        story_mode = False
        summary_mode = False
        for l in entry.split("\n"):
            matcher = preamble_match.match(l)
            if matcher:
                if "Story" in matcher.group():
                    story_mode = True
                    summary_mode = False
                elif "Summary" in matcher.group():
                    story_mode = False
                    summary_mode = True
                else:
                    story_mode = False
                    summary_mode = False

            if story_mode:
                current_story.append(l.strip())
            if summary_mode:
                l = l.replace("Summary:", "").strip()
                current_summary.append(l)
        inputs.append(" ".join(current_story) + "\nSummary:")
        outputs.append(" ".join(current_summary))

    input_map = tokenizer(inputs)
    labels = tokenizer(outputs, add_special_tokens=False)["input_ids"]
    input_map["labels"] = [l + [tokenizer.eos_token_id] for l in labels]
    return input_map


if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, cache_dir=script_args.cache_dir)
    ds = load_dataset(DATA_PATH, cache_dir=script_args.cache_dir)
    print(ds)

    if script_args.sample:
        ds["train"] = Dataset.from_dict(ds["train"].shuffle(seed=script_args.seed)[:script_args.sample])

    columns = set()
    for col_list in ds.column_names.values():
        columns.update(col_list)
    ds = ds.map(lambda batch: split_input_output(batch, tokenizer),
                batched=True,
                num_proc=8,
                remove_columns=list(columns)).shuffle(seed=script_args.seed)
    print(f"Split Dataset:\n{ds}")
    splits = [s for s in ds.num_rows if len(ds[s]) > 0]
    sample_row = ds["train"][0]
    print(f"Sample Input:\n{tokenizer.decode(sample_row['input_ids'])}")
    print(f"Sample Label:\n{tokenizer.decode(sample_row['labels'])}")
    ds.save_to_disk(script_args.final_output_dir)
