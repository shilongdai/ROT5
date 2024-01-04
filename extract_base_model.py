from dataclasses import field, dataclass
from typing import cast

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForMaskedLM
from rebert.model import *


@dataclass
class ScriptArguments:
    model_path: str = field()
    output_path: str = field(default="rebert-base")
    component: str = field(default="rebert")


if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    mlm_model = AutoModelForMaskedLM.from_pretrained(script_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)

    core_model = getattr(mlm_model, script_args.component)
    core_model.save_pretrained(script_args.output_path)
    tokenizer.save_pretrained(script_args.output_path)
