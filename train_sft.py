import random
from dataclasses import field, dataclass
from typing import Optional, cast

import nltk
import numpy as np
import torch
from datasets import load_from_disk
from evaluate import load
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, TrainingArguments, \
    HfArgumentParser, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from rot5 import (ROT5ForConditionalGeneration)


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="./t5-rgqa")
    dataset_path: Optional[str] = field(default="./data/mlm")
    train_name: Optional[str] = field(default="train")
    eval_name: Optional[str] = field(default="eval")
    eval_sample: Optional[int] = field(default=500)
    cache_dir: Optional[str] = field(default="./transformers_cache")
    final_output_dir: Optional[str] = field(default="./best_instruct_model")
    aux_loss: Optional[bool] = field(default=False)


def create_summarization_metrics(tokenizer):
    metric = load("rouge")
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('stopwords')
    lemmatizer = WordNetLemmatizer()
    stop_words = set([w.lower() for w in stopwords.words("english")])

    def clean_sentence(sentence):
        nltk_words = nltk.word_tokenize(sentence)
        nltk_words = [''.join(x for x in w if x.isalpha()) for w in nltk_words]
        lem_words = [lemmatizer.lemmatize(w) for w in nltk_words if w not in stop_words and len(w.strip()) > 0]
        return " ".join(lem_words)

    def clean_sentences(sentences):
        result = []
        for s in sentences:
            result.append(clean_sentence(s))
        return result

    def compute_rouge_metrics(labels, predicted):
        # Rouge expects a newline after each sentence
        decoded_labels = ["\n".join(clean_sentences(nltk.sent_tokenize(label.lower()))) for label in labels]
        decoded_preds = ["\n".join(clean_sentences(nltk.sent_tokenize(pred.lower()))) for pred in predicted]

        # Note that other metrics may not have a `use_aggregator` parameter
        # and thus will return a list, computing a metric for each sentence.
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False,
                                use_aggregator=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        return {k: round(v, 4) for k, v in result.items()}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Replace -100 in the labels as we can't decode them.
        predictions = np.where(predictions != -100, predictions, tokenizer.bos_token_id)
        labels = np.where(labels != -100, labels, tokenizer.bos_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_labels = []
        rouge_preds = []

        for l, p in zip(decoded_labels, decoded_preds):
            rouge_labels.append(l.lower().strip())
            rouge_preds.append(l.lower().strip())

        result = {}
        result.update(compute_rouge_metrics(rouge_labels, rouge_preds))
        return result

    return compute_metrics


if __name__ == "__main__":
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    rot5 = ROT5ForConditionalGeneration.from_pretrained(script_args.model_path)
    sink_token = rot5.config.decoder_start_token_id

    if train_args.gradient_checkpointing:
        rot5.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=rot5)
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
        compute_metrics=create_summarization_metrics(tokenizer),
        eval_dataset=eval_set,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        trainer.save_model(script_args.final_output_dir)
        tokenizer.save_pretrained(script_args.final_output_dir)
    trainer.accelerator.wait_for_everyone()
