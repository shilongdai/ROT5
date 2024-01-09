import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from torch.nn import functional as F
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from text_denoising.utils import random_spans_noise_mask


@dataclass
class DataCollatorForUL2(DataCollatorMixin):
    """

    Data collator used for UL2. Taken from https://github.com/theblackcat102/unify-learning-paradigms with adjustments
    to attention mask and padding.

    """
    tokenizer: PreTrainedTokenizerBase
    r_denoising: bool = True
    r_probability: float = 0.25
    r_denoising_config: Tuple[Tuple] = ((3, 0.15),)
    s_denoising: bool = True
    s_probability: float = 0.5
    x_denoising: bool = True
    x_probability: float = 0.25
    x_denoising_config: Tuple[Tuple] = ((32, 0.5), (64, 0.2))
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    label_pad_token_id: int = -100
    decoder_start_token_id: int = -1
    sentinel_map: Callable[[np.ndarray], np.ndarray] = lambda x: x

    def __post_init__(self):
        self.total_task = [0, 1, 2]
        task_prob = []
        task_prob.append(self.r_probability if self.r_denoising else 0.0)
        task_prob.append(self.s_probability if self.s_denoising else 0.0)
        task_prob.append(self.x_probability if self.x_denoising else 0.0)
        self.task_prob = task_prob
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.decoder_start_token_id == -1:
            print("Assuming decoder start token id is the pad token")
            self.decoder_start_token_id = self.tokenizer.pad_token_id

        assert sum(task_prob) == 1.0

    def assign_task_type(self, batch_size: int):
        '''
            Randomly assign S,R,X to each sentence based on weighted prob
        '''
        return random.choices(self.total_task, weights=self.task_prob, k=batch_size)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # print(examples)
        task_ids = self.assign_task_type(len(examples))
        task_type = torch.tensor(task_ids)
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer)
            }
        lengths = torch.tensor([len(e['input_ids']) for e in examples], dtype=torch.long)
        max_length = batch['input_ids'].shape[-1]

        new_batch = {
            "input_ids": np.zeros(shape=batch["input_ids"].shape[0], dtype=object),
            "labels": np.zeros(shape=batch["input_ids"].shape[0], dtype=object)
        }

        _, expanded_length = batch['input_ids'].shape
        input_ids = batch["input_ids"]
        r_denoising_idx = task_type == 0
        if r_denoising_idx.any():
            mask_indices = self.create_mask(lengths, r_denoising_idx, self.r_denoising_config)
            sub_input_ids = input_ids[r_denoising_idx]

            new_batch['input_ids'][r_denoising_idx], new_batch['labels'][
                r_denoising_idx] = self.create_inputs_from_mask(mask_indices, sub_input_ids)

        s_denoising_idx = task_type == 1
        if s_denoising_idx.any():
            offsets = torch.zeros_like(lengths)
            if self.tokenizer.padding_side == "left":
                offsets = max_length - lengths
            sub_input_ids = input_ids[s_denoising_idx]
            _labels = np.zeros(shape=len(sub_input_ids), dtype=object)
            _input_ids = np.zeros(shape=len(sub_input_ids), dtype=object)
            start = offsets
            middle = offsets + torch.maximum(lengths // 2, torch.tensor(2))
            end = offsets + lengths
            for i, (ids, s, m, e) in enumerate(zip(sub_input_ids, start, middle, end)):
                _input_ids[i] = ids[s:m]
                _labels[i] = ids[m:e]

            new_batch['input_ids'][s_denoising_idx] = np.array(_input_ids, dtype=object)
            new_batch['labels'][s_denoising_idx] = np.array(_labels, dtype=object)

        x_denoising_idx = task_type == 2
        if x_denoising_idx.any():
            mask_indices = self.create_mask(lengths, x_denoising_idx, self.x_denoising_config)
            sub_input_ids = input_ids[x_denoising_idx]

            new_batch['input_ids'][x_denoising_idx], new_batch['labels'][
                x_denoising_idx] = self.create_inputs_from_mask(mask_indices, sub_input_ids)

        return self.prepare_input(new_batch)

    def create_mask(self, lengths, idx, config):
        sub_lengths = lengths[idx]
        max_length = int(lengths.max())
        diff = max_length - sub_lengths
        mask_indices = np.zeros((len(sub_lengths), max_length), dtype=bool)
        # union of different denoising settings
        for (mean_span, noise) in config:
            _mask_indices = [
                random_spans_noise_mask(l, mean_span, noise) for l in lengths[idx]
            ]
            _mask_indices = np.array([self.pad_mask_row(m, diff[i], False) for i, m in enumerate(_mask_indices)])
            mask_indices = mask_indices | _mask_indices

        return mask_indices

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        result = np.zeros(shape=len(input_ids_full), dtype=object)
        for i, row in enumerate(input_ids_full):
            result[i] = row[row >= 0]
        return result

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, self.sentinel_map(sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def create_inputs_from_mask(self, mask_indices, input_ids):
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(int))
        labels_mask = ~mask_indices
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(int))
        _sub_input_ids = self.filter_input_ids(input_ids, input_ids_sentinel)
        _labels = self.filter_input_ids(input_ids, labels_sentinel)
        return _sub_input_ids, _labels

    def pad_mask_row(self, input_ids, diff, padding):
        if self.tokenizer.padding_side == "right":
            input_ids = np.pad(input_ids, (0, diff), 'constant', constant_values=padding)
        elif self.tokenizer.padding_side == "left":
            input_ids = np.pad(input_ids, (diff, 0), 'constant', constant_values=padding)
        else:
            raise ValueError("Padding side must be left or right")
        return input_ids

    def pad_torch(self, input_ids, diff, padding):
        if self.tokenizer.padding_side == "right":
            input_ids = F.pad(input_ids, (0, diff), 'constant', padding)
        elif self.tokenizer.padding_side == "left":
            input_ids = F.pad(input_ids, (diff, 0), 'constant', padding)
        else:
            raise ValueError("Padding side must be left or right")
        return input_ids

    def prepare_input(self, batch):
        batch["labels"] = self.tokenizer.pad({"input_ids": batch["labels"].tolist()},
                                             pad_to_multiple_of=self.pad_to_multiple_of,
                                             return_attention_mask=False, return_tensors="pt")["input_ids"]
        batch["input_ids"] = self.tokenizer.pad({"input_ids": batch["input_ids"].tolist()},
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                return_attention_mask=False, return_tensors="pt")["input_ids"]
        batch["labels"][batch["labels"] == self.pad_token_id] = self.label_pad_token_id
        shifted_labels = batch["labels"].new_zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].clone()
        shifted_labels[..., 0] = self.decoder_start_token_id  # decoder_start_token_id

        batch["decoder_input_ids"] = torch.masked_fill(
            shifted_labels,
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            shifted_labels == self.label_pad_token_id,
            0,
            torch.ones_like(shifted_labels),
        )
        batch["attention_mask"] = torch.where(
            batch['input_ids'] == self.pad_token_id,
            0,
            torch.ones_like(batch['input_ids']),
        )
        return batch
