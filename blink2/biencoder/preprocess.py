# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import multiprocessing as mp
from typing import Any, Dict, List, Optional, TypedDict, Iterable

import torch
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy


class MentionInput(TypedDict):
    mention: str
    context_left: str
    context_right: str

class EntityInput(TypedDict):
    title: str
    text: str


class ProcessInput(MentionInput, EntityInput):
    label_id: int
    world: Optional[str]


def tokenize_mention(
    sample: MentionInput,
    tokenizer: PreTrainedTokenizer,
    mention_start_token="[unused0]",
    mention_end_token="[unused1]",

    padding: bool | str | PaddingStrategy = 'max_length',
    truncation: bool | str | TruncationStrategy = 'only_first',
    max_length: int | None = None,
    stride: int = 0,
    pad_to_multiple_of: int | None = None,
    return_tensors: str | torch.TensorType | None = None,
    return_token_type_ids: bool | None = None,
    return_attention_mask: bool | None = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    prepend_batch_axis: bool = False,
    **kwargs: Any
) -> BatchEncoding:
    
    mention_tokens = tokenizer.tokenize(sample['mention'])
    context_left = tokenizer.tokenize(sample['context_left'])
    context_right = tokenizer.tokenize(sample['context_right'])

    # Truncate mention tokens
    num_mention_tokens = min(len(mention_tokens), max_length-4)
    tokens_remaining = max_length - 4 - num_mention_tokens

    left_extra = max(0, tokens_remaining // 2 - len(context_left))
    right_extra = max(0, tokens_remaining // 2 - len(context_right))

    num_tokens_on_left = tokens_remaining // 2 + right_extra - left_extra
    num_tokens_on_right = tokens_remaining // 2 + left_extra - right_extra

    mention_tokens = [mention_start_token] + mention_tokens[:num_mention_tokens] + [mention_end_token]
    context_tokens = (
        context_left[len(context_left)-num_tokens_on_left:]
        + mention_tokens
        + context_right[:num_tokens_on_right]
    )
    context_tokens = context_tokens

    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)

    tokenizer_output = tokenizer.prepare_for_model(
        input_ids,
        padding=padding,
        truncation = truncation,
        max_length = max_length,
        stride = stride,
        pad_to_multiple_of = pad_to_multiple_of,
        return_tensors = return_tensors,
        return_token_type_ids = return_token_type_ids,
        return_attention_mask = return_attention_mask,
        return_overflowing_tokens = return_overflowing_tokens,
        return_special_tokens_mask = return_special_tokens_mask,
        return_offsets_mapping = return_offsets_mapping,
        return_length = return_length,
        verbose = verbose,
        prepend_batch_axis = prepend_batch_axis,
        **kwargs
    )

    return tokenizer_output


def tokenize_entity(    
    sample: EntityInput, 
    tokenizer: PreTrainedTokenizer, 
    entity_sep_token='[unused2]',
    padding: bool | str | PaddingStrategy = 'max_length',
    truncation: bool | str | TruncationStrategy = 'only_first',
    max_length: int | None = None,
    stride: int = 0,
    pad_to_multiple_of: int | None = None,
    return_tensors: str | torch.TensorType | None = None,
    return_token_type_ids: bool | None = None,
    return_attention_mask: bool | None = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs: Any
) -> BatchEncoding:
    input_string = f"{sample['title']} {entity_sep_token} {sample['text']}"
    model_inputs = tokenizer(
        input_string, 
        padding=padding,
        truncation = truncation,
        max_length = max_length,
        stride = stride,
        pad_to_multiple_of = pad_to_multiple_of,
        return_tensors = return_tensors,
        return_token_type_ids = return_token_type_ids,
        return_attention_mask = return_attention_mask,
        return_overflowing_tokens = return_overflowing_tokens,
        return_special_tokens_mask = return_special_tokens_mask,
        return_offsets_mapping = return_offsets_mapping,
        return_length = return_length,
        verbose = verbose,
        **kwargs
    )
    return model_inputs


def tokenize_all(
    sample: ProcessInput,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int,
    max_cand_length: int,
    mention_start_token: str,
    mention_end_token: str,
    entity_sep_token: str,
):
    mentions = tokenize_mention(sample, tokenizer, mention_start_token=mention_start_token, mention_end_token=mention_end_token, max_length=max_context_length)
    entities = tokenize_entity(sample, tokenizer, entity_sep_token=entity_sep_token, max_length=max_cand_length)

    mentions = {f'mention_{key}': value for key, value in mentions.items()}
    entities = {f'entity_{key}': value for key, value in entities.items()}

    label_idx = sample["label_id"]

    record = mentions | entities | {'label_idx': label_idx}

    return record

