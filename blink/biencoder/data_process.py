# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import multiprocessing as mp

import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from blink.common.params import BERT_START_TOKEN, BERT_END_TOKEN

def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )
    context_tokens = context_tokens[0: max_seq_length - 2]
    context_tokens = [BERT_START_TOKEN] + context_tokens + [BERT_END_TOKEN]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data


def process_mention_data(
    sample,
    tokenizer,
    max_context_length,
    max_cand_length,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
):
    context_tokens = get_context_representation(
        sample,
        tokenizer,
        max_context_length,
        mention_key,
        context_key,
        ent_start_token,
        ent_end_token,
    )

    label = sample[label_key]
    title = sample.get(title_key, None)
    label_tokens = get_candidate_representation(
        label, tokenizer, max_cand_length, title,
    )
    label_idx = int(sample["label_id"])

    record = {
        "context_tokens": context_tokens["tokens"],
        "context_ids": context_tokens["ids"],
        "label_tokens": label_tokens["tokens"],
        "label_ids": label_tokens["ids"],
        "label_idx": label_idx,
    }

    if "world" in sample:
        src = sample["world"]
        src = world_to_id[src]
        record["src"] = src

    return record


from typing import Dict, List, Optional, TypedDict, Iterable

from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils import PaddingStrategy


class MentionInput(TypedDict):
    mention: str
    context_left: str
    context_right: str

class EntityInput(TypedDict):
    label: str
    label_title: str


class ProcessInput(MentionInput, EntityInput):
    label_id: int
    world: Optional[str]


def tokenize_mention(
    sample: MentionInput,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    ent_start_token="[unused0]",
    ent_end_token="[unused1]",
    padding_strategy: PaddingStrategy = PaddingStrategy.MAX_LENGTH,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[str] = None,
    return_attention_mask: Optional[bool] = None,
    return_token_type_ids: Optional[bool] = None
) -> BatchEncoding:

    # for sample in samples:
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

    mention_tokens = [ent_start_token] + mention_tokens[:num_mention_tokens] + [ent_end_token]
    context_tokens = (
        context_left[len(context_left)-num_tokens_on_left:]
        + mention_tokens
        + context_right[:num_tokens_on_right]
    )
    context_tokens = [tokenizer.cls_token] + context_tokens + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)

    tokenizer_output = tokenizer.prepare_for_model(
        input_ids,
        padding=padding_strategy.value,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        prepend_batch_axis=True,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        add_special_tokens=False
    )

    return tokenizer_output


def tokenize_entity(    
    sample: EntityInput, 
    tokenizer: PreTrainedTokenizer, 
    max_length: int, 
    title_token='[unused2]',
    padding_strategy: PaddingStrategy = PaddingStrategy.MAX_LENGTH,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[str] = None,
    return_attention_mask: Optional[bool] = True,
    return_token_type_ids: Optional[bool] = True
) -> BatchEncoding:
    input_string = f"{sample['label_title']} {title_token} {sample['label']}"
    model_inputs = tokenizer(
        input_string, 
        max_length=max_length,
        padding=padding_strategy.value,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
    )
    return model_inputs


def tokenize_all(
    sample: ProcessInput,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int,
    max_cand_length: int,
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
):
    mentions = tokenize_mention(sample, tokenizer, max_context_length, ent_start_token=ent_start_token, ent_end_token=ent_end_token)
    entities = tokenize_entity(sample, tokenizer, max_cand_length, title_token=title_token)

    mentions = {f'mention_{key}': value for key, value in mentions.items()}
    entities = {f'entity_{key}': value for key, value in entities.items()}


    label_idx = sample["label_id"]

    record =  mentions | entities | {'label_idx': label_idx}

    if "world" in sample:
        src = sample["world"]
        src = world_to_id[src]
        record["src"] = src

    return record

