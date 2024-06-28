from typing import Dict, List, Optional, TypedDict, Iterable

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy


class MentionInput(TypedDict):
    mention: str
    context_left: str
    context_right: str


def mention_tokenize(
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
) -> List[str]:

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