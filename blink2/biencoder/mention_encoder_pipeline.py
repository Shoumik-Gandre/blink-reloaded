from typing import Dict, List, Optional, TypedDict, Iterable

from transformers.pipelines import Pipeline, PIPELINE_REGISTRY
from transformers.pipelines.base import build_pipeline_init_args, GenericTensor
from transformers.utils import add_end_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import PreTrainedTokenizer, BatchEncoding, AutoModel
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


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True, supports_binary_output=False),
    r"""
        tokenize_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.""",
)
class MentionEncoderPipeline(Pipeline):
    """
    Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> extractor = pipeline(model=shomez/blink-biencoder-mention-encoder", task="mention-encoder")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, hidden_dimension] representing the input string.
    torch.Size([1, 1024])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    """

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, max_length=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation
            tokenize_kwargs["max_length"] = max_length

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: MentionInput | List[MentionInput], **tokenize_kwargs) -> Dict[str, GenericTensor]:
        model_inputs = mention_tokenize(
            inputs,
            tokenizer=self.tokenizer,
            return_tensors=self.framework,
            **tokenize_kwargs
        )
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs: BaseModelOutputWithPooling, return_tensors=False):
        if return_tensors:
            return model_outputs.last_hidden_state[:, 0, :].squeeze(0)
        if self.framework == "pt":
            return model_outputs.last_hidden_state[:, 0, :].squeeze(0).tolist()
        elif self.framework == "tf":
            return model_outputs.last_hidden_state[:, 0, :].numpy().tolist()

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (`str` or `List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)


def register_mention_encoder_pipeline():
    PIPELINE_REGISTRY.register_pipeline(
        "mention-encoder",
        pipeline_class=MentionEncoderPipeline,
        pt_model=AutoModel,
        default={
            'model': {
                "pt": ('shomez/blink-biencoder-mention-encoder', "")
            }
        },
        type="text",  # current support type: text, audio, image, multimodal
    )

register_mention_encoder_pipeline()